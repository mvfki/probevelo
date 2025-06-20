import pandas as pd
import pysam
from typing import Union
from .probe_set import probe_set
import multiprocessing as mp
import tqdm
from scipy.sparse import coo_matrix
import anndata


class bam_parser:
    """
    A class to parse BAM files and count spliced and unspliced reads.
    This class uses the pysam library to read BAM files and extract
    information about cell barcodes and gene regions.

    Attributes
    ----------
    cell_map : dict
        A dictionary mapping cell barcodes to their corresponding indices in
        the final count matrices.

    Methods
    -------
    __init__(file: str)
        Initializes the BamParser with a BAM file.
    count_splice(probe_set: probe_set, n_thread: int = 1, adata: bool = False)
        Count spliced and unspliced reads to cell by gene CSR sparse matrices.
    """
    def __init__(self, file: str, n_thread: int = 1, chunk_size: int = 100000):
        """
        Initializes the BamParser with a BAM file.

        Parameters
        ----------
        file : str
            The path to the BAM file to be parsed.
        n_thread : int, optional
            Number of threads to use for parallel processing, by default 1.
        chunk_size : int, optional
            Size of the chunks to process reads in parallel, by default 100000.
        """
        self.file = file
        if n_thread < 1:
            Warning(
                f"Requested {n_thread} threads, but it must be at least 1. " +
                "Using 1 thread."
            )
            n_thread = 1

        if n_thread > mp.cpu_count():
            Warning(
                f"Requested {n_thread} threads, but only {mp.cpu_count()} " +
                "are available. Using all available threads."
            )
            n_thread = mp.cpu_count()

        # Initialize the chunking parameters
        self.n_thread = n_thread
        self.chunk_size = chunk_size
        self.n_reads = 0
        with pysam.AlignmentFile(file, "rb") as bam:
            self.n_reads = bam.count()
        print(f"Found {self.n_reads} reads in the BAM file.")
        self.ranges = [(i, min(i + chunk_size, self.n_reads))
                       for i in range(0, self.n_reads, chunk_size)]

        with mp.Pool(self.n_thread) as pool:
            cb_per_region = list(tqdm.tqdm(
                pool.imap(
                    self._collect_chunk_cb, self.ranges
                ),
                total=len(self.ranges),
                desc="Collecting cell barcodes from chunks",
                unit="chunks"
            ))

        cell_barcodes = set()
        for region in cb_per_region:
            # region is a set of barcodes. Now update the union
            cell_barcodes = cell_barcodes.union(region)

        self.cell_map = pd.DataFrame(
            index=list(cell_barcodes),
            data={
                "index": range(len(cell_barcodes))
            }
        )
        print(f"Found {self.cell_map.shape[0]} unique cell barcodes in " +
              "the BAM file.")

    def _collect_chunk_cb(self, chunk_range):
        start, end = chunk_range
        cb = set()
        with pysam.AlignmentFile(self.file, "rb") as bam:
            for i, read in enumerate(bam.fetch(until_eof=True)):
                if i < start:
                    continue
                if i >= end:
                    break
                if read.is_unmapped or \
                        read.is_secondary or \
                        read.is_supplementary or \
                        not read.has_tag("CB"):
                    continue
                cb.add(read.get_tag("CB"))
        return cb

    def count_splice(
            self,
            probe_set: probe_set,
            adata: bool = False
            ) -> Union[tuple, anndata.AnnData]:
        """
        Count spliced and unspliced reads to cell by gene CSR sparse matrices.

        Parameters
        ----------
        probe_set : probe_set, required
            The probe set for mapping reads to spliced or unspliced.
        adata : bool, optional
            If True, return an AnnData object with the counts, by default
            False.

        Returns
        -------
        Union[tuple, AnnData]
            If adata is False, returns a tuple of (spliced_counts,
            unspliced_counts, cell_barcodes, gene_names). If adata is True,
            returns an AnnData object with the counts.
        """
        # Prepare arguments for parallel processing
        # args = [
        #     (self.file, region, probe_set)
        #     for region in self.regions
        # ]
        args = [
            (chunk_range, probe_set)
            for chunk_range in self.ranges
        ]
        with mp.Pool(self.n_thread) as pool:
            results = list(tqdm.tqdm(
                pool.imap(
                    self._process_chunk, args
                ),
                total=len(self.ranges),
                desc="Processing chunks",
                unit="chunks"
            ))
        # Use multiprocessing to process regions in parallel
        # with mp.Pool(processes=self.n_thread) as pool:
        #     results = list(tqdm.tqdm(
        #         pool.imap(self._process_region, args),
        #         total=len(args),
        #         desc="Processing regions",
        #         unit="regions"
        #     ))

        # Combine results from all regions
        print("Merging results...")
        spliced_merge = set()
        unspliced_merge = set()
        for spliced, unspliced in results:
            spliced_merge = spliced_merge.union(spliced)
            unspliced_merge = unspliced_merge.union(unspliced)

            # for key, count in spliced.items():
            #     if key not in spliced_merge:
            #         spliced_merge[key] = 0
            #     spliced_merge[key] += count
            # for key, count in unspliced.items():
            #     if key not in unspliced_merge:
            #         unspliced_merge[key] = 0
            #     unspliced_merge[key] += count
        print("Making sparse matrix for spliced counts...")
        spliced_csr, cells_1, genes_1 = self._make_csr(
            spliced_merge,
            probe_set
            )
        print("Making sparse matrix for unspliced counts...")
        unspliced_csr, cells_2, genes_2 = self._make_csr(
            unspliced_merge,
            probe_set
            )
        assert cells_1 == cells_2, "Cell barcodes do not match between spliced and unspliced counts."
        assert genes_1 == genes_2, "Gene names do not match between spliced and unspliced counts."
        if not adata:
            return spliced_csr, unspliced_csr, cells_1, genes_1
        else:
            adata = anndata.AnnData(
                X=spliced_csr + unspliced_csr,
                obs=pd.DataFrame(index=cells_1),
                var=pd.DataFrame(index=genes_1,
                                 data={
                                     'gene_name':
                                     probe_set.gene_map.loc[genes_1,
                                                            'gene_name']
                                 }),
                layers={
                    'spliced': spliced_csr,
                    'unspliced': unspliced_csr
                }
            )
            adata.uns['probe_set'] = probe_set.probe_set_meta
            return adata

    def _parse_read(self, read: pysam.libcalignedsegment.AlignedSegment) \
            -> tuple:
        """
        Parse a single read and obtain all necessary info.

        Parameters
        ----------
        read : pysam.libcalignedsegment.AlignedSegment
            The read to parse.

        Returns
        -------
        tuple
            A tuple containing the gene name, cell barcode, and the probe ID.
        """
        if not read.has_tag("CB") or \
                not read.has_tag("pr") or \
                not read.has_tag("UB"):
            return None, None, None, None
        cell_barcode = read.get_tag("CB")
        probe_id = read.get_tag('pr')
        umi = read.get_tag('UB')
        probes = probe_id.split(';')
        if len(probes) > 1:
            return None, None, None, None
        else:
            probe_id = probes[0]
            gene_id = probe_id.split('|')[0]
        return cell_barcode, gene_id, probe_id, umi

    def _process_chunk(self, args):
        chunk_range, probe_set = args
        start, end = chunk_range
        spliced_counts = set()
        unspliced_counts = set()
        with pysam.AlignmentFile(self.file, "rb") as bam:
            for i, read in enumerate(bam.fetch(until_eof=True)):
                if i < start:
                    continue
                if i >= end:
                    break
                if read.is_unmapped or \
                        read.is_secondary or \
                        read.is_supplementary:
                    continue
                try:
                    cell_barcode, gene_id, probe_id, umi = \
                        self._parse_read(read)
                    if cell_barcode is None or \
                            gene_id is None or \
                            probe_id is None or \
                            umi is None:
                        continue

                    cell_idx = self.cell_map.loc[cell_barcode, 'index']
                    gene_idx = probe_set.gene_map.loc[gene_id, 'index']
                    is_spliced = probe_set.is_spliced(probe_id)
                except KeyError:
                    print(
                        "Failed identifying the cell: ",
                        f"{cell_barcode}, " +
                        f"or gene: {gene_id}, probe: {probe_id}. " +
                        "Skipping this read."
                    )
                    continue

                key = (cell_idx, gene_idx, umi)
                if is_spliced:
                    spliced_counts.add(key)
                else:
                    unspliced_counts.add(key)
        return spliced_counts, unspliced_counts

    # def _process_region(self, args):
    #     """
    #     Process a region of the BAM file in parallel.

    #     Parameters
    #     ----------
    #     args : tuple
    #         A tuple containing the bam filename, region to process, probe_set
    #         object.

    #     Returns
    #     -------
    #     tuple
    #         A tuple containing the COO representation of the spliced and
    #         unspliced counts for the region.
    #     """
    #     file, region, probe_set = args
    #     spliced_counts = set()
    #     unspliced_counts = set()

    #     with pysam.AlignmentFile(file, "rb") as bam:
    #         for read in bam.fetch(region=region):
    #             if read.is_unmapped or \
    #                     read.is_secondary or \
    #                     read.is_supplementary:
    #                 continue
    #             try:
    #                 cell_barcode, gene_id, probe_id, umi = \
    #                     self._parse_read(read)
    #                 if cell_barcode is None or \
    #                         gene_id is None or \
    #                         probe_id is None or \
    #                         umi is None:
    #                     continue

    #                 cell_idx = self.cell_map.loc[cell_barcode, 'index']
    #                 gene_idx = probe_set.gene_map.loc[gene_id, 'index']
    #                 is_spliced = probe_set.is_spliced(probe_id)
    #             except KeyError:
    #                 print(
    #                     "Failed identifying the cell: ",
    #                     f"{cell_barcode}, " +
    #                     f"or gene: {gene_id}, probe: {probe_id}. " +
    #                     "Skipping this read."
    #                 )
    #                 continue

    #             key = (cell_idx, gene_idx, umi)
    #             if is_spliced:
    #                 spliced_counts.add(key)
    #             else:
    #                 unspliced_counts.add(key)

    #     return spliced_counts, unspliced_counts

    def _make_csr(self, struct, probe_set: probe_set):
        """
        Convert a set of counts to a CSR sparse matrix.
        Parameters
        ----------
        struct : set
            A set of tuples where each tuple is (cell_index, gene_index, umi).
        probe_set : probe_set
            The probe set object containing gene mapping information.
        Returns
        -------
        tuple
            A tuple containing the CSR sparse matrix, list of cell barcodes,
            and list of gene names.
        """
        data = []
        i = []
        j = []
        counting = {}
        for triplet in struct:
            cell_idx, gene_idx, _ = triplet
            # Use umi as a unique identifier for the read
            key = (cell_idx, gene_idx)
            if key not in counting:
                counting[key] = 0
            counting[key] += 1

        for key, count in counting.items():
            data.append(count)
            i.append(key[0])
            j.append(key[1])
        coo = coo_matrix(
            (data, (i, j)),
            shape=(self.cell_map.shape[0], probe_set.gene_map.shape[0])
        )
        csr = coo.tocsr()
        cells = self.cell_map.index.tolist()
        genes = probe_set.gene_map.index.tolist()
        return csr, cells, genes
