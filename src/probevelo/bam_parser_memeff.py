"""
Memory-efficient BAM parser for large-scale Visium HD data.

Key optimizations:
1. Chunk-based processing: Process barcodes in chunks to limit memory
2. Direct sparse construction: Build COO arrays without dict intermediate
3. Streaming writes: Write intermediate results to disk
4. Memory monitoring: Track and limit memory usage
"""
import pandas as pd
import pysam
from typing import Union
from .probe_set import probe_set
import multiprocessing as mp
import tqdm
from scipy.sparse import coo_matrix, csr_matrix, vstack
import anndata
import logging
from .logger import logger
import numpy as np
import tempfile
import os
import psutil


class bam_parser_memeff:
    """
    Memory-efficient version of bam_parser for large-scale data.

    Uses chunk-based processing and streaming writes to disk to keep
    memory usage bounded. Suitable for Visium HD 2µm data with millions
    of barcodes.

    Additional Parameters
    ---------------------
    max_mem_gb : float
        Maximum memory to use in GB. When exceeded, intermediate results
        are written to disk. Default: 8.0 GB
    chunk_size : int
        Number of COO entries to accumulate before converting to sparse.
        Default: 10_000_000 (10M entries ~1.6GB)
    temp_dir : str
        Directory for temporary files. Default: system temp
    """

    def __init__(
            self,
            file: str,
            n_thread: int = 1,
            quiet: bool = False,
            max_mem_gb: float = 8.0,
            chunk_size: int = 10_000_000,
            temp_dir: str = None
            ):
        """
        Initialize memory-efficient BAM parser.

        Parameters
        ----------
        file : str
            Path to BAM file
        n_thread : int
            Number of threads for parallel processing
        quiet : bool
            Suppress progress bars
        max_mem_gb : float
            Maximum memory usage in GB before flushing to disk
        chunk_size : int
            Number of entries to accumulate before creating sparse matrix
        temp_dir : str
            Directory for temporary files (default: system temp)
        """
        self.file = file
        self.quiet = quiet
        self.n_thread = n_thread
        self.max_mem_gb = max_mem_gb
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Initialize regions
        with pysam.AlignmentFile(file, "rb") as bam:
            regions = list(bam.references)
        regions.append('*')
        self.regions = regions

        # Collect cell barcodes
        with mp.Pool(self.n_thread) as pool:
            cb_per_region = list(tqdm.tqdm(
                pool.imap(self._collect_region_cb, self.regions),
                total=len(self.regions),
                desc="Collecting cell barcodes",
                unit="regions",
                disable=self.quiet
            ))

        cell_barcodes = set()
        for region in cb_per_region:
            cell_barcodes = cell_barcodes.union(region)

        self.cell_map = pd.DataFrame(
            index=list(cell_barcodes),
            data={"index": range(len(cell_barcodes))}
        )
        logger.info(
            f"Found {self.cell_map.shape[0]:,} unique cell barcodes"
        )
        logger.info(
            f"Memory limit: {self.max_mem_gb:.1f} GB, "
            f"Chunk size: {self.chunk_size:,} entries"
        )

    def _collect_region_cb(self, region):
        """Collect cell barcodes from a genomic region."""
        cell_barcodes = set()
        with pysam.AlignmentFile(self.file, "rb") as bam:
            for read in bam.fetch(region=region):
                if not read.has_tag('xf'):
                    continue
                if read.get_tag('xf') != 25:
                    continue
                cell_barcodes.add(read.get_tag("CB"))
        return cell_barcodes

    def _get_memory_usage_gb(self):
        """Get current process memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)

    def count_splice(
            self,
            probe_set: probe_set,
            adata: bool = False
            ) -> Union[tuple, anndata.AnnData]:
        """
        Count spliced/unspliced reads with memory-efficient streaming.

        Instead of accumulating all counts in memory, this method:
        1. Processes regions and accumulates counts in arrays
        2. When chunk_size is reached, converts to sparse and writes to temp file
        3. At the end, loads and merges all temp sparse matrices

        Parameters
        ----------
        probe_set : probe_set
            Probe set for mapping reads
        adata : bool
            Return AnnData object if True

        Returns
        -------
        Union[tuple, AnnData]
            Sparse matrices or AnnData object
        """
        n_genes = probe_set.gene_map.shape[0]
        n_cells = self.cell_map.shape[0]

        logger.info(
            f"Counting for {n_cells:,} cells × {n_genes:,} genes "
            f"= {n_cells * n_genes:,} potential entries"
        )

        # Accumulate COO arrays
        spliced_i, spliced_j, spliced_data = [], [], []
        unspliced_i, unspliced_j, unspliced_data = [], [], []

        # Temporary file tracking
        temp_files_spliced = []
        temp_files_unspliced = []

        def flush_to_disk():
            """Write current accumulated counts to temp files."""
            nonlocal spliced_i, spliced_j, spliced_data
            nonlocal unspliced_i, unspliced_j, unspliced_data

            if len(spliced_data) > 0:
                coo_s = coo_matrix(
                    (spliced_data, (spliced_i, spliced_j)),
                    shape=(n_cells, n_genes)
                )
                temp_file = os.path.join(
                    self.temp_dir,
                    f"probevelo_spliced_{len(temp_files_spliced)}.npz"
                )
                from scipy.sparse import save_npz
                save_npz(temp_file, coo_s.tocsr())
                temp_files_spliced.append(temp_file)
                logger.info(
                    f"Flushed {len(spliced_data):,} spliced entries to disk"
                )
                spliced_i, spliced_j, spliced_data = [], [], []

            if len(unspliced_data) > 0:
                coo_u = coo_matrix(
                    (unspliced_data, (unspliced_i, unspliced_j)),
                    shape=(n_cells, n_genes)
                )
                temp_file = os.path.join(
                    self.temp_dir,
                    f"probevelo_unspliced_{len(temp_files_unspliced)}.npz"
                )
                from scipy.sparse import save_npz
                save_npz(temp_file, coo_u.tocsr())
                temp_files_unspliced.append(temp_file)
                logger.info(
                    f"Flushed {len(unspliced_data):,} unspliced entries to disk"
                )
                unspliced_i, unspliced_j, unspliced_data = [], [], []

        # Process regions
        args = [(region, probe_set) for region in self.regions]

        for region_idx, (region, probe_set_arg) in enumerate(
            tqdm.tqdm(args, desc="Processing regions", disable=self.quiet)
        ):
            spliced_counts, unspliced_counts = self._process_region(
                (region, probe_set_arg)
            )

            # Add to accumulation arrays
            for (cell_idx, gene_idx), count in spliced_counts.items():
                spliced_i.append(cell_idx)
                spliced_j.append(gene_idx)
                spliced_data.append(count)

            for (cell_idx, gene_idx), count in unspliced_counts.items():
                unspliced_i.append(cell_idx)
                unspliced_j.append(gene_idx)
                unspliced_data.append(count)

            # Check if we need to flush
            total_entries = len(spliced_data) + len(unspliced_data)
            mem_used = self._get_memory_usage_gb()

            if total_entries >= self.chunk_size or mem_used >= self.max_mem_gb:
                logger.info(
                    f"Memory check: {mem_used:.2f} GB used, "
                    f"{total_entries:,} entries accumulated"
                )
                flush_to_disk()

        # Final flush
        if len(spliced_data) > 0 or len(unspliced_data) > 0:
            flush_to_disk()

        # Merge all temp files
        logger.info("Merging temporary sparse matrices...")

        def merge_temp_files(temp_files):
            """Load and sum all temporary sparse matrices."""
            if len(temp_files) == 0:
                return csr_matrix((n_cells, n_genes))

            from scipy.sparse import load_npz
            matrices = []
            for temp_file in tqdm.tqdm(
                temp_files,
                desc="Loading temp files",
                disable=self.quiet
            ):
                matrices.append(load_npz(temp_file))
                os.remove(temp_file)  # Clean up

            # Sum matrices efficiently
            result = matrices[0]
            for mat in matrices[1:]:
                result = result + mat
            return result

        spliced_csr = merge_temp_files(temp_files_spliced)
        unspliced_csr = merge_temp_files(temp_files_unspliced)

        logger.info(
            f"Final matrices: {spliced_csr.nnz:,} spliced, "
            f"{unspliced_csr.nnz:,} unspliced entries"
        )

        cell_list = self.cell_map.index.tolist()
        genes_list = probe_set.gene_map.index.tolist()

        if not adata:
            return spliced_csr, unspliced_csr, cell_list, genes_list
        else:
            logger.info("Creating AnnData object...")
            adata_obj = anndata.AnnData(
                X=spliced_csr + unspliced_csr,
                obs=pd.DataFrame(index=cell_list),
                var=pd.DataFrame(
                    index=genes_list,
                    data={
                        'gene_name': probe_set.gene_map.loc[
                            genes_list, 'gene_name'
                        ].astype('object')
                    }
                ),
                layers={
                    'spliced': spliced_csr,
                    'unspliced': unspliced_csr
                }
            )
            adata_obj.uns['probe_set'] = probe_set.probe_set_meta
            return adata_obj

    def _process_region(self, args):
        """Process a single genomic region (same as original)."""
        region, probe_set = args
        spliced_counts = {}
        unspliced_counts = {}

        with pysam.AlignmentFile(self.file, "rb") as bam:
            for read in bam.fetch(region=region):
                if not read.has_tag('xf'):
                    continue
                if read.get_tag('xf') != 25:
                    continue

                cell_barcode = read.get_tag("CB")
                probe_id = read.get_tag('pr')
                gene_id = probe_id.split('|')[0]
                cell_idx = self.cell_map.loc[cell_barcode, 'index']
                gene_idx = probe_set.gene_map.loc[gene_id, 'index']
                is_spliced = probe_set.is_spliced(probe_id)

                key = (cell_idx, gene_idx)
                if is_spliced:
                    spliced_counts[key] = spliced_counts.get(key, 0) + 1
                else:
                    unspliced_counts[key] = unspliced_counts.get(key, 0) + 1

        return spliced_counts, unspliced_counts
