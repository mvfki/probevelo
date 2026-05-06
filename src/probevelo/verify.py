"""
Verification module for probevelo output.

Validates that probevelo spliced + unspliced counts match the
raw counts from 10X Genomics (spaceranger/cellranger) output.
"""

import pandas as pd
import numpy as np
import h5py
import anndata
from scipy.sparse import csr_matrix
from .logger import logger


class VerificationResult:
    """
    Container for verification results.

    Attributes
    ----------
    n_probevelo_barcodes : int
        Number of barcodes in probevelo output
    n_tenx_barcodes : int
        Number of barcodes in 10X output
    n_overlapping : int
        Number of overlapping barcodes
    n_probevelo_genes : int
        Number of genes in probevelo output
    n_tenx_genes : int
        Number of genes in 10X output
    n_common_genes : int
        Number of genes present in both outputs
    total_counts_match : bool
        Whether total counts match across all overlapping barcodes
    per_barcode_match : pd.DataFrame
        Per-barcode statistics (barcode, probevelo_counts, tenx_counts, match)
    per_gene_match : pd.DataFrame
        Per-gene statistics (gene, probevelo_counts, tenx_counts, match)
    mismatches : pd.DataFrame
        Details of mismatches (barcode, gene, probevelo_count, tenx_count, diff)
    """

    def __init__(self):
        self.n_probevelo_barcodes = 0
        self.n_tenx_barcodes = 0
        self.n_overlapping = 0
        self.n_probevelo_genes = 0
        self.n_tenx_genes = 0
        self.n_common_genes = 0
        self.total_counts_match = False
        self.per_barcode_match = None
        self.per_gene_match = None
        self.mismatches = None

    def __repr__(self):
        s = "ProbeVelo Verification Results\n"
        s += "=" * 50 + "\n\n"
        s += "Barcode Statistics:\n"
        s += f"  ProbeVelo barcodes:    {self.n_probevelo_barcodes:>10,}\n"
        s += f"  10X barcodes:          {self.n_tenx_barcodes:>10,}\n"
        s += f"  Overlapping:           {self.n_overlapping:>10,}\n"
        if self.n_probevelo_barcodes > 0:
            s += f"  Overlap rate:          {100*self.n_overlapping/self.n_probevelo_barcodes:>10.2f}%\n"
        s += "\n"

        s += "Gene Statistics:\n"
        s += f"  ProbeVelo genes:       {self.n_probevelo_genes:>10,}\n"
        s += f"  10X genes:             {self.n_tenx_genes:>10,}\n"
        s += f"  Common genes:          {self.n_common_genes:>10,}\n\n"

        s += "Verification Result:\n"
        if self.total_counts_match:
            s += "  ✓ PASSED: Matrices are IDENTICAL\n"
            s += "    - data, indices, indptr all match\n"
        else:
            s += "  ✗ FAILED: Matrices differ\n"

        return s


def load_tenx_h5(h5_path):
    """
    Load raw_feature_bc_matrix.h5 from 10X Genomics (spaceranger/cellranger) output.

    Parameters
    ----------
    h5_path : str
        Path to raw_feature_bc_matrix.h5 file

    Returns
    -------
    csr_matrix
        Sparse matrix of counts (cells x genes)
    list
        List of barcode strings
    pd.DataFrame
        DataFrame with gene information (id, name)
    """
    logger.info(f"Loading 10X output: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Read barcodes
        barcodes = [bc.decode() if isinstance(bc, bytes) else bc
                   for bc in f['matrix']['barcodes'][:]]

        # Read features (genes)
        gene_ids = [gid.decode() if isinstance(gid, bytes) else gid
                   for gid in f['matrix']['features']['id'][:]]
        gene_names = [gn.decode() if isinstance(gn, bytes) else gn
                     for gn in f['matrix']['features']['name'][:]]

        # Read sparse matrix (CSC format in h5, convert to CSR)
        data = f['matrix']['data'][:]
        indices = f['matrix']['indices'][:]
        indptr = f['matrix']['indptr'][:]
        shape = f['matrix']['shape'][:]

        # 10X stores as CSC (genes x cells), we want CSR (cells x genes)
        mat_csr = csr_matrix((data, indices, indptr), shape=[shape[1], shape[0]])

    genes_df = pd.DataFrame({
        'gene_id': gene_ids,
        'gene_name': gene_names
    })

    logger.info(f"  Loaded {len(barcodes):,} barcodes x {len(gene_ids):,} genes")
    logger.info(f"  Total counts: {mat_csr.sum():,.0f}")

    return mat_csr, barcodes, genes_df


def verify(
    probevelo_h5ad,
    tenx_h5
):
    """
    Verify probevelo output against 10X Genomics (spaceranger/cellranger) raw counts.

    Checks that spliced + unspliced counts from probevelo equal the
    raw feature counts from 10X output for overlapping barcodes.

    Parameters
    ----------
    probevelo_h5ad : str or AnnData
        Path to probevelo output h5ad file or AnnData object
    tenx_h5 : str
        Path to 10X raw_feature_bc_matrix.h5 file (spaceranger or cellranger output)

    Returns
    -------
    VerificationResult
        Object containing verification statistics and results
    """
    result = VerificationResult()

    # Load probevelo output
    logger.info("Loading probevelo output...")
    if isinstance(probevelo_h5ad, str):
        adata = anndata.read_h5ad(probevelo_h5ad)
    else:
        adata = probevelo_h5ad

    logger.info(f"  ProbeVelo: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Calculate total counts (spliced + unspliced)
    probevelo_total = adata.layers['spliced'] + adata.layers['unspliced']
    probevelo_barcodes = adata.obs.index.tolist()
    probevelo_genes = adata.var.index.tolist()

    result.n_probevelo_barcodes = len(probevelo_barcodes)
    result.n_probevelo_genes = len(probevelo_genes)

    # Load 10X output
    sr_matrix, sr_barcodes, sr_genes_df = load_tenx_h5(tenx_h5)

    result.n_tenx_barcodes = len(sr_barcodes)
    result.n_tenx_genes = len(sr_genes_df)

    # Find overlapping barcodes
    logger.info("Finding overlapping barcodes...")
    probevelo_bc_set = set(probevelo_barcodes)
    sr_bc_set = set(sr_barcodes)
    overlapping_bcs = probevelo_bc_set & sr_bc_set

    result.n_overlapping = len(overlapping_bcs)

    logger.info(f"  Overlapping barcodes: {result.n_overlapping:,}")
    logger.info(f"  ProbeVelo only: {len(probevelo_bc_set - sr_bc_set):,}")
    logger.info(f"  10X only: {len(sr_bc_set - probevelo_bc_set):,}")

    if result.n_overlapping == 0:
        logger.warning("No overlapping barcodes found!")
        return result

    # Find common genes (by gene ID)
    logger.info("Finding common genes...")
    sr_gene_ids = sr_genes_df['gene_id'].tolist()
    common_genes = list(set(probevelo_genes) & set(sr_gene_ids))
    result.n_common_genes = len(common_genes)

    logger.info(f"  Common genes: {result.n_common_genes:,}")

    if result.n_common_genes == 0:
        logger.warning("No common genes found!")
        return result

    # Create index mappings for efficient lookup
    logger.info("Creating index mappings...")
    pv_bc_to_idx = {bc: i for i, bc in enumerate(probevelo_barcodes)}
    sr_bc_to_idx = {bc: i for i, bc in enumerate(sr_barcodes)}

    pv_gene_to_idx = {gene: i for i, gene in enumerate(probevelo_genes)}
    sr_gene_to_idx = {gene: i for i, gene in enumerate(sr_gene_ids)}

    # Get indices for overlapping barcodes and common genes
    overlapping_bc_list = sorted(overlapping_bcs)
    pv_bc_indices = np.array([pv_bc_to_idx[bc] for bc in overlapping_bc_list])
    sr_bc_indices = np.array([sr_bc_to_idx[bc] for bc in overlapping_bc_list])

    common_gene_list = sorted(common_genes)
    pv_gene_indices = np.array([pv_gene_to_idx[gene] for gene in common_gene_list])
    sr_gene_indices = np.array([sr_gene_to_idx[gene] for gene in common_gene_list])

    # Extract submatrices for overlapping regions
    logger.info("Extracting overlapping regions...")
    pv_sub = probevelo_total[np.ix_(pv_bc_indices, pv_gene_indices)]
    sr_sub = sr_matrix[np.ix_(sr_bc_indices, sr_gene_indices)]

    # Ensure both are CSR format for efficient comparison
    if not isinstance(pv_sub, csr_matrix):
        pv_sub = pv_sub.tocsr()
    if not isinstance(sr_sub, csr_matrix):
        sr_sub = sr_sub.tocsr()

    logger.info(f"  Submatrix shape: {pv_sub.shape}")
    logger.info(f"  ProbeVelo nnz: {pv_sub.nnz:,}")
    logger.info(f"  10X nnz: {sr_sub.nnz:,}")

    # Compare sparse matrices directly without converting to dense
    logger.info("Comparing sparse matrices...")

    # Check if barcodes and genes are in the same order (they should be after sorting)
    logger.info("  Barcodes and genes are in matching order after sorting")

    # Compare sparse matrix structure and values
    matrices_identical = False
    if pv_sub.shape == sr_sub.shape and pv_sub.nnz == sr_sub.nnz:
        # Check if sparse components are identical
        data_match = np.array_equal(pv_sub.data, sr_sub.data)
        indices_match = np.array_equal(pv_sub.indices, sr_sub.indices)
        indptr_match = np.array_equal(pv_sub.indptr, sr_sub.indptr)

        matrices_identical = data_match and indices_match and indptr_match

        logger.info(f"  Shapes match: {pv_sub.shape == sr_sub.shape}")
        logger.info(f"  NNZ match: {pv_sub.nnz == sr_sub.nnz}")
        logger.info(f"  Data match: {data_match}")
        logger.info(f"  Indices match: {indices_match}")
        logger.info(f"  Indptr match: {indptr_match}")
    else:
        logger.info(f"  Shapes or nnz differ - matrices are not identical")
        logger.info(f"    ProbeVelo shape: {pv_sub.shape}, nnz: {pv_sub.nnz:,}")
        logger.info(f"    10X shape: {sr_sub.shape}, nnz: {sr_sub.nnz:,}")

    # Overall statistics
    total_pv = pv_sub.sum()
    total_sr = sr_sub.sum()
    result.total_counts_match = matrices_identical

    logger.info(f"  Total ProbeVelo counts: {total_pv:,.0f}")
    logger.info(f"  Total 10X counts: {total_sr:,.0f}")

    if matrices_identical:
        logger.info("  ✓ VERIFICATION PASSED: Sparse matrices are IDENTICAL")
    else:
        logger.warning("  ✗ VERIFICATION FAILED: Matrices differ")

    # Store simple statistics without computing differences
    result.per_barcode_match = None
    result.per_gene_match = None
    result.mismatches = None

    return result
