# ProbeVelo Verification Module

## Overview

The `verify` module validates probevelo output by comparing it against 10X Genomics (spaceranger/cellranger) raw counts. It ensures that `spliced + unspliced` counts from probevelo match the total raw counts.

## Key Features

✅ **Efficient comparison** using sparse matrices
✅ **Handles millions of barcodes** without loading everything into memory
✅ **Detailed statistics** per barcode and per gene
✅ **Mismatch reporting** for debugging
✅ **CLI and Python API** for easy integration

## Quick Start

### Python API

```python
from probevelo import verify, verify_and_save

# Basic verification
result = verify(
    'probevelo_output.h5ad',
    'raw_feature_bc_matrix.h5'
)

print(result)  # Shows summary statistics

# Save detailed results to CSV
result = verify_and_save(
    'probevelo_output.h5ad',
    'raw_feature_bc_matrix.h5',
    output_prefix='verification_results'
)
```

### Command Line

```bash
# Run verification
python3 -m probevelo.verify_cli \\
    probevelo_output.h5ad \\
    raw_feature_bc_matrix.h5 \\
    --output verification_results

# With tolerance (allow ±1 count)
python3 -m probevelo.verify_cli \\
    probevelo_output.h5ad \\
    raw_feature_bc_matrix.h5 \\
    --output results \\
    --tolerance 1
```

### Using the Example Script

```bash
cd /nfs/turbo/umms-welchjd-code/code/wayichen/paper_probe_velo

# Edit paths in scripts/verify_probevelo_counts.py
# Then run:
python3 scripts/verify_probevelo_counts.py
```

## Output Files

When using `verify_and_save()`, the following files are created:

### 1. `*_summary.txt`
Overall statistics and top mismatches:
```
ProbeVelo Verification Results
==================================================

Barcode Statistics:
  ProbeVelo barcodes:      6,951,758
  10X barcodes:           11,222,500
  Overlapping:             6,951,758
  Overlap rate:              100.00%

Gene Statistics:
  ProbeVelo genes:            19,405
  10X genes:                  19,405
  Common genes:               19,405

Count Statistics:
  Total ProbeVelo counts:  1,234,567,890
  Total 10X counts:        1,234,567,890
  Difference:                          0
  Match:                           True

✓ All counts match perfectly!
```

### 2. `*_per_barcode.csv`
Per-barcode comparison:
```csv
barcode,probevelo_counts,tenx_counts,difference,match
s_002um_00700_02130-1,1234,1234,0,True
s_002um_02043_00141-1,5678,5678,0,True
...
```

### 3. `*_per_gene.csv`
Per-gene comparison:
```csv
gene,probevelo_counts,tenx_counts,difference,match
ENSMUSG00000000001,123456,123456,0,True
ENSMUSG00000000003,789012,789012,0,True
...
```

### 4. `*_mismatches.csv` (if any found)
Detailed mismatch information:
```csv
barcode,gene,probevelo_count,tenx_count,diff,abs_diff
s_002um_01234_05678-1,ENSMUSG00000012345,100,102,-2,2
...
```

## API Reference

### `verify(probevelo_h5ad, tenx_h5, tolerance=0, max_mismatches_to_report=1000)`

Run verification and return results object.

**Parameters:**
- `probevelo_h5ad`: Path to probevelo h5ad file or AnnData object
- `tenx_h5`: Path to 10X `raw_feature_bc_matrix.h5` (spaceranger or cellranger output)
- `tolerance`: Allowable difference per entry (default: 0)
- `max_mismatches_to_report`: Max mismatches to store (default: 1000)

**Returns:** `VerificationResult` object

### `verify_and_save(..., output_prefix)`

Run verification and save results to CSV files.

**Additional parameter:**
- `output_prefix`: Prefix for output files (e.g., `"results/verification"`)

**Returns:** `VerificationResult` object

### `VerificationResult` Attributes

- `n_probevelo_barcodes`: Number of barcodes in probevelo
- `n_tenx_barcodes`: Number in 10X output
- `n_overlapping`: Overlapping barcodes
- `n_probevelo_genes`: Genes in probevelo
- `n_tenx_genes`: Genes in 10X output
- `n_common_genes`: Common genes
- `total_counts_match`: Whether totals match
- `per_barcode_match`: DataFrame with per-barcode stats
- `per_gene_match`: DataFrame with per-gene stats
- `mismatches`: DataFrame with mismatch details

## How It Works

### Efficient Sparse Matrix Comparison

1. **Load only overlapping regions**
   ```python
   # Find overlapping barcodes and genes
   overlapping_bcs = set(probevelo_bcs) & set(spaceranger_bcs)
   common_genes = set(probevelo_genes) & set(spaceranger_genes)

   # Extract only the overlapping submatrices
   pv_sub = probevelo_matrix[overlapping_bc_idx][:, common_gene_idx]
   sr_sub = spaceranger_matrix[overlapping_bc_idx][:, common_gene_idx]
   ```

2. **Compare in chunks** (if needed)
   - For 6.95M barcodes × 19.4K genes, the overlapping region is manageable
   - Only converts sparse to dense for the overlapping region
   - Memory: ~2-4 GB for comparison

3. **Report efficiently**
   - Aggregates per-barcode and per-gene statistics
   - Stores only top mismatches (by absolute difference)
   - Avoids storing every comparison

### 10X Genomics H5 Format

The module reads the HDF5 file structure:
```
raw_feature_bc_matrix.h5
├── matrix/
│   ├── barcodes          # Cell barcodes
│   ├── data              # Non-zero values
│   ├── indices           # Row indices
│   ├── indptr            # Column pointers (CSC format)
│   ├── shape             # Matrix dimensions
│   └── features/
│       ├── id            # Gene IDs (ENSEMBL)
│       └── name          # Gene names (symbols)
```

Note: 10X stores as **CSC (genes × cells)**, so we transpose to **CSR (cells × genes)** to match probevelo format.

## Typical Results

### Expected: Perfect Match
For properly functioning probevelo, you should see:
```
✓ All counts match perfectly!
Total ProbeVelo counts:  1,234,567,890
Total 10X counts:        1,234,567,890
Difference:                          0
```

### Possible Mismatches

If mismatches occur, possible causes:

1. **Different filtering**
   - ProbeVelo may filter certain reads (xf tag != 25)
   - Check 10X filtering settings

2. **Multi-mapping reads**
   - Difference in handling multi-mapped reads
   - Check MAPQ thresholds

3. **Probe set mismatch**
   - Ensure using same probe set as 10X pipeline

4. **Gene ID mapping**
   - Verify gene IDs match between outputs

## Example Use Case

```python
# After running probevelo
import scanpy as sc
from probevelo import verify_and_save

# Verify 2µm resolution
result = verify_and_save(
    'results/velocity_2um.h5ad',
    'datasets/.../square_002um/raw_feature_bc_matrix.h5',
    'results/verification_2um'
)

if len(result.mismatches) == 0:
    print("✓ ProbeVelo counting verified!")
    print("  Safe to proceed with velocity analysis")
else:
    print("⚠ Issues detected, investigating...")
    # Check mismatches
    print(result.mismatches.head())
```

## Performance

### Memory Usage
- **Barcode collection**: ~100 MB
- **Sparse matrices**: ~2-4 GB (for overlapping region)
- **Comparison**: ~4-8 GB peak
- **Total**: ~8-16 GB for 7M cells

### Runtime
- Load probevelo: ~30 seconds
- Load 10X output: ~1 minute
- Find overlaps: ~10 seconds
- Comparison: ~2-5 minutes
- **Total: ~5-10 minutes**

Much faster than probevelo counting itself!

## Troubleshooting

### "No overlapping barcodes found"
- Check barcode formats match (e.g., `-1` suffix)
- Verify you're using matching 10X output

### "No common genes found"
- Ensure gene IDs are in same format (ENSEMBL IDs)
- Check probe set matches 10X reference
# (not yet implemented, but possible extension)
```

## Integration with Workflow

```bash
# 1. Run 10X pipeline (spaceranger/cellranger)
sbatch scripts/mouse_intestine_visium_hd_alignment.sh

# 2. Run probevelo
sbatch scripts/mouse_intestine_visium_hd_count_spliced.sh

# 3. Verify results
python3 scripts/verify_probevelo_counts.py

# 4. If verification passes, proceed with analysis
python3 scripts/mouse_intestine_visium_hd_analysis.py
```

## Future Enhancements

Possible additions:
- [ ] Batch processing for extremely large datasets
- [ ] Visualization of mismatch patterns
- [ ] Gene-level filtering statistics
- [ ] Read-level debugging (trace specific mismatches back to BAM)
- [ ] Multi-resolution verification (2µm vs 8µm vs 16µm)
