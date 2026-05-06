# Memory-Efficient ProbeVelo Refactoring

## Problem Analysis

### Current Implementation
- **Data size**: 6,951,758 barcodes × 19,405 genes = 134 billion potential entries
- **Sparse occupancy**: ~0.1-1% → 135M - 1.35B actual entries
- **Memory bottleneck**: Using Python dict with tuple keys before sparse conversion

### Memory Estimation (Current)
```python
# Dict entry structure:
# - Tuple (cell_idx, gene_idx): ~56 bytes
# - Count value (int): ~28 bytes
# - Dict overhead: ~100 bytes
# Total per entry: ~184 bytes

# For 135M entries (0.1% occupancy):
135,000,000 entries × 184 bytes = 24.8 GB memory

# For 1.35B entries (1% occupancy):
1,350,000,000 entries × 184 bytes = 248 GB memory (!)
```

## Refactoring Strategy

### Key Optimizations

1. **Chunk-based Processing**
   - Process regions, accumulate in arrays
   - When chunk_size reached → convert to sparse → write to disk
   - Default: 10M entries per chunk (~1.6GB memory)

2. **Direct COO Construction**
   - Use numpy arrays for (i, j, data)
   - Avoid dict overhead
   - ~16 bytes per entry vs ~184 bytes

3. **Memory Monitoring**
   - Track RSS memory usage with psutil
   - Flush to disk when `max_mem_gb` exceeded
   - Default: 8 GB limit

4. **Streaming Merge**
   - Load temp files one at a time
   - Sum sparse matrices incrementally
   - Clean up temp files as we go

### Memory Comparison

| Method | Memory per Entry | 135M Entries | 1.35B Entries |
|--------|-----------------|--------------|---------------|
| Dict (current) | 184 bytes | 24.8 GB | 248 GB |
| COO arrays | 16 bytes | 2.2 GB | 22 GB |
| **Chunked (8GB)** | **streaming** | **8 GB** | **8 GB** |

## Usage

### Basic Usage (Drop-in Replacement)

```python
from probevelo import probe_set
from probevelo.bam_parser_memeff import bam_parser_memeff

# Load probe set
probe_set_obj = probe_set('probe_set.csv')

# Initialize memory-efficient parser
bam = bam_parser_memeff(
    'possorted_genome_bam.bam',
    n_thread=12,
    max_mem_gb=8.0,          # Memory limit
    chunk_size=10_000_000     # Entries per chunk
)

# Count spliced/unspliced reads
adata = bam.count_splice(probe_set_obj, adata=True)

# Save results
adata.write('velocity_counts.h5ad')
```

### Advanced Usage

```python
# For very large datasets (10M+ barcodes):
bam = bam_parser_memeff(
    'possorted_genome_bam.bam',
    n_thread=24,
    max_mem_gb=16.0,          # More memory if available
    chunk_size=20_000_000,    # Larger chunks = fewer disk writes
    temp_dir='/scratch/temp'  # Fast scratch disk
)

# Get raw matrices instead of AnnData
spliced_csr, unspliced_csr, cells, genes = bam.count_splice(
    probe_set_obj,
    adata=False
)
```

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=probevelo_memeff
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB              # 16GB for 8GB limit + overhead
#SBATCH --time=12:00:00
#SBATCH --output=probevelo-%j.out

module load python/3.10

python << 'EOF'
from probevelo import probe_set
from probevelo.bam_parser_memeff import bam_parser_memeff

probe_set_obj = probe_set(
    'outs/probe_set.csv'
)

bam = bam_parser_memeff(
    'outs/possorted_genome_bam.bam',
    n_thread=12,
    max_mem_gb=8.0,
    chunk_size=10_000_000,
    temp_dir='/tmp'
)

adata = bam.count_splice(probe_set_obj, adata=True)
adata.write('velocity_2um.h5ad')
print(f"Saved: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"Spliced: {adata.layers['spliced'].nnz:,} entries")
print(f"Unspliced: {adata.layers['unspliced'].nnz:,} entries")
EOF
```

## Performance Benchmarks

### Expected Performance (6.95M cells × 19.4K genes)

| Method | Peak Memory | Runtime | Disk Usage |
|--------|-------------|---------|------------|
| Original dict | 25-250 GB | 3-6 hours | 0 GB temp |
| **Memory-efficient** | **8-16 GB** | **4-8 hours** | **5-10 GB temp** |

*Runtime increased 25-50% due to disk I/O, but memory controlled*

### Scaling

| Dataset Size | Recommended Settings | Memory | Runtime |
|--------------|---------------------|--------|---------|
| 1-2M cells | `max_mem_gb=4, chunk_size=5M` | 6 GB | 1-2 h |
| 2-7M cells | `max_mem_gb=8, chunk_size=10M` | 12 GB | 3-6 h |
| 7-15M cells | `max_mem_gb=16, chunk_size=20M` | 20 GB | 6-12 h |

## Implementation Details

### Flushing Logic

```python
# Flush triggered by EITHER condition:
if total_entries >= chunk_size or mem_used >= max_mem_gb:
    flush_to_disk()

# This ensures:
# 1. Memory never exceeds max_mem_gb
# 2. Chunks are reasonable size for I/O efficiency
```

### Temporary File Format

```
/tmp/probevelo_spliced_0.npz    # scipy CSR sparse matrix
/tmp/probevelo_spliced_1.npz
/tmp/probevelo_unspliced_0.npz
...
```

Files are automatically cleaned up after merging.

### Memory Overhead

```
Total memory ≈ max_mem_gb + overhead

Overhead includes:
- Cell/gene mappings: ~500 MB for 7M cells
- BAM file handles: ~100 MB
- Python process: ~500 MB
- scipy operations: ~1-2 GB

Recommended: SLURM mem = 2 × max_mem_gb
```

## Migration Path

### Option 1: Import New Class

```python
# Add to probevelo/__init__.py
from .bam_parser_memeff import bam_parser_memeff

# Users can choose:
from probevelo import bam_parser          # Original
from probevelo import bam_parser_memeff   # Memory-efficient
```

### Option 2: Add Flag to Original Class

```python
# Add to bam_parser.__init__():
def __init__(self, file, n_thread=1, memory_efficient=False, max_mem_gb=8.0):
    if memory_efficient:
        # Use chunked processing
    else:
        # Use original dict method
```

### Option 3: Auto-detect

```python
# In bam_parser.count_splice():
n_potential_entries = n_cells * n_genes
if n_potential_entries > 100_000_000:  # 100M entries
    logger.warning("Large dataset detected, using memory-efficient mode")
    return self._count_splice_chunked(...)
else:
    return self._count_splice_dict(...)
```

## Testing

```python
# Compare results between methods
from probevelo import probe_set, bam_parser
from probevelo.bam_parser_memeff import bam_parser_memeff

ps = probe_set('probe_set.csv')

# Original
bam1 = bam_parser('test.bam')
adata1 = bam1.count_splice(ps, adata=True)

# Memory-efficient
bam2 = bam_parser_memeff('test.bam', max_mem_gb=2.0)
adata2 = bam2.count_splice(ps, adata=True)

# Verify identical results
assert (adata1.X != adata2.X).nnz == 0
assert (adata1.layers['spliced'] != adata2.layers['spliced']).nnz == 0
assert (adata1.layers['unspliced'] != adata2.layers['unspliced']).nnz == 0
print("✓ Results identical!")
```

## Future Optimizations

1. **HDF5 Backend**: Stream directly to HDF5 instead of NPZ
2. **GPU Acceleration**: Use cupy for sparse operations
3. **Distributed Processing**: Dask/Ray for multi-node
4. **Incremental Updates**: Resume from checkpoint if job fails

## Questions?

For issues or suggestions, please open a GitHub issue at:
https://github.com/mvfki/probevelo/issues
