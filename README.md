# probevelo

Work-around for RNA velocity inference from 10X Fixed RNA Profiling using probe sets

## Installation

Only available in experimental stage on GitHub for now:

```bash
pip install git+https://github.com/mvfki/probevelo.git
```

## Usage

Load the probe set information into a `probe_set` instance and create a `bam_parser` instance that will do the counting.
Then, simply use the `bam_parser.count_splice` method to count spliced and unspliced reads and return an AnnData object.

```python
import probevelo

probe_set = probevelo.probe_set('/path/to/10x/out/probe_set.csv')
bam_parser = probevelo.bam_parser('/path/to/10x/out/bam_file.bam')

adata = bam_parser.count_splice(probe_set, adata=True)
```

The returned `anndata.AnnData` object will contain:

- `adata.X`: the total raw counts, i.e. sum of spliced and unspliced counts
- `adata.layers['spliced']`: the spliced counts
- `adata.layers['unspliced']`: the unspliced counts
- `adata.obs`: Only the barcodes as the index
- `adata.var`: All gene IDs available in the probe set will be used as the index,
and the `gene_name` column will be mapped as an alternative column. Note that multiple
gene IDs can map to the same gene name, so we by default use the ID as the index.
