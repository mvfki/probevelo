import pandas as pd
from .logger import logger


class probe_set:
    """
    Class to contain the probe set information and allow fast access to
    necessary information.

    Attributes
    ----------
    probes : pd.DataFrame
        DataFrame containing the probe set information.
    probe_set_meta : dict
        Dictionary containing metadata about the probe set, such as the
        version, species, and genome build.
    gene_map : dict
        Map unique gene names to their corresponding row indices in the final
        count matrix.

    Methods
    -------
    __init__(file: str)
        Initialize the probe_set object by loading the probe set from a file.
    __repr__()
        Return a string representation of the probe set metadata and table.
    _load_probe_set(file: str)
        Load the probe set from a file and parse the metadata and probe
        information into a DataFrame.
    row_for(gene: str) -> int
        Get the row index for a given gene in the final count matrix.
    gene_for(probe_id: str) -> str
        Get the gene name for a given probe ID.
    is_spliced(probe_id: str) -> bool
        Check if a given probe ID is for a spliced region.
    """

    def __init__(self, file: str, quiet: bool = False):
        """
        Initialize the probe_set object by loading the probe set from a file.

        Parameters
        ----------
        file : str
            Path to the file containing the probe set. This is usually found in
            the 10X cellranger output folder named after `probe_set.csv`. You
            might also find it in the cellranger installation location, e.g.
            `/path/to/cellranger/7.1.0/probe_sets/Chromium_Mouse_Transcriptome_Probe_Set_v1.0.1_mm10-2020-A.csv`
        quiet : bool, optional
            If True, suppress logging output. Default is False.
        """
        self._load_probe_set(file)
        if not quiet:
            logger.info(f"Found {self.gene_map.shape[0]} unique genes " +
                        "in the probe set.")

    def __repr__(self):
        repr_str = 'Probe set metadata:\n'
        for key, value in self.probe_set_meta.items():
            repr_str += f'{key}: {value}\n'
        repr_str += f'Probe set table:\n{self.probes.__repr__()}'
        return repr_str

    def _load_probe_set(self, file: str) -> pd.DataFrame:
        """
        Load the probe set from a file.

        Parameters
        ----------
        file : str
            Path to the file containing the probe set.
        """
        # Parse the commented headers of the CSV files into a dictionary
        self.probe_set_meta = {}
        header_n_row = 0
        with open(file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    key_value = line[1:].strip().split("=", 1)
                    self.probe_set_meta[key_value[0].strip()] = key_value[1]\
                        .strip()
                    header_n_row += 1
                else:
                    break

        self.probes = pd.read_csv(file, sep=",", skiprows=header_n_row)
        self.probes.drop(columns=['probe_seq'], inplace=True)
        # Fetch gene name from probe_id, which is formed as
        # "gene_id|gene_symbol|probe_id"
        self.probes['gene_name'] = \
            self.probes['probe_id'].str.split('|').str[1]
        gene_ids = self.probes['gene_id'].unique()
        self.probes.index = self.probes['probe_id']
        self.probes.drop(columns=['probe_id'], inplace=True)

        # Make row index for final count matrix based on unique genes
        self.gene_map = pd.DataFrame(
            index=gene_ids,
            data={
                'index': range(len(gene_ids)),
                'gene_name':
                self.probes.groupby('gene_id')['gene_name'].first()
            }
        )

    def is_spliced(self, probe_id: str) -> bool:
        """
        Check if a given probe ID is for a spliced region.

        Parameters
        ----------
        probe_id : str
            The probe ID to check.

        Returns
        -------
        bool
            True if the probe is for a spliced region, False otherwise.
        """
        return self.probes.loc[probe_id, 'region'] == 'spliced'
