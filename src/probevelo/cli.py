import argparse
from .bam_parser import bam_parser
from .probe_set import probe_set
import sys
from importlib.metadata import version
from .logger import logger


__version__ = version("probevelo")


def arg_parse():
    """
    Parse command line arguments for the probevelo CLI.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="ProbeVelo: Work around for counting spliced and unspliced reads from FRP BAM files."
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit"
    )

    parser.add_argument(
        "BAM",
        type=str,
        help="Path to the BAM file containing RNA sequencing data. Must be indexed."
    )

    parser.add_argument(
        "PROBE_SET",
        type=str,
        help="Path to the probe set CSV file."
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for counting (default: 1)."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.h5ad",
        help="Output file path for the results (default: output.h5ad)."
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress logging output (default: False)."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    return parser.parse_args()


def main():
    """
    Main function to run the probevelo CLI.
    Parses command line arguments and executes the counting process.
    """

    args = arg_parse()

    b = bam_parser(args.BAM, n_thread=args.threads, quiet=args.quiet)
    p = probe_set(args.PROBE_SET, quiet=args.quiet)
    adata = b.count_splice(probe_set=p, adata=True)

    # Save the results to an AnnData object
    adata.write_h5ad(args.output)
    # logger level is set when passing the quiet argument to bam_parser
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
