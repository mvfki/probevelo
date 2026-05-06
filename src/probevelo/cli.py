import argparse
from .bam_parser import bam_parser
from .bam_parser_memeff import bam_parser_memeff
from .probe_set import probe_set
from .verify import verify
import sys
from importlib.metadata import version
from .logger import logger


__version__ = version("probevelo")


def count_command(args):
    """Execute the count subcommand."""
    # Choose parser based on --memeff flag
    if args.memeff:
        logger.info("Using memory-efficient counting mode")
        b = bam_parser_memeff(
            args.BAM,
            max_mem_gb=args.max_mem,
            chunk_size=args.chunk_size,
            temp_dir=args.temp_dir,
            n_thread=args.threads,
            quiet=args.quiet
        )
    else:
        b = bam_parser(args.BAM, n_thread=args.threads, quiet=args.quiet)

    p = probe_set(args.PROBE_SET, quiet=args.quiet)
    adata = b.count_splice(probe_set=p, adata=True)

    # Save the results to an AnnData object
    adata.write_h5ad(args.output)
    logger.info(f"Results saved to {args.output}")
    return 0


def verify_command(args):
    """Execute the verify subcommand."""
    if args.quiet:
        import logging
        logger.setLevel(logging.WARNING)

    try:
        result = verify(
            args.probevelo_h5ad,
            args.tenx_h5
        )

        # Print summary
        print("\n" + str(result))

        # Exit with error code if verification failed
        if not result.total_counts_match:
            print("\n⚠ Verification failed: matrices differ")
            return 1
        else:
            print("\n✓ Verification passed: matrices are identical!")
            return 0

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


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

    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )

    # ===== COUNT subcommand =====
    count_parser = subparsers.add_parser(
        'count',
        help='Count spliced and unspliced reads from BAM file',
        description='Count spliced and unspliced reads from probe-based FRP BAM files.'
    )

    count_parser.add_argument(
        "BAM",
        type=str,
        help="Path to the BAM file containing RNA sequencing data. Must be indexed."
    )

    count_parser.add_argument(
        "PROBE_SET",
        type=str,
        help="Path to the probe set CSV file."
    )

    count_parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of threads to use for counting (default: 1)."
    )

    count_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.h5ad",
        help="Output file path for the results (default: output.h5ad)."
    )

    count_parser.add_argument(
        "--memeff",
        action="store_true",
        help="Use memory-efficient counting mode for large datasets (6M+ barcodes)."
    )

    count_parser.add_argument(
        "--max-mem",
        type=float,
        default=8.0,
        help="Maximum memory to use in GB for memeff mode (default: 8.0)."
    )

    count_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Number of entries to accumulate before flushing to disk in memeff mode (default: 10,000,000)."
    )

    count_parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for temporary files in memeff mode (default: system temp)."
    )

    count_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress logging output (default: False)."
    )

    count_parser.set_defaults(func=count_command)

    # ===== VERIFY subcommand =====
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify probevelo output against 10X raw counts',
        description='Verify that probevelo spliced+unspliced counts match 10X Genomics (spaceranger/cellranger) raw counts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify output matches reference counts
  probevelo verify output.h5ad raw_feature_bc_matrix.h5
"""
    )

    verify_parser.add_argument(
        'probevelo_h5ad',
        help="Path to probevelo output h5ad file"
    )

    verify_parser.add_argument(
        'tenx_h5',
        help="Path to 10X raw_feature_bc_matrix.h5 file (spaceranger or cellranger output)"
    )

    verify_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Suppress info messages"
    )

    verify_parser.set_defaults(func=verify_command)

    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] in ['count', 'verify']:
        parser.parse_args([sys.argv[1], '-h'])
        sys.exit(0)

    return parser.parse_args()


def main():
    """
    Main function to run the probevelo CLI.
    Parses command line arguments and executes the appropriate subcommand.
    """
    args = arg_parse()

    # Execute the subcommand
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
