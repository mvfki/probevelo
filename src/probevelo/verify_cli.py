#!/usr/bin/env python3
"""
Verification CLI for probevelo.

Usage:
    probevelo-verify \\
        probevelo_output.h5ad \\
        raw_feature_bc_matrix.h5 \\
        --output verification_results

This will generate:
- verification_results_summary.txt
- verification_results_per_barcode.csv
- verification_results_per_gene.csv
- verification_results_mismatches.csv (if any found)
"""

import argparse
import sys
from .verify import verify_and_save
from .logger import logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify probevelo output against 10X Genomics (spaceranger/cellranger) raw counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  probevelo-verify output.h5ad raw_feature_bc_matrix.h5 -o results

  # With tolerance (allow ±1 count difference)
  probevelo-verify output.h5ad raw_feature_bc_matrix.h5 -o results -t 1

  # Report more mismatches
  probevelo-verify output.h5ad raw_feature_bc_matrix.h5 -o results -m 10000
"""
    )

    parser.add_argument(
        'probevelo_h5ad',
        help="Path to probevelo output h5ad file"
    )

    parser.add_argument(
        'tenx_h5',
        help="Path to 10X raw_feature_bc_matrix.h5 file (spaceranger or cellranger output)"
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help="Output prefix for result files"
    )

    parser.add_argument(
        '-t', '--tolerance',
        type=int,
        default=0,
        help="Allowable count difference per entry (default: 0, exact match)"
    )

    parser.add_argument(
        '-m', '--max-mismatches',
        type=int,
        default=1000,
        help="Maximum mismatches to report in detail (default: 1000)"
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Suppress info messages"
    )

    args = parser.parse_args()

    if args.quiet:
        import logging
        logger.setLevel(logging.WARNING)

    # Run verification
    try:
        result = verify_and_save(
            args.probevelo_h5ad,
            args.tenx_h5,
            args.output,
            tolerance=args.tolerance,
            max_mismatches_to_report=args.max_mismatches
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


if __name__ == '__main__':
    sys.exit(main())
