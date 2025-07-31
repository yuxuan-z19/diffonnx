import argparse

import onnx

from .diff import MainDiff


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Diff Tool")
    parser.add_argument("ref_model", type=str, help="Path to the reference ONNX model")
    parser.add_argument("usr_model", type=str, help="Path to the user ONNX model")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--profile-dir", type=str, default=None, help="Directory for profiling results"
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        help="List of providers to use for runtime diff",
    )

    args = parser.parse_args()

    ref_model = onnx.load(args.ref_model)
    usr_model = onnx.load(args.usr_model)

    diff = MainDiff(
        ref_model,
        usr_model,
        profile_dir=args.profile_dir,
        providers=args.providers,
        verbose=args.verbose,
    )
    diff.summary(output=True)
