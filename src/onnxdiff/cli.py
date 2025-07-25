import argparse
import onnx

from .diff import OnnxDiff


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Diff Tool")
    parser.add_argument("ref_model", type=str, help="Path to the reference ONNX model")
    parser.add_argument("usr_model", type=str, help="Path to the user ONNX model")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    ref_model = onnx.load(args.ref_model)
    usr_model = onnx.load(args.usr_model)

    diff = OnnxDiff(ref_model, usr_model, verbose=args.verbose)
    results = diff.summary(output=True)

    if args.verbose:
        print(results)
