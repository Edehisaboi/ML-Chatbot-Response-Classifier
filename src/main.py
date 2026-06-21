from japeto_classifier.cli import build_parser


if __name__ == "__main__":
    # Use bootstrap for the first time startup
    # Use serve for reuse
    args = build_parser().parse_args(["serve"])
    args.handler(args)
