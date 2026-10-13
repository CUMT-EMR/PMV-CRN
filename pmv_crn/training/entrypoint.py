from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--encoder",
        choices=["inception", "sensor_grouped", "patch_transformer", "precursor"],
        default="inception",
    )
    encoder_args, remaining = parser.parse_known_args()
    os.environ["IMAGE_ENCODER_VARIANT"] = encoder_args.encoder

    from pmv_crn.training import balanced
    from pmv_crn.training import engine

    args = engine.build_argparser().parse_args(remaining)
    if args.auto_search:
        engine.auto_search(args)
    else:
        engine.train(args)


if __name__ == "__main__":
    main()
