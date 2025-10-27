"""
Fixed-depth (3-level) training wrapper for the adaptive U-Net pipeline.

This script reuses the main training entry point but forces ``depth_override=3`` so
we can sweep different scale factors while keeping the encoder/decoder depth
constant. Every other CLI argument from ``train_adaptive_unet.py`` is available.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure the repository root is on sys.path so sibling imports resolve.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from train_adaptive_unet import parse_args as base_parse_args  # noqa: E402
from train_adaptive_unet import train as base_train  # noqa: E402

FIXED_DEPTH = 3


def parse_args():
    """Delegate to the main parser and then pin depth_override to the fixed value."""
    args = base_parse_args()
    if args.depth_override is not None and args.depth_override != FIXED_DEPTH:
        print(
            f"[warn] Ignoring --depth_override={args.depth_override}; "
            f"using fixed depth {FIXED_DEPTH}."
        )
    args.depth_override = FIXED_DEPTH
    return args


def train(args):
    """Call the shared trainer with the fixed depth override in place."""
    args.depth_override = FIXED_DEPTH
    base_train(args)


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments)
