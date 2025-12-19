"""
gRestorer CLI entry point

Run:
  python -m gRestorer.cli --input ... --output ...   (restorer pipeline)
  python -m gRestorer.cli mosaic --input ... --output ...  (synthetic mosaic generator)
"""

from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0].lower() == "mosaic":
        from .mosaic import main as mosaic_main
        return int(mosaic_main(argv[1:]))
    else:
        from .main import main as pipeline_main
        return int(pipeline_main())


if __name__ == "__main__":
    raise SystemExit(main())
