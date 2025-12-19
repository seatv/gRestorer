# gRestorer/cli/main.py
from __future__ import annotations

import sys

from .config import parse_args
from .pipeline import Pipeline


def main() -> int:
    try:
        cfg = parse_args()
        pipeline = Pipeline(cfg)
        pipeline.run()
        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
