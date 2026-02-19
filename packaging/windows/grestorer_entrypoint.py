# packaging/windows/grestorer_entrypoint.py
from __future__ import annotations

def main() -> int:
    # Import as a package module so relative imports inside gRestorer.cli.main work.
    from gRestorer.cli.main import main as _main
    return int(_main())

if __name__ == "__main__":
    raise SystemExit(main())
