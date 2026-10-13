"""Allow ``python -m omictopo`` as an alias for the ``omictopo`` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
