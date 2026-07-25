"""Allow ``python -m src.generator`` without the RuntimeWarning."""

from .generator import main

if __name__ == "__main__":
    main()
