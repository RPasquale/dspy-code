"""Run the DSPy agent as a module.

This allows `python -m dspy_agent` to behave identically to invoking the
installed `dspy-agent` console script.
"""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

