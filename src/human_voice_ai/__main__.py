#!/usr/bin/env python3
"""
Main entry point for the Human Voice AI application.
"""


def main():
    """Run the Streamlit application."""
    import streamlit.cli as stcli
    from pathlib import Path

    # Get the absolute path to the app.py file
    app_path = Path(__file__).parent.parent / "app.py"

    # Run the Streamlit app
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    import sys

    main()
