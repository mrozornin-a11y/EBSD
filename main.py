"""Точка входа приложения (используется также PyInstaller)."""
import sys

from ebsd_pf.app import main

if __name__ == "__main__":
    sys.exit(main())
