"""Console-script launcher for the PdfAgent browser UI.

Installed as the `pdfagent` entry point (see pyproject.toml). Boots the
FastAPI app via uvicorn so end users do not need to know the module path.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdfagent",
        description="Launch the PdfAgent browser UI (FastAPI + uvicorn).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="enable auto-reload (dev)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "uvicorn is required but not installed. Run: pip install uvicorn\n"
        )
        sys.exit(1)

    print(f"PdfAgent UI starting on http://{args.host}:{args.port}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
