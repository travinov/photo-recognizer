from __future__ import annotations

import argparse

from photo_recognizer.config import load_settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.services import IndexService


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo Recognizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Scan the dataset and build the SQLite face index")
    index_parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep the current database and refresh records in place",
    )

    args = parser.parse_args()
    settings = load_settings()
    repository = FaceRepository(settings.db_path)
    service = IndexService(settings, repository)

    if args.command == "index":
        summary = service.rebuild(reset=not args.no_reset)
        print(f"Indexed {summary['photos_indexed']} photos and {summary['faces_indexed']} faces.")


if __name__ == "__main__":
    main()
