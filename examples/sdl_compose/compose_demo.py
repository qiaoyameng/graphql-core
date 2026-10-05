"""Demo: compose SDL fragments delivered by different teams.

Run from the repository root::

    python examples/sdl_compose/compose_demo.py            # show the conflict
    python examples/sdl_compose/compose_demo.py --merged   # show a clean merge

The three GraphQL files in this directory play the roles requested:

1. ``01_base_dependencies.graphql`` introduces shared dependencies (interfaces,
   unions, enums, inputs, directives and the root type assignment).
2. ``02_team_extension.graphql`` adds fields/members and re-declares one field
   identically (with a different description).
3. ``03_team_conflict.graphql`` creates a real conflict on ``Query.media`` by
   changing the wrapper type from ``[Media!]!`` to ``Media``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from graphql import ComposeSdlError, SdlDocument, compose_sdl

HERE = Path(__file__).resolve().parent


def _read(name: str) -> SdlDocument:
    return SdlDocument(name, (HERE / name).read_text())


def main() -> int:
    show_merged = "--merged" in sys.argv

    if show_merged:
        documents = [
            _read("01_base_dependencies.graphql"),
            _read("02_team_extension.graphql"),
        ]
        result = compose_sdl(documents)
        print("Merged schema (validated):\n")
        print(result.sdl)
        if result.notes:
            print("\nNon-conflicting notes:")
            for note in result.notes:
                print(f"  - {note}")
        return 0

    documents = [
        _read("01_base_dependencies.graphql"),
        _read("02_team_extension.graphql"),
        _read("03_team_conflict.graphql"),
    ]
    try:
        compose_sdl(documents)
    except ComposeSdlError as error:
        print(error)
        print(
            "\nNo schema was returned: the batch is atomic, so the conflicting "
            "fragment must be fixed before composition can succeed."
        )
        return 1

    print("Unexpectedly composed without conflicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
