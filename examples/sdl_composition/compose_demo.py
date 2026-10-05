"""Demo: compose SDL fragments delivered by different teams.

Run with: python examples/sdl_composition/compose_demo.py
"""

from pathlib import Path

from graphql import compose_sdl, print_schema

HERE = Path(__file__).parent


def load(*names: str):
    return [(name, (HERE / name).read_text(encoding="utf-8")) for name in names]


def main() -> None:
    print("== base + catalog (composes cleanly) ==")
    result = compose_sdl(load("team_base.graphql", "team_catalog.graphql"))
    assert result.ok, result.conflicts
    print(print_schema(result.schema))

    print("== base + catalog + billing (conflict) ==")
    result = compose_sdl(
        load("team_base.graphql", "team_catalog.graphql", "team_billing.graphql")
    )
    if result.ok:
        print("unexpectedly composed")
        return
    print(f"schema returned: {result.schema}")  # None: no partial schema
    for conflict in result.conflicts:
        print(conflict)
    for difference in result.description_differences:
        print(difference)


if __name__ == "__main__":
    main()
