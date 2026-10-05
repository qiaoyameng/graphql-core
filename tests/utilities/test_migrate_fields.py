from graphql import (
    build_schema,
    execute_sync,
    parse,
    print_ast,
    validate,
)
from graphql.utilities import (
    FieldMigrationError,
    FieldMigrationResult,
    migrate_fields,
)

from ..utils import dedent

OLD_SDL = """
interface Character {
  id: ID!
  name: String
}

type Human implements Character {
  id: ID!
  name: String
  height: Float
}

type Droid implements Character {
  id: ID!
  name: String
  primaryFunction: String
}

union SearchResult = Human | Droid

type Villain {
  name: String
}

type Query {
  hero: Character
  human: Human
  villain: Villain
  search: SearchResult
}
"""

NEW_SDL = """
interface Character {
  id: ID!
  fullName: String
}

type Human implements Character {
  id: ID!
  fullName: String
  stature: Float
}

type Droid implements Character {
  id: ID!
  fullName: String
  primaryFunction: String
}

union SearchResult = Human | Droid

type Villain {
  name: String
}

type Query {
  hero: Character
  human: Human
  villain: Villain
  search: SearchResult
}
"""

DOCUMENT = """
query GetHero($withHeight: Boolean!) {
  hero {
    ...HeroFields
    ... on Human {
      height @include(if: $withHeight)
    }
  }
  search {
    ... on Human {
      height
    }
    ... on Droid {
      primaryFunction
    }
  }
}

query GetVillain {
  villain {
    name
  }
  human {
    name
    ...HeroFields
  }
}

fragment HeroFields on Character {
  id
  name
}
"""

FIELD_MAP = {
    "Character.name": "fullName",
    "Human.height": "stature",
}

MIGRATED = """
query GetHero($withHeight: Boolean!) {
  hero {
    ...HeroFields
    ... on Human {
      height: stature @include(if: $withHeight)
    }
  }
  search {
    ... on Human {
      height: stature
    }
    ... on Droid {
      primaryFunction
    }
  }
}

query GetVillain {
  villain {
    name
  }
  human {
    name: fullName
    ...HeroFields
  }
}

fragment HeroFields on Character {
  id
  name: fullName
}
"""

OLD_ROOT = {
    "hero": {"__typename": "Human", "id": "1", "name": "Luke", "height": 172},
    "human": {"id": "1", "name": "Luke", "height": 172},
    "villain": {"name": "Vader"},
    "search": {"__typename": "Droid", "id": "2", "name": "R2-D2", "primaryFunction": "Astromech"},
}

NEW_ROOT = {
    "hero": {"__typename": "Human", "id": "1", "fullName": "Luke", "stature": 172},
    "human": {"id": "1", "fullName": "Luke", "stature": 172},
    "villain": {"name": "Vader"},
    "search": {"__typename": "Droid", "id": "2", "fullName": "R2-D2", "primaryFunction": "Astromech"},
}


def _migrate(document, field_map=FIELD_MAP, old_sdl=OLD_SDL, new_sdl=NEW_SDL):
    old_schema = build_schema(old_sdl)
    new_schema = build_schema(new_sdl)
    return migrate_fields(old_schema, new_schema, parse(document), field_map)


def describe_migrate_fields():
    def rewrites_only_mapped_fields_and_preserves_response_keys():
        result = _migrate(DOCUMENT)
        assert result.errors == ()
        assert print_ast(result.document) == dedent(MIGRATED)

    def does_not_modify_the_original_document():
        document = parse(DOCUMENT)
        printed_before = print_ast(document)
        result = _migrate(DOCUMENT)
        assert result.document is not None
        assert result.document is not document
        assert print_ast(document) == printed_before

    def preserves_existing_aliases():
        document = """
        {
          human {
            nickname: name
            stature: height
          }
        }
        """
        result = _migrate(document)
        assert result.errors == ()
        assert print_ast(result.document) == dedent(
            """
            {
              human {
                nickname: fullName
                stature: stature
              }
            }
            """
        )

    def migrates_a_fragment_shared_by_two_operations_and_keeps_response_keys():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse(DOCUMENT)

        result = migrate_fields(old_schema, new_schema, document, FIELD_MAP)
        assert result.errors == ()
        migrated = result.document

        # The migrated document is valid against the new schema.
        assert validate(new_schema, migrated) == []

        for operation_name in ("GetHero", "GetVillain"):
            variables = {"withHeight": True} if operation_name == "GetHero" else None
            old_result = execute_sync(
                old_schema,
                document,
                root_value=OLD_ROOT,
                operation_name=operation_name,
                variable_values=variables,
            )
            new_result = execute_sync(
                new_schema,
                migrated,
                root_value=NEW_ROOT,
                operation_name=operation_name,
                variable_values=variables,
            )
            assert old_result.errors is None
            assert new_result.errors is None
            # Response keys (and values) are unchanged by the migration.
            assert new_result.data == old_result.data

        hero_result = execute_sync(
            new_schema,
            migrated,
            root_value=NEW_ROOT,
            operation_name="GetHero",
            variable_values={"withHeight": True},
        )
        assert hero_result.data == {
            "hero": {"id": "1", "name": "Luke", "height": 172},
            "search": {"primaryFunction": "Astromech"},
        }

    def reports_a_missing_target_field_in_the_new_schema():
        new_schema = build_schema(NEW_SDL.replace("  stature: Float\n", ""))
        old_schema = build_schema(OLD_SDL)
        document = parse("{ human { height } }")
        result = migrate_fields(old_schema, new_schema, document, FIELD_MAP)
        assert result.document is None
        assert len(result.errors) == 1
        error = result.errors[0]
        assert isinstance(error, FieldMigrationError)
        assert error.coordinate == "Human.height"
        assert "stature" in error.message
        assert "does not exist" in error.message
        assert error.locations == [(1, 11)]

    def reports_a_missing_type_in_the_new_schema():
        new_schema = build_schema(
            NEW_SDL.replace("  villain: Villain\n", "").replace(
                "type Villain {\n  name: String\n}\n\n", ""
            )
        )
        old_schema = build_schema(OLD_SDL)
        document = parse("{ villain { name } }")
        field_map = {"Villain.name": "fullName"}
        result = migrate_fields(old_schema, new_schema, document, field_map)
        assert result.document is None
        assert len(result.errors) == 1
        assert "Villain" in result.errors[0].message
        assert "does not exist in the new schema" in result.errors[0].message

    def reports_incompatible_mappings_across_implementations():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse("{ hero { name } }")
        # Only Human is mapped; Droid keeps the old name.
        field_map = {"Human.name": "fullName"}
        result = migrate_fields(old_schema, new_schema, document, field_map)
        assert result.document is None
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.coordinate == "Character.name"
        assert "incompatible" in error.message
        assert "Human.name" in error.message
        assert "Droid.name" in error.message
        assert error.locations == [(1, 10)]

    def allows_consistent_per_implementation_mappings():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse("{ hero { name } }")
        field_map = {"Human.name": "fullName", "Droid.name": "fullName"}
        result = migrate_fields(old_schema, new_schema, document, field_map)
        assert result.errors == ()
        assert print_ast(result.document) == "{\n  hero {\n    name: fullName\n  }\n}"

    def reports_a_field_conflict_introduced_by_the_migration():
        old_schema = build_schema(
            """
            interface A { common: String }
            interface B { common: String }
            type T implements A & B { common: String }
            type Query { t: T }
            """
        )
        new_schema = build_schema(
            """
            interface A { common: String }
            interface B { common: String }
            type T implements A & B { renamed: String }
            type Query { t: T }
            """
        )
        # `common` selected directly on T is renamed via interface A, while
        # the same field selected via interface B keeps its old name, so both
        # selections end up with the response key `common` for different fields.
        document = parse("{ t { common ... on B { common } } }")
        result = migrate_fields(old_schema, new_schema, document, {"A.common": "renamed"})
        assert result.document is None
        assert len(result.errors) == 1
        error = result.errors[0]
        assert "conflicts" in error.message
        assert "'common'" in error.message
        assert len(error.locations) == 2

    def does_not_report_conflicts_for_mutually_exclusive_types():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse(
            """
            {
              search {
                ... on Human {
                  height
                }
                ... on Droid {
                  primaryFunction
                }
              }
            }
            """
        )
        result = migrate_fields(old_schema, new_schema, document, FIELD_MAP)
        assert result.errors == ()

    def reports_unknown_coordinates_in_the_field_map():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse("{ human { name } }")
        result = migrate_fields(
            old_schema, new_schema, document, {"Human.unknown": "nope"}
        )
        assert result.document is None
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.coordinate == "Human.unknown"
        assert "does not exist" in error.message

    def reports_conflicting_field_map_entries():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse("{ human { height } }")
        field_map = [("Human.height", "stature"), ("Human.height", "size")]
        result = migrate_fields(old_schema, new_schema, document, field_map)
        assert result.document is None
        assert len(result.errors) == 1
        assert "Conflicting field map entries" in result.errors[0].message

    def rejects_invalid_coordinates_and_field_names():
        old_schema = build_schema(OLD_SDL)
        new_schema = build_schema(NEW_SDL)
        document = parse("{ human { name } }")
        for field_map in ({"Human": "x"}, {"Human.name": "not a name"}):
            try:
                migrate_fields(old_schema, new_schema, document, field_map)
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")

    def returns_a_result_named_tuple():
        result = _migrate("{ human { name } }")
        assert isinstance(result, FieldMigrationResult)
        assert result.errors == ()
        assert print_ast(result.document) == "{\n  human {\n    name: fullName\n  }\n}"
