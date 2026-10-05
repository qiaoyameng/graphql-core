from pathlib import Path

import pytest

from graphql import (
    GraphQLError,
    compose_sdl,
    lexicographic_sort_schema,
    print_schema,
    validate_schema,
)

from ..utils import dedent

EXAMPLES = Path(__file__).parents[2] / "examples" / "sdl_composition"


def load_examples(*names):
    return [(name, (EXAMPLES / name).read_text()) for name in names]


def describe_compose_sdl():
    def merges_non_conflicting_fields_from_multiple_documents():
        result = compose_sdl([
            ("a.graphql", "type Query { a: Int }"),
            ("b.graphql", "extend type Query { b: String }"),
        ])
        assert result.ok
        assert result.conflicts == ()
        assert print_schema(result.schema) == dedent(
            """
            type Query {
              a: Int
              b: String
            }
            """
        )

    def keeps_identical_declarations_only_once():
        result = compose_sdl([
            ("a.graphql", "type Query { user: User } type User { id: ID! }"),
            ("b.graphql", "type User { id: ID! } extend type Query { user: User }"),
        ])
        assert result.ok
        assert print_schema(result.schema) == dedent(
            """
            type Query {
              user: User
            }

            type User {
              id: ID!
            }
            """
        )

    def compares_defaults_and_values_semantically_not_textually():
        result = compose_sdl([
            (
                "a.graphql",
                "type Query { f(filter: Filter = {a: 1, b: [1, 2]}): Int }"
                " input Filter { a: Int, b: [Int] }",
            ),
            (
                "b.graphql",
                # same default, different key order and formatting
                "type Query { f(filter: Filter = { b: [1, 2], a: 1 }): Int }"
                " input Filter { a: Int, b: [Int] }",
            ),
        ])
        assert result.ok
        assert result.conflicts == ()

    def treats_block_and_regular_strings_as_equal():
        result = compose_sdl([
            ("a.graphql", 'type Query { f(arg: String = "x"): Int }'),
            ("b.graphql", 'type Query { f(arg: String = """x"""): Int }'),
        ])
        assert result.ok
        assert result.conflicts == ()

    def detects_conflicting_wrapped_types():
        result = compose_sdl([
            ("a.graphql", "type Query { tags: [String!] }"),
            ("b.graphql", "type Query { tags: [String] }"),
        ])
        assert not result.ok
        assert result.schema is None
        (conflict,) = result.conflicts
        assert conflict.coordinate == "Query.tags"
        assert "[String!]" in conflict.message

    def detects_conflicting_default_values():
        result = compose_sdl([
            ("a.graphql", "type Query { f(limit: Int = 10): Int }"),
            ("b.graphql", "type Query { f(limit: Int = 20): Int }"),
        ])
        assert not result.ok
        (conflict,) = result.conflicts
        assert conflict.coordinate == "Query.f"
        assert "limit" in conflict.message

    def compares_directive_applications_semantically():
        result = compose_sdl([
            (
                "a.graphql",
                "directive @tag(name: String!, prio: Int) on FIELD_DEFINITION"
                " type Query { f: Int @tag(name: \"a\", prio: 1) }",
            ),
            (
                "b.graphql",
                "directive @tag(name: String!, prio: Int) on FIELD_DEFINITION"
                # same directive, different argument order
                " type Query { f: Int @tag(prio: 1, name: \"a\") }",
            ),
        ])
        assert result.ok
        assert result.conflicts == ()

    def detects_conflicting_directive_applications():
        result = compose_sdl([
            (
                "a.graphql",
                "directive @tag(name: String!) on FIELD_DEFINITION"
                " type Query { f: Int @tag(name: \"a\") }",
            ),
            (
                "b.graphql",
                "directive @tag(name: String!) on FIELD_DEFINITION"
                " type Query { f: Int @tag(name: \"b\") }",
            ),
        ])
        assert not result.ok
        (conflict,) = result.conflicts
        assert conflict.coordinate == "Query.f"

    def reports_description_differences_separately():
        result = compose_sdl([
            ("a.graphql", '"First." type Query { f: Int }'),
            ("b.graphql", '"Second." type Query { f: Int }'),
        ])
        assert result.ok
        assert result.conflicts == ()
        (difference,) = result.description_differences
        assert difference.coordinate == "Query"
        assert difference.first.source == "a.graphql"
        assert difference.first_description == "First."
        assert difference.second.source == "b.graphql"
        assert difference.second_description == "Second."

    def definition_and_extension_order_does_not_matter():
        documents = [
            ("a.graphql", "extend type Query { b: String }"),
            ("b.graphql", "type Query { a: Int }"),
        ]
        forward = compose_sdl(documents)
        backward = compose_sdl(list(reversed(documents)))
        assert forward.ok and backward.ok
        assert print_schema(lexicographic_sort_schema(forward.schema)) == (
            print_schema(lexicographic_sort_schema(backward.schema))
        )

    def conflicts_report_both_sources_and_coordinates():
        result = compose_sdl([
            ("team_a.graphql", "type Query {\n  price: Int\n}"),
            ("team_b.graphql", "type Query {\n  price: Float\n}"),
        ])
        assert not result.ok
        assert result.schema is None
        (conflict,) = result.conflicts
        assert conflict.coordinate == "Query.price"
        assert conflict.first.source == "team_a.graphql"
        assert (conflict.first.line, conflict.first.column) == (2, 3)
        assert conflict.second.source == "team_b.graphql"
        assert (conflict.second.line, conflict.second.column) == (2, 3)
        rendered = str(conflict)
        assert "team_a.graphql:2:3" in rendered
        assert "team_b.graphql:2:3" in rendered

    def detects_conflicting_type_kinds():
        result = compose_sdl([
            ("a.graphql", "type Thing { id: ID }"),
            ("b.graphql", "interface Thing { id: ID }"),
        ])
        assert not result.ok
        (conflict,) = result.conflicts
        assert conflict.coordinate == "Thing"

    def merges_enum_values_and_detects_value_conflicts():
        result = compose_sdl([
            ("a.graphql", "enum Color { RED GREEN } type Query { c: Color }"),
            ("b.graphql", "enum Color { BLUE } type Query { c: Color }"),
        ])
        assert result.ok
        enum_type = result.schema.type_map["Color"]
        assert list(enum_type.values) == ["RED", "GREEN", "BLUE"]

    def merges_union_members_and_interfaces():
        result = compose_sdl([
            (
                "a.graphql",
                "interface Node { id: ID! }"
                " type A implements Node { id: ID! }"
                " union U = A"
                " type Query { u: U }",
            ),
            (
                "b.graphql",
                "type B implements Node { id: ID! }"
                " union U = B"
                " extend type A { extra: String }",
            ),
        ])
        assert result.ok
        schema = result.schema
        assert list(schema.type_map["U"].types) == [
            schema.type_map["A"],
            schema.type_map["B"],
        ]
        assert list(schema.type_map["A"].interfaces) == [schema.type_map["Node"]]
        assert "extra" in schema.type_map["A"].fields

    def preserves_root_operation_type_specifications():
        result = compose_sdl([
            (
                "a.graphql",
                "schema { query: MyQuery } type MyQuery { a: Int }",
            ),
            (
                "b.graphql",
                "extend schema { mutation: MyMutation } type MyMutation { b: Int }",
            ),
        ])
        assert result.ok
        schema = result.schema
        assert schema.query_type is schema.type_map["MyQuery"]
        assert schema.mutation_type is schema.type_map["MyMutation"]

    def detects_conflicting_root_operation_types():
        result = compose_sdl([
            ("a.graphql", "schema { query: A } type A { x: Int }"),
            ("b.graphql", "schema { query: B } type B { x: Int }"),
        ])
        assert not result.ok
        (conflict,) = result.conflicts
        assert conflict.coordinate == "schema.query"

    def output_passes_schema_validation():
        result = compose_sdl([
            ("a.graphql", "type Query { a: Int }"),
            ("b.graphql", "extend type Query { b: String }"),
        ])
        assert result.ok
        assert validate_schema(result.schema) == []

    def invalid_merged_schema_is_not_returned():
        result = compose_sdl([
            ("a.graphql", "type Query { a: Int }"),
            ("b.graphql", "type Empty"),
        ])
        assert not result.ok
        assert result.schema is None
        assert result.validation_errors

    def rejects_executable_documents():
        with pytest.raises(GraphQLError) as exc_info:
            compose_sdl([("op.graphql", "{ field }")])
        assert "only accepts schema definition language" in str(exc_info.value)

    def accepts_a_mapping_of_source_names_to_sdl():
        result = compose_sdl({
            "a.graphql": "type Query { a: Int }",
            "b.graphql": "extend type Query { b: String }",
        })
        assert result.ok

    def describe_example_files():
        def base_and_catalog_compose_cleanly():
            result = compose_sdl(
                load_examples("team_base.graphql", "team_catalog.graphql")
            )
            assert result.ok
            assert validate_schema(result.schema) == []

        def billing_conflicts_point_back_to_both_files():
            result = compose_sdl(load_examples(
                "team_base.graphql",
                "team_catalog.graphql",
                "team_billing.graphql",
            ))
            assert not result.ok
            assert result.schema is None
            (conflict,) = result.conflicts
            assert conflict.coordinate == "Product.price"
            assert conflict.first.source == "team_catalog.graphql"
            assert conflict.second.source == "team_billing.graphql"
            (difference,) = result.description_differences
            assert difference.coordinate == "Product"
            assert difference.first.source == "team_catalog.graphql"
            assert difference.second.source == "team_billing.graphql"
