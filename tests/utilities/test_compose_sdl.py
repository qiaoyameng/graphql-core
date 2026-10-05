from graphql import (
    ComposeIssue,
    ComposeSdlError,
    GraphQLInterfaceType,
    GraphQLUnionType,
    assert_valid_schema,
    compose_sdl,
)

from ..utils import dedent


def describe_compose_sdl():
    def merges_non_conflicting_fields_from_definition_and_extension():
        result = compose_sdl(
            [
                ("team-a", "type Query { a: String }"),
                ("team-b", "extend type Query { b: Int }"),
            ]
        )
        assert sorted(result.schema.query_type.fields) == ["a", "b"]
        assert result.notes == ()

    def deduplicates_semantically_identical_declarations():
        result = compose_sdl(
            [
                ("team-a", 'type Query { f(tags: [String!] = ["x"]): [Int!]! }'),
                ("team-b", 'extend type Query { f(tags:[String!]=["x"]):[Int!]! }'),
            ]
        )
        assert list(result.schema.query_type.fields) == ["f"]
        field = result.schema.query_type.fields["f"]
        assert str(field.type) == "[Int!]!"
        assert list(field.args) == ["tags"]

    def merging_does_not_depend_on_definition_or_extension_order():
        extension = "extend type Query { b: Int }"
        definition = "type Query { a: String }"
        result = compose_sdl([("b", extension), ("a", definition)])
        assert sorted(result.schema.query_type.fields) == ["a", "b"]

    def merges_interfaces_unions_enums_and_input_objects():
        result = compose_sdl(
            [
                (
                    "base",
                    dedent(
                        """
                        interface Node { id: ID! }
                        type Book implements Node { id: ID! }
                        type Movie implements Node { id: ID! }
                        union Media = Book
                        enum Status { ACTIVE }
                        input Filter { q: String }
                        type Query {
                          media: Media, status: Status
                          with(filter: Filter): String
                        }
                        """
                    ),
                ),
                (
                    "extra",
                    dedent(
                        """
                        extend type Book implements Node { title: String }
                        extend type Movie implements Node { title: String }
                        extend union Media = Movie
                        extend enum Status { ARCHIVED }
                        extend input Filter { limit: Int = 10 }
                        """
                    ),
                ),
            ]
        )
        schema = result.schema
        assert_valid_schema(schema)
        book = schema.get_type("Book")
        assert "Node" in {interface.name for interface in book.interfaces}
        assert set(book.fields) == {"id", "title"}
        media = schema.get_type("Media")
        assert isinstance(media, GraphQLUnionType)
        assert {type_.name for type_ in media.types} == {"Book", "Movie"}
        assert set(schema.get_type("Status").values) == {"ACTIVE", "ARCHIVED"}
        assert set(schema.get_type("Filter").fields) == {"q", "limit"}

    def keeps_explicit_root_type_assignment():
        result = compose_sdl(
            [
                ("a", "schema { query: Root } type Root { a: String }"),
                ("b", "extend type Root { b: String }"),
            ]
        )
        assert result.schema.query_type.name == "Root"
        assert sorted(result.schema.query_type.fields) == ["a", "b"]

    def synthesizes_a_base_from_extensions_only():
        result = compose_sdl(
            [
                ("a", "extend type Query { a: String }"),
                ("b", "extend type Query { b: Int }"),
            ]
        )
        assert sorted(result.schema.query_type.fields) == ["a", "b"]

    def compares_default_values_semantically():
        result = compose_sdl(
            [
                (
                    "a",
                    "input Filter { tags: [Int] = [1, 2] } "
                    "type Query { f(filter: Filter): String }",
                ),
                ("b", "extend input Filter { tags:[Int]=[1,2] }"),
            ]
        )
        assert result.notes == ()
        ast_node = result.schema.get_type("Filter").fields["tags"].ast_node
        assert [item.value for item in ast_node.default_value.values] == ["1", "2"]

    def different_descriptions_are_notes_not_conflicts():
        result = compose_sdl(
            [
                ("a", '"""first description"""\ntype Query { f: String }'),
                ("b", '"""second description"""\ntype Query { f: String }'),
            ]
        )
        assert len(result.notes) == 1
        note = result.notes[0]
        assert isinstance(note, ComposeIssue)
        assert note.severity == "note"
        assert note.coordinate == "Query"
        assert {loc.source_name for loc in note.locations} == {"a", "b"}

    def reports_field_type_conflict_with_both_sources_and_coordinates():
        try:
            compose_sdl(
                [
                    ("team-a", "type Query { f: String }"),
                    ("team-b", "type Query { f: Int }"),
                ]
            )
        except ComposeSdlError as error:
            assert len(error.conflicts) == 1
            issue = error.conflicts[0]
            assert issue.severity == "conflict"
            assert issue.coordinate == "Query.f"
            assert "String" in issue.message and "Int" in issue.message
            source_names = [loc.source_name for loc in issue.locations]
            assert source_names == ["team-a", "team-b"]
            assert all(loc.line and loc.column for loc in issue.locations)
        else:
            raise AssertionError("expected ComposeSdlError")

    def reports_argument_and_default_value_conflicts():
        def expect_conflict(a_sdl, b_sdl, coordinate):
            try:
                compose_sdl([("a", a_sdl), ("b", b_sdl)])
            except ComposeSdlError as error:
                assert any(
                    issue.coordinate == coordinate for issue in error.conflicts
                ), error.conflicts
            else:
                raise AssertionError("expected ComposeSdlError")

        expect_conflict(
            "type Query { f(x: Int): String }",
            "extend type Query { f(x: String): String }",
            "Query.f(x:)",
        )
        expect_conflict(
            "input I { x: Int = 1 } type Query { f(i: I): String }",
            "extend input I { x: Int = 2 }",
            "I(x:)",
        )

    def reports_enum_value_directive_conflicts():
        try:
            compose_sdl(
                [
                    (
                        "a",
                        'enum E { A @deprecated(reason: "old") } '
                        "type Query { e: E }",
                    ),
                    ("b", 'extend enum E { A @deprecated(reason: "new") }'),
                ]
            )
        except ComposeSdlError as error:
            assert [issue.coordinate for issue in error.conflicts] == ["E.A"]
            assert {loc.source_name for loc in error.conflicts[0].locations} == {
                "a",
                "b",
            }
        else:
            raise AssertionError("expected ComposeSdlError")

    def reports_root_type_assignment_conflicts():
        try:
            compose_sdl(
                [
                    ("a", "schema { query: Q1 } type Q1 { a: String }"),
                    ("b", "schema { query: Q2 } type Q2 { b: String }"),
                ]
            )
        except ComposeSdlError as error:
            assert [issue.coordinate for issue in error.conflicts] == [
                "schema.query"
            ]
        else:
            raise AssertionError("expected ComposeSdlError")

    def reports_kind_mismatch_conflicts():
        try:
            compose_sdl(
                [
                    ("a", "type X { f: String } type Query { x: X }"),
                    ("b", "interface X { g: String }"),
                ]
            )
        except ComposeSdlError as error:
            assert any(
                issue.coordinate == "X" and "kind mismatch" in issue.message
                for issue in error.conflicts
            )
        else:
            raise AssertionError("expected ComposeSdlError")

    def conflict_batch_is_atomic_and_collects_all_conflicts():
        try:
            compose_sdl(
                [
                    ("a", "type Query { f: String, g: String }"),
                    ("b", "type Query { f: Int, g: Int }"),
                ]
            )
        except ComposeSdlError as error:
            coordinates = {issue.coordinate for issue in error.conflicts}
            assert coordinates == {"Query.f", "Query.g"}
            assert not hasattr(error, "schema")
        else:
            raise AssertionError("expected ComposeSdlError")

    def merged_schema_is_validated():
        try:
            compose_sdl(
                [
                    (
                        "a",
                        "interface Node { id: ID! } "
                        "type User implements Node { name: String }",
                    )
                ]
            )
        except ComposeSdlError as error:
            assert "does not pass validation" in error.conflicts[0].message
        else:
            raise AssertionError("expected ComposeSdlError")

    def renders_validated_sdl():
        result = compose_sdl(
            [
                ("a", "type Query { a: String }"),
                ("b", "extend type Query { b: Int }"),
            ]
        )
        assert "a: String" in result.sdl
        assert "b: Int" in result.sdl
