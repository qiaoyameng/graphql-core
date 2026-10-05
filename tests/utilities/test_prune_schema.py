"""Tests for pruning a schema to the parts used by a set of operations."""

from __future__ import annotations

import pytest

from graphql import (
    GraphQLSchema,
    build_schema,
    execute,
    get_named_type,
    parse,
    print_schema,
    prune_schema,
    validate,
    validate_schema,
)


def build_star_wars_like_schema() -> GraphQLSchema:
    return build_schema(
        """
        enum Episode { NEW_HOPE EMPIRE JEDI }

        interface Node {
          id: ID!
        }

        interface Character {
          id: ID!
          name: String!
          friends: [Character]
          appearsIn(limit: Int): [Episode]
        }

        type Human implements Node & Character {
          id: ID!
          name: String!
          friends: [Character]
          appearsIn(limit: Int): [Episode]
          homePlanet: String
          secretBackstory: String
        }

        type Droid implements Node & Character {
          id: ID!
          name: String!
          friends: [Character]
          appearsIn(limit: Int): [Episode]
          primaryFunction: String
        }

        union SearchResult = Human | Droid

        input RecursiveInput {
          label: String
          requiredNote: String!
          children: [RecursiveInput]
        }

        input UnusedInput {
          value: Int!
        }

        type Query {
          hero(episode: Episode = JEDI, filter: RecursiveInput): Character
          search(text: String!): [SearchResult]
          node(id: ID!): Node
          humans: [Human]
          unusedField: String
          unusedInput(unused: UnusedInput): String
        }

        type Mutation {
          createHuman(input: RecursiveInput!): Human
        }

        directive @tag(value: String!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
        directive @clientOnly on FIELD
        directive @auth(role: String, level: Int) on FIELD_DEFINITION
        """
    )


OPERATIONS = parse(
    """
    query HeroAndNode($ep: Episode, $filter: RecursiveInput) {
      hero(episode: $ep, filter: $filter) {
        id
        name
        ...humanFields
        ... on Droid {
          id
          name @tag(value: "droid")
        }
      }
      node(id: "2001") {
        id
        ... on Human {
          name
          appearsIn(limit: 2)
        }
      }
    }

    fragment humanFields on Human {
      friends {
        id
        name
      }
    }

    mutation CreateHuman($input: RecursiveInput!) {
      createHuman(input: $input) {
        id
        name
      }
    }
    """
)


def describe_prune_schema():
    def removes_unrelated_root_fields_and_types():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)

        assert result.report.removed_root_fields == (
            "Query.humans",
            "Query.search",
            "Query.unusedField",
            "Query.unusedInput",
        )
        assert "UnusedInput" in result.report.removed_types
        query = result.schema.query_type
        assert set(query.fields) == {"hero", "node"}

    def keeps_the_pruned_schema_valid():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)

        assert validate_schema(result.schema) == []
        assert validate(result.schema, OPERATIONS) == []

    def keeps_operations_executable():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        pruned = result.schema

        luke = {
            "__typename": "Human",
            "id": "1000",
            "name": "Luke Skywalker",
            "appearsIn": ["NEW_HOPE"],
            "friends": [
                {"__typename": "Human", "id": "1002", "name": "Han Solo"},
                {"__typename": "Droid", "id": "2001", "name": "R2-D2"},
            ],
        }
        for name in ("Human", "Droid"):
            pruned.get_type(name).is_type_of = (
                lambda value, _info, name=name: value.get("__typename") == name
            )
        pruned.query_type.fields["hero"].resolve = lambda *args, **kwargs: luke
        pruned.query_type.fields["node"].resolve = lambda *args, **kwargs: luke
        pruned.mutation_type.fields["createHuman"].resolve = lambda *args, **kwargs: {
            "__typename": "Human",
            "id": "1001",
            "name": "Darth Vader",
        }

        variables = {
            "ep": "JEDI",
            "filter": {
                "label": "jedi",
                "requiredNote": "keep me",
                "children": [
                    {"label": "nested", "requiredNote": "also required"}
                ],
            },
            "input": {"label": "vader", "requiredNote": "sith"},
        }
        query_result = execute(
            pruned, OPERATIONS, variable_values=variables,
            operation_name="HeroAndNode",
        )
        assert query_result.errors is None
        assert query_result.data == {
            "hero": {
                "id": "1000",
                "name": "Luke Skywalker",
                "friends": [
                    {"id": "1002", "name": "Han Solo"},
                    {"id": "2001", "name": "R2-D2"},
                ],
            },
            "node": {
                "id": "1000",
                "name": "Luke Skywalker",
                "appearsIn": ["NEW_HOPE"],
            },
        }

        mutation_result = execute(
            pruned, OPERATIONS, variable_values=variables,
            operation_name="CreateHuman",
        )
        assert mutation_result.errors is None
        assert mutation_result.data == {
            "createHuman": {"id": "1001", "name": "Darth Vader"}
        }

    def keeps_complete_recursive_input_field_contract():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)

        recursive_input = result.schema.get_type("RecursiveInput")
        assert recursive_input is not None
        assert set(recursive_input.fields) == {
            "label",
            "requiredNote",
            "children",
        }
        # The recursive reference is closed with the very same type instance.
        children_type = recursive_input.fields["children"].type
        assert get_named_type(children_type) is recursive_input

    def closes_interface_implementations_and_union_members():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        pruned = result.schema

        node = pruned.get_type("Node")
        human = pruned.get_type("Human")
        droid = pruned.get_type("Droid")
        assert {interface.name for interface in human.interfaces} == {
            "Node",
            "Character",
        }
        assert {interface.name for interface in droid.interfaces} == {
            "Node",
            "Character",
        }
        # Interface fields required by the implementation relationship are kept.
        assert "id" in human.fields
        assert "id" in droid.fields
        assert (
            get_named_type(node.fields["id"].type)
            is get_named_type(human.fields["id"].type)
        )

        # The Character interface is retained because Human implements it. It
        # must declare every field and argument used on a retained implementation
        # (here friends and appearsIn(limit:)) so the implementation relationship
        # stays legal, even though friends is only selected concretely on Human.
        character = pruned.get_type("Character")
        assert character is not None
        assert set(character.fields) == {
            "id",
            "name",
            "friends",
            "appearsIn",
        }
        assert "limit" in character.fields["appearsIn"].args
        # Interface field arguments are propagated to every implementation.
        assert "limit" in human.fields["appearsIn"].args
        assert "limit" in droid.fields["appearsIn"].args
        # The Human type keeps its concretely selected recursive field.
        assert "friends" in human.fields
        # Fields not selected anywhere on the retained contract are removed.
        assert "primaryFunction" not in droid.fields
        assert "secretBackstory" not in human.fields

        interface_notes = [
            note for note in result.report.notes
            if "implementations of interface 'Node'" in note
        ]
        assert interface_notes

    def does_not_leave_empty_types_or_dangling_references():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        type_map = result.schema.type_map

        assert "UnusedInput" not in type_map
        for name, type_ in type_map.items():
            fields = getattr(type_, "fields", None)
            if isinstance(fields, dict):
                assert fields, f"{name} must not be empty"
            if hasattr(type_, "types"):
                assert type_.types, f"union {name} must have members"

    def keeps_enum_values_and_referenced_default_values():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        episode = result.schema.get_type("Episode")

        assert set(episode.values) == {"NEW_HOPE", "EMPIRE", "JEDI"}
        # The default value reference (JEDI) is part of the retained argument.
        hero_argument = result.schema.query_type.fields["hero"].args["episode"]
        assert hero_argument.default is not None
        assert hero_argument.default.literal is not None
        assert hero_argument.default.literal.value == "JEDI"

    def prunes_unused_arguments_and_directives():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)

        directives = {
            directive.name for directive in result.schema.directives
        }
        assert "@auth".lstrip("@") not in directives
        assert "clientOnly" not in directives
        assert "tag" in directives
        # The level/role arguments of @auth are gone with the directive; the
        # retained @tag keeps only the argument the operations use.
        tag = next(
            directive for directive in result.schema.directives
            if directive.name == "tag"
        )
        assert set(tag.args) == {"value"}
        # Unused field arguments are removed while used ones are retained.
        assert set(result.schema.query_type.fields["hero"].args) == {
            "episode",
            "filter",
        }

    def reports_necessary_retentions():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        notes = "\n".join(result.report.notes)

        assert "complete field contract of input object 'RecursiveInput'" in notes
        assert "all values of enum 'Episode'" in notes
        assert "implementations of interface 'Node'" in notes

    def raises_type_error_for_invalid_operations():
        schema = build_star_wars_like_schema()
        with pytest.raises(TypeError) as exc_info:
            prune_schema(schema, parse("{ nonExistent }"))
        assert "invalid operations" in str(exc_info.value)

    def accepts_multiple_documents():
        schema = build_schema(
            """
            type Query {
              alpha: String
              beta: Int
              gamma: Float
            }
            """
        )
        result = prune_schema(schema, [parse("{ alpha }"), parse("{ beta }")])
        assert set(result.schema.query_type.fields) == {"alpha", "beta"}
        assert result.report.removed_root_fields == ("Query.gamma",)

    def retains_a_minimal_query_root_for_mutation_only_schemas():
        schema = build_schema(
            """
            type Query {
              hello: String
              unused: Int
            }
            type Mutation {
              ping: String
            }
            """
        )
        result = prune_schema(schema, parse("mutation { ping }"))

        assert validate_schema(result.schema) == []
        assert result.schema.mutation_type is not None
        assert set(result.schema.mutation_type.fields) == {"ping"}
        # A Query root must still exist and define a field.
        assert result.schema.query_type is not None
        assert result.schema.query_type.fields
        assert any(
            "Query root type" in note for note in result.report.notes
        )

    def anchors_a_retained_root_type_when_only_typename_is_selected():
        # Selecting only __typename keeps the root type but none of its fields;
        # the pruner retains one anchor field to avoid an empty object type.
        schema = build_schema(
            """
            type Query {
              hello: String
              world: Int
            }
            """
        )
        result = prune_schema(schema, parse("{ __typename }"))

        assert validate_schema(result.schema) == []
        assert set(result.schema.query_type.fields) == {"hello"}
        assert any(
            "Retained field 'Query.hello'" in note
            for note in result.report.notes
        )

    def closes_union_members_without_leaving_empty_types():
        schema = build_schema(
            """
            type Cat { sound: String, whiskers: Int }
            type Dog { sound: String, tail: Int }
            union Pet = Cat | Dog
            type Query { pet: Pet }
            """
        )
        document = parse("{ pet { __typename } }")
        result = prune_schema(schema, document)

        assert validate_schema(result.schema) == []
        assert validate(result.schema, document) == []
        cat = result.schema.get_type("Cat")
        dog = result.schema.get_type("Dog")
        assert set(cat.fields) == {"sound"}
        assert set(dog.fields) == {"sound"}
        assert "Cat.whiskers" in result.report.removed_fields
        assert "Dog.tail" in result.report.removed_fields
        notes = "\n".join(result.report.notes)
        assert "Retained field 'Cat.sound'" in notes
        assert "all members of union 'Pet'" in notes

    def prunes_directive_arguments_and_their_types():
        schema = build_schema(
            """
            input UsedInput { x: String }
            input UnusedInput { y: Int! }
            directive @tagged(data: UsedInput, secret: UnusedInput) on FIELD
            type Query { f: String }
            """
        )
        document = parse('{ f @tagged(data: {x: "hi"}) }')
        result = prune_schema(schema, document)

        assert validate(result.schema, document) == []
        tagged = next(
            directive for directive in result.schema.directives
            if directive.name == "tagged"
        )
        assert set(tagged.args) == {"data"}
        assert "UsedInput" in result.schema.type_map
        assert "UnusedInput" not in result.schema.type_map

    def keeps_recursive_output_type_cycle():
        schema = build_schema(
            """
            type Tree {
              value: Int!
              left: Tree
              right: Tree
              ignored: String
            }
            type Query { root: Tree! other: Tree }
            """
        )
        document = parse("{ root { value left { value left { value } } } }")
        result = prune_schema(schema, document)

        assert validate_schema(result.schema) == []
        assert validate(result.schema, document) == []
        tree = result.schema.get_type("Tree")
        assert set(tree.fields) == {"value", "left"}
        assert get_named_type(tree.fields["left"].type) is tree
        assert result.report.removed_root_fields == ("Query.other",)
        assert "Tree.ignored" in result.report.removed_fields

    def does_not_invoke_resolvers_while_pruning():
        schema = build_schema("type Query { value: String }")

        def resolver(*_args):
            raise AssertionError("resolvers must not run during pruning")

        schema.query_type.fields["value"].resolve = resolver
        result = prune_schema(schema, parse("{ value }"))
        assert validate_schema(result.schema) == []

    def printed_schema_is_self_contained():
        schema = build_star_wars_like_schema()
        result = prune_schema(schema, OPERATIONS)
        printed = print_schema(result.schema)

        assert "secretBackstory" not in printed
        assert "primaryFunction" not in printed
        assert "homePlanet" not in printed
        assert "UnusedInput" not in printed
        assert "@auth" not in printed
        assert "RecursiveInput" in printed
