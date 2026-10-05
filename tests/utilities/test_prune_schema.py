from __future__ import annotations

import pytest

from graphql import (
    GraphQLArgument,
    GraphQLDefaultInput,
    GraphQLField,
    GraphQLID,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    assert_valid_schema,
    build_schema,
    execute,
    get_introspection_query,
    parse,
    print_schema,
    validate,
)
from graphql.utilities.prune_schema import (
    PrunedSchema,
    PruningReport,
    prune_schema,
)


def test_prunes_unrelated_root_fields_and_types():
    schema = build_schema(
        """
        type User { id: ID! name: String email: String }
        type Secret { token: String }
        type Query {
          user(id: ID!): User
          secret: Secret
          ping: Boolean
        }
        """
    )
    result = prune_schema(schema, '{ user(id: "1") { id name } }')

    assert isinstance(result, PrunedSchema)
    assert isinstance(result.report, PruningReport)
    assert_valid_schema(result.schema)

    query_fields = result.schema.query_type.fields
    assert set(query_fields) == {"user"}
    assert "Secret" not in result.schema.type_map
    assert set(result.schema.type_map["User"].fields) == {"id", "name"}
    assert result.report.removed_root_fields == {
        "Query": ["secret", "ping"]
    }
    assert "Secret" in result.report.removed_types


def test_keeps_full_recursive_input_contract_independent_of_variables():
    tree_filter = GraphQLInputObjectType("TreeFilter", {})
    tree_filter._fields = {
        "keyword": GraphQLInputField(GraphQLString),
        "depth": GraphQLInputField(GraphQLString),
        "child": GraphQLInputField(tree_filter),
        "children": GraphQLInputField(GraphQLList(GraphQLNonNull(tree_filter))),
        "requiredCode": GraphQLInputField(GraphQLNonNull(GraphQLString)),
    }

    query = GraphQLObjectType(
        "Query",
        {
            "find": GraphQLField(
                GraphQLString,
                args={"criteria": GraphQLArgument(tree_filter)},
                resolve=lambda _root, _info, criteria: criteria["requiredCode"],
            )
        },
    )
    schema = GraphQLSchema(query=query)
    document = parse(
        "query Q($c: TreeFilter) { find(criteria: $c) }"
    )
    variables = {"c": {"requiredCode": "ROOT", "child": {"requiredCode": "CH"}}}

    result = prune_schema(schema, document, variables)
    pruned = result.schema
    assert_valid_schema(pruned)
    assert not validate(pruned, document)

    # The recursive input type survives and keeps every field, including
    # optional fields absent from the sample variables.
    pruned_filter = pruned.type_map["TreeFilter"]
    assert set(pruned_filter.fields) == {
        "keyword",
        "depth",
        "child",
        "children",
        "requiredCode",
    }
    assert pruned_filter.fields["child"].type is pruned_filter
    assert str(pruned_filter.fields["children"].type) == "[TreeFilter!]"

    execution = execute(
        pruned,
        document,
        variable_values={
            # Variables are allowed to use fields that never appeared in the
            # original samples because the full contract was retained.
            "c": {
                "requiredCode": "ROOT",
                "keyword": "k",
                "depth": "deep",
                "children": [{"requiredCode": "A"}],
            }
        },
    )
    assert execution.errors is None
    assert execution.data == {"find": "ROOT"}


def test_interface_fragments_keep_implementation_relations_valid():
    schema = build_schema(
        """
        interface Node { id: ID! label: String }
        interface Searchable implements Node {
          id: ID!
          label: String
          score: Float
        }
        type User implements Node & Searchable {
          id: ID!
          label: String
          score: Float
          email: String
        }
        type Post implements Node {
          id: ID!
          label: String
          body: String
        }
        type Company implements Node {
          id: ID!
          label: String
        }
        union Result = User | Post | Company
        type Query {
          result: Result
          all: [Node!]!
        }
        """
    )
    document = parse(
        """
        query {
          result {
            __typename
            ... on Searchable {
              score
              ... on User { label email }
            }
            ... on Post { id }
          }
        }
        """
    )

    result = prune_schema(schema, document)
    pruned = result.schema

    assert_valid_schema(pruned)
    assert not validate(pruned, document)

    user = pruned.type_map["User"]
    searchable = pruned.type_map["Searchable"]
    post = pruned.type_map["Post"]
    assert searchable.name in {iface.name for iface in user.interfaces}
    # Fields of the interface fragment (score) and of the object fragment
    # (label, email) are all retained on the concrete type.
    assert set(user.fields) == {"score", "label", "email"}
    assert set(post.fields) == {"id"}
    # Company is not referenced by any fragment and gets removed from the union.
    assert "Company" not in pruned.type_map
    assert "Company" in result.report.removed_union_members["Result"]
    assert "all" not in pruned.query_type.fields
    # The unused Node interface is removed; Post's dangling implements edge is
    # removed as well rather than forcing the interface to survive.
    assert "Node" not in pruned.type_map
    assert post.interfaces == ()


def test_interface_arguments_are_synchronized_on_implementation():
    schema = build_schema(
        """
        interface Node {
          picture(size: Int!, format: String = "png"): String
        }
        type User implements Node {
          picture(size: Int!, format: String = "png"): String
          name: String
        }
        type Query { node: Node }
        """
    )
    document = parse("{ node { picture(size: 42) } }")
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    user_picture = result.schema.type_map["User"].fields["picture"]
    assert set(user_picture.args) == {"size", "format"}


def test_transitive_interface_chain_is_closed_for_required_edge():
    schema = build_schema(
        """
        interface A { x: String }
        interface B implements A { x: String y: String }
        type T implements A & B { x: String y: String z: String }
        type Query { b: B }
        """
    )
    # Unused ancestor interface and its implements edge are removed cleanly.
    pruned_unused = prune_schema(schema, parse("{ b { ... on B { y } } }"))
    assert_valid_schema(pruned_unused.schema)
    assert "A" not in pruned_unused.schema.type_map
    assert {
        iface.name
        for iface in pruned_unused.schema.type_map["T"].interfaces
    } == {"B"}

    # When both interfaces are used by fragments, the transitive chain
    # B implements A and T implements A & B must stay intact.
    document = parse(
        "{ b { ... on A { x } ... on B { y } } }"
    )
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    assert not validate(result.schema, document)

    t_type = result.schema.type_map["T"]
    b_type = result.schema.type_map["B"]
    a_type = result.schema.type_map["A"]
    assert {iface.name for iface in t_type.interfaces} == {"A", "B"}
    assert {iface.name for iface in b_type.interfaces} == {"A"}
    assert set(t_type.fields) == {"x", "y"}
    assert "z" not in t_type.fields


def test_interface_return_type_keeps_implementors_and_field_arguments():
    schema = build_schema(
        """
        interface Node {
          id: ID!
          picture(size: Int!, format: String = "png"): String
        }
        type User implements Node {
          id: ID!
          picture(size: Int!, format: String = "png"): String
          name: String
        }
        type Bot implements Node {
          id: ID!
          picture(size: Int!, format: String = "png"): String
        }
        type Query { node: Node dropped: Boolean }
        """
    )
    document = parse("{ node { id picture(size: 42) } }")
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    assert not validate(result.schema, document)

    # No fragment mentions User or Bot, but the interface may resolve to
    # either at runtime, so both concrete types survive.
    assert set(result.schema.type_map) >= {"Node", "User", "Bot"}
    for implementor_name in ("User", "Bot"):
        implementor = result.schema.type_map[implementor_name]
        assert "id" in implementor.fields
        picture = implementor.fields["picture"]
        # The used argument and the defaulted argument both survive, keeping
        # the interface implementation valid.
        assert set(picture.args) == {"size", "format"}
    # Unselected concrete fields are pruned.
    assert "name" not in result.schema.type_map["User"].fields


def test_default_value_type_dependencies_are_closed():
    schema = build_schema(
        """
        enum Mode { FAST SLOW }
        input Options {
          mode: Mode = FAST
          note: String
          unused: Int
        }
        type Query {
          run(options: Options = { mode: SLOW, note: "n" }): String
          other: Int
        }
        """
    )
    result = prune_schema(schema, "{ run }")
    assert_valid_schema(result.schema)
    assert "Mode" in result.schema.type_map
    options = result.schema.type_map["Options"]
    assert set(options.fields) == {"mode", "note"}
    printed = print_schema(result.schema)
    assert 'mode: Mode = FAST' in printed
    assert "other" not in printed


def test_literal_input_values_prune_unused_optional_fields():
    schema = build_schema(
        """
        input Filter {
          used: String
          unused: String
          required: String!
        }
        type Query { search(filter: Filter): String dropped: Boolean }
        """
    )
    document = parse('{ search(filter: {used: "u", required: "r"}) }')
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    assert set(result.schema.type_map["Filter"].fields) == {"used", "required"}


def test_directives_are_pruned_with_required_arguments():
    schema = build_schema(
        """
        directive @auth(role: String!, scope: String) on FIELD
        directive @unused on FIELD
        type Item { id: ID! }
        type Query { item: Item other: Item }
        """
    )
    document = parse('{ item @auth(role: "admin") { id } }')
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)

    directive_map = {d.name: d for d in result.schema.directives}
    assert "auth" in directive_map
    assert set(directive_map["auth"].args) == {"role"}
    assert "unused" not in directive_map
    # Standard directives always survive.
    assert {"include", "skip"} <= set(directive_map)
    assert "other" not in result.schema.query_type.fields


def test_mutation_and_subscription_roots_are_independent():
    schema = build_schema(
        """
        type Query { q: String droppedQ: String }
        type Mutation { m: String droppedM: String }
        type Subscription { s: String droppedS: String }
        """
    )
    result = prune_schema(
        schema,
        [
            parse("query { q }"),
            parse("mutation { m }"),
            parse("subscription { s }"),
        ],
    )
    assert_valid_schema(result.schema)
    assert set(result.schema.query_type.fields) == {"q"}
    assert set(result.schema.mutation_type.fields) == {"m"}
    assert set(result.schema.subscription_type.fields) == {"s"}


def test_typename_only_object_kept_valid_with_backstop_note():
    schema = build_schema(
        """
        type Thing { a: String b: Int }
        type Query { things: [Thing!]! dropped: Boolean }
        """
    )
    result = prune_schema(schema, "{ things { __typename } }")
    assert_valid_schema(result.schema)
    assert set(result.schema.type_map["Thing"].fields) == {"a", "b"}
    assert any("Thing" in note for note in result.report.notes)


def test_input_object_with_only_optional_literal_fields_is_backstopped():
    # An input object that is referenced but ends up with zero retained fields
    # is invalid; in that case all fields are retained and the report explains
    # why.
    schema = build_schema(
        """
        input OptionalInput { a: String b: Int }
        type Query {
          use(input: OptionalInput!): String
          dropped: Int
        }
        """
    )
    result = prune_schema(schema, "{ use(input: {}) }")
    assert_valid_schema(result.schema)
    assert set(result.schema.type_map["OptionalInput"].fields) == {"a", "b"}
    assert any("OptionalInput" in note for note in result.report.notes)


def test_operations_remain_executable_with_resolvers_copied():
    node = GraphQLInterfaceType(
        "Node", {"id": GraphQLField(GraphQLNonNull(GraphQLID))}
    )
    user = GraphQLObjectType(
        "User",
        {
            "id": GraphQLField(GraphQLNonNull(GraphQLID)),
            "name": GraphQLField(GraphQLString),
            "email": GraphQLField(GraphQLString),
        },
        interfaces=[node],
        is_type_of=lambda value, _ctx: value.get("kind") == "user",
    )

    def resolve_node(_root, _info, id):
        return {"kind": "user", "id": id, "name": "Ada", "email": "a@x"}

    query = GraphQLObjectType(
        "Query",
        {
            "node": GraphQLField(
                node,
                args={"id": GraphQLArgument(GraphQLNonNull(GraphQLID))},
                resolve=resolve_node,
            ),
            "unused": GraphQLField(GraphQLString, resolve=lambda *_: "x"),
        },
    )
    schema = GraphQLSchema(query=query, types=[user])
    document = parse(
        """
        query GetNode($id: ID!) {
          node(id: $id) { id ... on User { name } }
        }
        """
    )

    result = prune_schema(schema, document, {"id": "7"})
    pruned = result.schema
    assert not validate(pruned, document)
    execution = execute(pruned, document, variable_values={"id": "7"})
    assert execution.errors is None
    assert execution.data == {
        "node": {"id": "7", "name": "Ada"}
    }
    assert "unused" not in pruned.query_type.fields


def test_accepts_strings_and_multiple_documents():
    schema = build_schema(
        "type Query { a: String b: String }"
    )
    result = prune_schema(
        schema, ["{ a }", parse("{ b }")]
    )
    assert set(result.schema.query_type.fields) == {"a", "b"}


def test_rejects_invalid_operations():
    schema = build_schema("type Query { a: String }")
    with pytest.raises(ValueError, match="invalid operations"):
        prune_schema(schema, "{ nonExistent }")


def test_requires_at_least_one_operation():
    schema = build_schema("type Query { a: String }")
    with pytest.raises(ValueError, match="at least one operation"):
        prune_schema(schema, [])


def test_introspection_query_keeps_a_valid_root_type():
    schema = build_schema("type Query { a: String b: Int }")
    document = parse(get_introspection_query())

    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    assert not validate(result.schema, document)

    execution = execute(result.schema, document)
    assert execution.errors is None
    assert execution.data["__schema"]["queryType"]["name"] == "Query"
    # The root type cannot be empty, so its fields are retained and explained.
    assert set(result.schema.query_type.fields) == {"a", "b"}
    assert any("Query" in note for note in result.report.notes)


def test_directives_on_fragments_and_introspection_meta_fields():
    schema = build_schema(
        """
        directive @client on FIELD
        type Item { id: ID! secret: String }
        type Query { item: Item dropped: Boolean }
        """
    )
    document = parse(
        """
        query Q($flag: Boolean!) {
          item {
            __typename
            id @client @include(if: $flag)
            ...F
          }
        }
        fragment F on Item { id }
        """
    )
    result = prune_schema(schema, document)
    assert_valid_schema(result.schema)
    assert not validate(result.schema, document)
    assert set(result.schema.type_map["Item"].fields) == {"id"}
    directive_names = {d.name for d in result.schema.directives}
    assert {"client", "include", "skip"} <= directive_names


def test_variable_value_count_must_match_documents():
    schema = build_schema("type Query { a(x: String): String }")
    with pytest.raises(ValueError, match="number of variable value"):
        prune_schema(schema, ["{ a }", "{ a }"], [{}])
