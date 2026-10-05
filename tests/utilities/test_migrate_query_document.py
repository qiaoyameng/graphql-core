from graphql import (
    FieldCoordinate,
    build_schema,
    execute_sync,
    migrate_query_document,
    parse,
    print_ast,
    validate,
)


OLD_SCHEMA = build_schema(
    """
    interface Node {
      id: ID!
      name: String!
      fullName: String
      displayName: String
    }

    type User implements Node {
      id: ID!
      name: String!
      fullName: String
      displayName: String
    }

    type Org implements Node {
      id: ID!
      name: String!
      fullName: String
      displayName: String
    }

    union SearchResult = User | Org

    type Query {
      name: String
      node(id: ID!): Node
      user: User
      org: Org
      search(term: String!): [SearchResult!]!
    }
    """
)

NEW_SCHEMA = build_schema(
    """
    interface Node {
      id: ID!
      displayName: String!
      title: String
    }

    type User implements Node {
      id: ID!
      displayName: String!
      title: String
    }

    type Org implements Node {
      id: ID!
      displayName: String!
      title: String
      name: String
    }

    union SearchResult = User | Org

    type Query {
      name: String
      node(id: ID!): Node
      user: User
      org: Org
      search(term: String!): [SearchResult!]!
    }
    """
)


def test_migrates_shared_fragment_and_preserves_response_keys():
    document = parse(
        """
        query GetNode($id: ID!) {
          node(id: $id) {
            ...NodeFields @include(if: true)
          }
        }

        query Search($term: String!) {
          search(term: $term) {
            ...NodeFields
          }
        }

        fragment NodeFields on Node {
          id
          name
          fullName
        }
        """
    )

    result = migrate_query_document(
        OLD_SCHEMA,
        NEW_SCHEMA,
        document,
        {
            "Node.name": "Node.displayName",
            "Node.fullName": "Node.title",
        },
    )

    assert result.ok
    assert result.document is not None
    assert print_ast(document).count("name: displayName") == 0
    fragment = next(
        definition
        for definition in document.definitions
        if getattr(definition, "name", None)
        and definition.name.value == "NodeFields"
    )
    assert [
        field.name.value for field in fragment.selection_set.selections
    ] == ["id", "name", "fullName"]
    assert print_ast(result.document) == print_ast(
        parse(
            """
            query GetNode($id: ID!) {
              node(id: $id) {
                ...NodeFields @include(if: true)
              }
            }

            query Search($term: String!) {
              search(term: $term) {
                ...NodeFields
              }
            }

            fragment NodeFields on Node {
              id
              name: displayName
              fullName: title
            }
            """
        )
    )
    assert validate(NEW_SCHEMA, result.document) == []

    def old_resolver(source, info, **args):
        if info.parent_type.name == "Query":
            if info.field_name == "node":
                return {
                    "__typename": "User",
                    "id": args["id"],
                    "name": "Ada",
                    "fullName": "Dr Ada",
                }
            return [
                {
                    "__typename": "User",
                    "id": "1",
                    "name": "Ada",
                    "fullName": "Dr Ada",
                },
                {
                    "__typename": "Org",
                    "id": "2",
                    "name": "Acme",
                    "fullName": "Acme Inc",
                },
            ]
        return source.get(info.field_name)

    def new_resolver(source, info, **args):
        if info.parent_type.name == "Query":
            if info.field_name == "node":
                return {
                    "__typename": "User",
                    "id": args["id"],
                    "displayName": "Ada",
                    "title": "Dr Ada",
                }
            return [
                {
                    "__typename": "User",
                    "id": "1",
                    "displayName": "Ada",
                    "title": "Dr Ada",
                },
                {
                    "__typename": "Org",
                    "id": "2",
                    "displayName": "Acme",
                    "title": "Acme Inc",
                },
            ]
        return source.get(info.field_name)

    variables = {"id": "1", "term": "Ada"}
    old_node = execute_sync(
        OLD_SCHEMA,
        document,
        root_value=None,
        variable_values=variables,
        operation_name="GetNode",
        field_resolver=old_resolver,
    )
    new_node = execute_sync(
        NEW_SCHEMA,
        result.document,
        root_value=None,
        variable_values=variables,
        operation_name="GetNode",
        field_resolver=new_resolver,
    )
    new_search = execute_sync(
        NEW_SCHEMA,
        result.document,
        root_value=None,
        variable_values=variables,
        operation_name="Search",
        field_resolver=new_resolver,
    )

    assert old_node.errors is None
    assert new_node.errors is None
    assert new_search.errors is None
    assert new_node.data == old_node.data
    assert new_search.data == {
        "search": [
            {"id": "1", "name": "Ada", "fullName": "Dr Ada"},
            {"id": "2", "name": "Acme", "fullName": "Acme Inc"},
        ]
    }


def test_only_rewrites_matching_type_coordinate_and_keeps_existing_alias():
    document = parse(
        """
        query Viewer {
          name
          user {
            label: name @include(if: true)
          }
          org {
            name
          }
        }
        """
    )

    result = migrate_query_document(
        OLD_SCHEMA,
        NEW_SCHEMA,
        document,
        {(FieldCoordinate("User", "name")): FieldCoordinate("User", "displayName")},
    )

    assert result.ok
    assert result.document is not None
    assert print_ast(result.document) == print_ast(
        parse(
            """
                query Viewer {
                  name
                  user {
                    label: displayName @include(if: true)
                  }
              org {
                name
              }
            }
            """
        )
    )


def test_reports_conflict_with_existing_target_field_selection():
    conflicting_schema = build_schema(
        """
        type User {
          name: String
          displayName: String
        }

        type Query {
          user: User
        }
        """
    )
    document = parse(
        """
        {
          user {
            name
            displayName
          }
        }
        """
    )

    result = migrate_query_document(
        conflicting_schema,
        conflicting_schema,
        document,
        {"User.name": "User.displayName"},
    )

    assert result.document is None
    assert result.errors
    assert "displayName" in result.errors[0].message
    assert result.errors[0].locations


def test_reports_missing_target_field_coordinate():
    result = migrate_query_document(
        OLD_SCHEMA,
        NEW_SCHEMA,
        "{ user { name } }",
        {"User.name": "User.missing"},
    )

    assert result.document is None
    assert len(result.errors) == 1
    assert "User.missing" in result.errors[0].message


def test_reports_incompatible_abstract_implementations():
    incompatible_schema = build_schema(
        """
        interface Node {
          id: ID!
          name: String!
        }

        type User implements Node {
          id: ID!
          name: String!
        }

        type Org implements Node {
          id: ID!
          name: String!
        }

        type Query {
          node: Node
        }
        """
    )
    migrated_schema = build_schema(
        """
        interface Node {
          id: ID!
        }

        type User implements Node {
          id: ID!
          displayName: String!
        }

        type Org implements Node {
          id: ID!
          legalName: String!
        }

        type Query {
          node: Node
        }
        """
    )
    document = parse(
        """
        {
          node {
            ...NodeFields
          }
        }

        fragment NodeFields on Node {
          name
        }
        """
    )

    result = migrate_query_document(
        incompatible_schema,
        migrated_schema,
        document,
        {
            "User.name": "User.displayName",
            "Org.name": "Org.legalName",
        },
    )

    assert result.document is None
    assert len(result.errors) == 1
    assert "incompatible" in result.errors[0].message
    assert result.errors[0].locations


def test_handles_concrete_union_type_conditions_independently():
    incompatible_schema = build_schema(
        """
        union SearchResult = User | Org

        type User {
          name: String!
        }

        type Org {
          name: String!
        }

        type Query {
          search: [SearchResult!]!
        }
        """
    )
    migrated_schema = build_schema(
        """
        union SearchResult = User | Org

        type User {
          displayName: String!
        }

        type Org {
          legalName: String!
        }

        type Query {
          search: [SearchResult!]!
        }
        """
    )
    document = parse(
        """
        {
          search {
            ... on User {
              name
            }
            ... on Org {
              name
            }
          }
        }
        """
    )

    result = migrate_query_document(
        incompatible_schema,
        migrated_schema,
        document,
        {
            "User.name": "User.displayName",
            "Org.name": "Org.legalName",
        },
    )

    assert result.ok
    assert print_ast(result.document) == print_ast(
        parse(
            """
            {
              search {
                ... on User {
                  name: displayName
                }
                ... on Org {
                  name: legalName
                }
              }
            }
            """
        )
    )
