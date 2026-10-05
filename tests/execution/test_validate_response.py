"""Tests for validating GraphQL responses against a request and a schema."""

from __future__ import annotations

from graphql import (
    GraphQLError,
    build_schema,
    parse,
    validate_response,
)
from graphql.execution import (
    ResponseIssueKind,
    ResponseValidationResult,
)

from ..utils import dedent

schema = build_schema(
    dedent(
        """
        interface Character {
          id: ID!
          name: String!
        }

        type Human implements Character {
          id: ID!
          name: String!
          height: Int
        }

        type Droid implements Character {
          id: ID!
          name: String!
          primaryFunction: String
        }

        union Pet = Cat | Dog

        type Cat {
          name: String!
        }

        type Dog {
          name: String!
        }

        enum Episode {
          NEWHOPE
          EMPIRE
          JEDI
        }

        scalar Date

        type User {
          id: ID!
          name: String!
          friends: [User]
        }

        type Query {
          hero: Character
          pet: Pet
          user: User
          users: [User!]
          strictUsers: [User!]!
          tags: [String!]!
          episode: Episode
          date: Date
          skipped: String
          included: String
        }
        """
    )
)


def validate(query, response, variables=None, scalar_checkers=None):
    document = parse(dedent(query))
    return validate_response(
        schema,
        document,
        response,
        variable_values=variables,
        scalar_checkers=scalar_checkers,
    )


def kinds(result: ResponseValidationResult):
    return [(issue.kind, issue.path) for issue in result.issues]


def describe_validate_response():
    def rejects_unknown_operation_name():
        document = parse("query One { tags } query Two { tags }")
        try:
            validate_response(schema, document, {"data": {"tags": []}}, "Three")
        except GraphQLError as error:
            assert error.message == "Unknown operation named 'Three'."
        else:  # pragma: no cover
            raise AssertionError("Expected GraphQLError")

    def reports_invalid_variable_values():
        document = parse("query ($skip: Boolean!) { tags }")
        result = validate_response(
            schema, document, {"data": {"tags": []}}, variable_values={"skip": "x"}
        )
        assert not result.is_valid
        assert [issue.kind for issue in result.issues] == [
            ResponseIssueKind.INVALID_VARIABLES
        ]

    def describe_response_envelope():
        def rejects_non_object_response():
            result = validate("{ tags }", ["not", "an", "object"])
            assert [issue.kind for issue in result.issues] == [
                ResponseIssueKind.INVALID_RESPONSE
            ]

        def reports_missing_data_entry():
            result = validate("{ tags }", {})
            assert kinds(result) == [(ResponseIssueKind.MISSING_DATA, None)]

        def allows_missing_data_entry_with_request_errors():
            result = validate("{ tags }", {"errors": [{"message": "Bad request"}]})
            assert result.is_valid

        def rejects_non_object_data():
            result = validate("{ tags }", {"data": "nope"})
            assert kinds(result) == [
                (ResponseIssueKind.INVALID_RESPONSE, ())
            ]

        def reports_null_data_without_errors():
            result = validate("{ tags }", {"data": None})
            assert kinds(result) == [(ResponseIssueKind.UNEXPECTED_NULL, ())]

    def describe_selected_response_keys():
        def accepts_a_valid_response():
            result = validate(
                "{ user { id name } tags }",
                {"data": {"user": {"id": "1", "name": "Ann"}, "tags": ["x"]}},
            )
            assert result.is_valid
            assert result.issues == []

        def reports_missing_response_keys():
            result = validate(
                "{ user { id name } tags }",
                {"data": {"user": {"id": "1"}, "tags": []}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.MISSING_FIELD, ("user", "name"))
            ]

        def reports_unexpected_response_keys():
            result = validate(
                "{ user { id } }",
                {"data": {"user": {"id": "1", "email": "a@b.c"}}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_FIELD, ("user", "email"))
            ]

        def distinguishes_aliases_from_field_names():
            result = validate(
                "{ fullName: user { id } }",
                {"data": {"user": {"id": "1"}}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.MISSING_FIELD, ("fullName",)),
                (ResponseIssueKind.UNEXPECTED_FIELD, ("user",)),
            ]

        def evaluates_skip_and_include_with_variables():
            query = """
                query ($skip: Boolean!, $include: Boolean!) {
                  skipped @skip(if: $skip)
                  included @include(if: $include)
                }
            """
            result = validate(
                query, {"data": {}}, variables={"skip": True, "include": False}
            )
            assert result.is_valid

            result = validate(
                query,
                {"data": {"skipped": "x", "included": "y"}},
                variables={"skip": True, "include": False},
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_FIELD, ("skipped",)),
                (ResponseIssueKind.UNEXPECTED_FIELD, ("included",)),
            ]

            result = validate(
                query, {"data": {}}, variables={"skip": False, "include": True}
            )
            assert kinds(result) == [
                (ResponseIssueKind.MISSING_FIELD, ("skipped",)),
                (ResponseIssueKind.MISSING_FIELD, ("included",)),
            ]

        def collects_fields_from_fragments():
            query = """
                { user { ...userFields } }
                fragment userFields on User { id name }
            """
            result = validate(query, {"data": {"user": {"id": "1", "name": "Ann"}}})
            assert result.is_valid

            result = validate(query, {"data": {"user": {"id": "1"}}})
            assert kinds(result) == [
                (ResponseIssueKind.MISSING_FIELD, ("user", "name"))
            ]

    def describe_null_and_non_null():
        def allows_explicit_null_at_nullable_positions():
            result = validate("{ user }", {"data": {"user": None}})
            assert result.is_valid

        def reports_explicit_null_at_non_null_positions():
            result = validate(
                "{ user { id name } }",
                {"data": {"user": {"id": None, "name": "Ann"}}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_NULL, ("user", "id"))
            ]

        def reports_null_list_items_at_non_null_positions():
            result = validate(
                "{ tags }", {"data": {"tags": ["a", None, "b"]}}
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_NULL, ("tags", 1))
            ]

    def describe_list_levels():
        def reports_a_missing_list_level():
            result = validate("{ tags }", {"data": {"tags": "not-a-list"}})
            assert kinds(result) == [(ResponseIssueKind.EXPECTED_LIST, ("tags",))]

        def reports_an_unexpected_list_level():
            result = validate(
                "{ user { id } }", {"data": {"user": [{"id": "1"}]}}
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_LIST, ("user",))
            ]

        def reports_a_missing_object_level():
            result = validate("{ user { id } }", {"data": {"user": "Ann"}})
            assert kinds(result) == [
                (ResponseIssueKind.EXPECTED_OBJECT, ("user",))
            ]

        def validates_nested_lists():
            result = validate(
                "{ users { id } }",
                {"data": {"users": [{"id": "1"}, {"id": None}]}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNEXPECTED_NULL, ("users", 1, "id"))
            ]

    def describe_leaf_values():
        def validates_specified_scalar_output_values():
            query = "{ user { id name } episode }"
            good = {"data": {"user": {"id": "1", "name": "Ann"}, "episode": "JEDI"}}
            assert validate(query, good).is_valid

            bad = {
                "data": {"user": {"id": True, "name": 42}, "episode": "SITH"}
            }
            result = validate(query, bad)
            assert kinds(result) == [
                (ResponseIssueKind.INVALID_SCALAR, ("user", "id")),
                (ResponseIssueKind.INVALID_SCALAR, ("user", "name")),
                (ResponseIssueKind.INVALID_ENUM, ("episode",)),
            ]

        def checks_custom_scalars_with_a_caller_provided_function():
            def is_date(value):
                return (
                    isinstance(value, str)
                    and len(value) == 10
                    and value[4] == "-"
                    and value[7] == "-"
                )

            result = validate(
                "{ date }",
                {"data": {"date": "not-a-date"}},
                scalar_checkers={"Date": is_date},
            )
            assert kinds(result) == [
                (ResponseIssueKind.INVALID_SCALAR, ("date",))
            ]

            result = validate(
                "{ date }",
                {"data": {"date": "2026-10-05"}},
                scalar_checkers={"Date": is_date},
            )
            assert result.is_valid

        def accepts_custom_scalars_without_a_checker():
            result = validate("{ date }", {"data": {"date": "anything"}})
            assert result.is_valid

    def describe_abstract_types():
        def uses_typename_to_determine_the_runtime_type():
            query = "{ hero { id name ... on Human { height } } }"
            result = validate(
                query,
                {
                    "data": {
                        "hero": {
                            "__typename": "Human",
                            "id": "1",
                            "name": "Ann",
                            "height": 172,
                        }
                    }
                },
            )
            assert result.is_valid

        def reports_an_invalid_typename():
            query = "{ hero { id name } }"
            result = validate(
                query,
                {"data": {"hero": {"__typename": "Cat", "id": "1", "name": "Tom"}}},
            )
            assert kinds(result) == [
                (ResponseIssueKind.INVALID_TYPENAME, ("hero", "__typename"))
            ]

        def reports_undetermined_branches_without_typename():
            query = "{ hero { id name ... on Human { height } } }"
            result = validate(
                query, {"data": {"hero": {"id": "1", "name": "Ann"}}}
            )
            assert result.errors == []
            assert not result.is_valid or result.undetermined
            assert kinds(result) == [
                (ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE, ("hero",))
            ]
            assert "Human" in result.issues[0].message
            assert "Droid" in result.issues[0].message

        def still_checks_common_keys_without_typename():
            query = "{ hero { id name ... on Human { height } } }"
            result = validate(query, {"data": {"hero": {"name": "Ann"}}})
            assert kinds(result) == [
                (ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE, ("hero",)),
                (ResponseIssueKind.MISSING_FIELD, ("hero", "id")),
            ]

        def does_not_require_branch_specific_keys_without_typename():
            query = "{ hero { id name ... on Human { height } } }"
            result = validate(
                query, {"data": {"hero": {"id": "1", "name": "Ann"}}}
            )
            assert ResponseIssueKind.MISSING_FIELD not in [
                issue.kind for issue in result.issues
            ]

        def reports_unexpected_keys_without_typename():
            query = "{ pet { ... on Cat { name } } }"
            result = validate(
                query, {"data": {"pet": {"name": "Tom", "barks": True}}}
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE, ("pet",)),
                (ResponseIssueKind.UNEXPECTED_FIELD, ("pet", "barks")),
            ]

    def describe_response_errors():
        def accepts_a_valid_partial_failure_response():
            result = validate(
                "{ user { id } tags }",
                {
                    "data": {"user": None, "tags": []},
                    "errors": [{"message": "User failed", "path": ["user"]}],
                },
            )
            assert result.is_valid

        def accepts_null_propagation_to_a_nullable_parent():
            result = validate(
                "{ user { id name } }",
                {
                    "data": {"user": None},
                    "errors": [{"message": "id failed", "path": ["user", "id"]}],
                },
            )
            assert result.is_valid

        def accepts_null_propagation_to_the_root():
            result = validate(
                "{ strictUsers { id } }",
                {
                    "data": None,
                    "errors": [
                        {"message": "item failed", "path": ["strictUsers", 0, "id"]}
                    ],
                },
            )
            assert result.is_valid

        def accepts_null_propagation_to_a_nullable_list():
            result = validate(
                "{ users { id } }",
                {
                    "data": {"users": None},
                    "errors": [{"message": "id failed", "path": ["users", 1, "id"]}],
                },
            )
            assert result.is_valid

        def reports_null_that_was_not_propagated():
            result = validate(
                "{ user { id name } }",
                {
                    "data": {"user": {"id": None, "name": "Ann"}},
                    "errors": [{"message": "id failed", "path": ["user", "id"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.EXPECTED_NULL, ("user",)),
                (ResponseIssueKind.UNEXPECTED_NULL, ("user", "id")),
            ]

        def reports_error_paths_with_a_wrong_alias():
            result = validate(
                "{ fullName: user { id } }",
                {
                    "data": {"fullName": {"id": "1"}},
                    "errors": [{"message": "x", "path": ["user", "id"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNMATCHED_ERROR_PATH, ("user", "id"))
            ]

        def reports_error_paths_with_a_wrong_list_index():
            result = validate(
                "{ users { id } }",
                {
                    "data": {"users": [{"id": "1"}]},
                    "errors": [{"message": "x", "path": ["users", 5, "id"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNMATCHED_ERROR_PATH, ("users", 5, "id"))
            ]

        def reports_error_paths_with_a_string_list_index():
            result = validate(
                "{ users { id } }",
                {
                    "data": {"users": [{"id": "1"}]},
                    "errors": [{"message": "x", "path": ["users", "0", "id"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNMATCHED_ERROR_PATH, ("users", "0", "id"))
            ]

        def reports_unrelated_error_paths():
            result = validate(
                "{ tags }",
                {
                    "data": {"tags": []},
                    "errors": [{"message": "x", "path": ["something", "else"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNMATCHED_ERROR_PATH, ("something", "else"))
            ]

        def reports_malformed_error_paths():
            result = validate(
                "{ tags }",
                {"data": {"tags": []}, "errors": [{"message": "x", "path": "tags"}]},
            )
            assert kinds(result) == [(ResponseIssueKind.UNMATCHED_ERROR_PATH, None)]

        def one_error_does_not_excuse_the_whole_data():
            result = validate(
                "{ users { id } strictUsers { id } }",
                {
                    "data": {"users": None},
                    "errors": [{"message": "x", "path": ["users", 0, "id"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.MISSING_FIELD, ("strictUsers",))
            ]

        def an_unmatched_error_excuses_no_null_position():
            result = validate(
                "{ user { id } }",
                {
                    "data": {"user": {"id": None}},
                    "errors": [{"message": "x", "path": ["unrelated"]}],
                },
            )
            assert kinds(result) == [
                (ResponseIssueKind.UNMATCHED_ERROR_PATH, ("unrelated",)),
                (ResponseIssueKind.UNEXPECTED_NULL, ("user", "id")),
            ]

        def reports_data_that_should_be_null_because_of_an_error():
            result = validate(
                "{ user { id } }",
                {
                    "data": {"user": {"id": "1"}},
                    "errors": [{"message": "x", "path": ["user"]}],
                },
            )
            assert kinds(result) == [(ResponseIssueKind.EXPECTED_NULL, ("user",))]
