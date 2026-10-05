from __future__ import annotations

from graphql import (
    ExecutionResult,
    GraphQLError,
    build_schema,
    get_introspection_query,
    graphql_sync,
    parse,
    validate_response,
)
from graphql.response_validation import ResponseValidationIssueKind


SCHEMA = build_schema(
    """
    type Query {
      me: User
      node(id: ID!): Node
      search(term: String!): [Result!]!
      tags: [String!]
      matrix: [[Int!]!]
      status: Status!
      when: DateTime
      deep: DeepA
    }

    type DeepA {
      b: DeepB
    }

    type DeepB {
      c: DeepC
      cn: DeepC!
    }

    type DeepC {
      value: String!
    }

    interface Node {
      id: ID!
    }

    type User implements Node {
      id: ID!
      name: String!
      nickname: String
      role: Role
    }

    type Bot implements Node {
      id: ID!
      code: Int!
    }

    union Result = User | Bot

    enum Role {
      ADMIN
      MEMBER
    }

    enum Status {
      ACTIVE
      INACTIVE
    }

    scalar DateTime
    """
)


def kinds(report) -> list[tuple[str, tuple | None]]:
    return [(issue.kind.value, issue.path) for issue in report.issues]


def describe_happy_paths():
    def validates_a_complete_response():
        query = """
          {
            me { id name nickname role }
            tags
            matrix
            status
          }
        """
        response = {
            "data": {
                "me": {
                    "id": "1",
                    "name": "Ada",
                    "nickname": None,
                    "role": "ADMIN",
                },
                "tags": ["a", "b"],
                "matrix": [[1, 2], [3]],
                "status": "ACTIVE",
            }
        }
        report = validate_response(SCHEMA, query, response)
        assert report.valid
        assert kinds(report) == []

    def accepts_parsed_documents_and_graphqlerrors():
        response = {
            "data": {"me": None},
            "errors": [GraphQLError("down", path=["me"])],
        }
        report = validate_response(SCHEMA, parse("{ me { id name } }"), response)
        assert report.valid

    def accepts_execution_results_and_introspection_payloads():
        result = ExecutionResult(
            data={"me": {"id": "1", "name": "Ada"}}, errors=None
        )
        report = validate_response(SCHEMA, "{ me { id name } }", result)
        assert report.valid

        introspection = graphql_sync(SCHEMA, get_introspection_query())
        assert introspection.errors is None
        report = validate_response(
            SCHEMA, get_introspection_query(), introspection.formatted
        )
        assert report.valid

    def honors_aliases():
        response = {"data": {"who": {"id": "1", "name": "Ada"}}}
        report = validate_response(
            SCHEMA, "{ who: me { id name } }", response
        )
        assert report.valid

    def honors_skip_and_include_directives():
        query = """
          query ($skipName: Boolean!, $includeNick: Boolean!) {
            me {
              id
              name @skip(if: $skipName)
              nickname @include(if: $includeNick)
            }
          }
        """
        response = {"data": {"me": {"id": "1", "nickname": None}}}
        report = validate_response(
            SCHEMA, query, response, {"skipName": True, "includeNick": True}
        )
        assert report.valid

        skipped_present = {"data": {"me": {"id": "1", "name": "Ada"}}}
        report = validate_response(
            SCHEMA,
            query,
            skipped_present,
            {"skipName": True, "includeNick": False},
        )
        assert kinds(report) == [
            ("unexpected-key", ("me", "name")),
        ]


def describe_keys_and_nullability():
    def distinguishes_missing_explicit_null_and_violations():
        response = {
            "data": {
                "me": {"id": "1", "nickname": None},
                "tags": None,
            }
        }
        report = validate_response(SCHEMA, "{ me { id name nickname } tags }", response)
        assert kinds(report) == [
            ("missing-key", ("me", "name")),
        ]

    def flags_explicit_non_null_violation_without_error():
        response = {"data": {"me": {"id": None, "name": "Ada"}}}
        report = validate_response(SCHEMA, "{ me { id name } }", response)
        assert kinds(report) == [
            ("non-null-violation", ("me", "id")),
        ]

    def allows_non_null_null_when_an_error_propagates():
        response = {
            "data": {"me": None},
            "errors": [{"message": "boom", "path": ["me", "id"]}],
        }
        report = validate_response(SCHEMA, "{ me { id name } }", response)
        assert report.valid
        assert ("me",) in report.nulled_positions
        assert ("me", "id") in report.nulled_positions

    def flags_null_root_without_propagating_error():
        response = {"data": None, "errors": [{"message": "auth"}]}
        report = validate_response(SCHEMA, "{ me { id } }", response)
        assert kinds(report) == [("non-null-violation", ())]

    def flags_unexpected_keys():
        response = {"data": {"me": {"id": "1", "name": "Ada", "age": 3}}}
        report = validate_response(SCHEMA, "{ me { id name } }", response)
        assert kinds(report) == [("unexpected-key", ("me", "age"))]


def describe_lists():
    def reports_list_level_errors():
        response = {"data": {"matrix": [[1], 2]}}
        report = validate_response(SCHEMA, "{ matrix }", response)
        assert kinds(report) == [
            ("list-level-error", ("matrix", 1)),
        ]

    def reports_scalar_where_list_expected():
        response = {"data": {"tags": "nope"}}
        report = validate_response(SCHEMA, "{ tags }", response)
        assert kinds(report) == [("list-level-error", ("tags",))]

    def non_null_list_items_need_a_justifying_error():
        response = {"data": {"tags": ["a", None]}}
        report = validate_response(SCHEMA, "{ tags }", response)
        assert kinds(report) == [
            ("non-null-violation", ("tags", 1)),
        ]

    def non_null_list_items_may_be_null_after_item_error():
        response = {
            "data": {"tags": None},
            "errors": [{"message": "x", "path": ["tags", 1]}],
        }
        report = validate_response(SCHEMA, "{ tags }", response)
        assert report.valid
        assert ("tags",) in report.nulled_positions


def describe_abstract_types():
    def uses_typename_to_validate_a_concrete_branch():
        query = """
          {
            node(id: "1") {
              __typename
              id
              ... on User { name role }
              ... on Bot { code }
            }
          }
        """
        response = {
            "data": {
                "node": {
                    "__typename": "Bot",
                    "id": "1",
                    "code": 7,
                }
            }
        }
        report = validate_response(SCHEMA, query, response)
        assert report.valid

    def rejects_an_unknown_typename():
        query = '{ node(id: "1") { __typename id } }'
        response = {"data": {"node": {"__typename": "Ghost", "id": "1"}}}
        report = validate_response(SCHEMA, query, response)
        assert kinds(report) == [
            ("invalid-typename", ("node", "__typename")),
        ]

    def reports_undecidable_branches_without_type_information():
        query = """
          {
            node(id: "1") {
              id
              ... on User { name }
              ... on Bot { code }
            }
          }
        """
        response = {"data": {"node": {"id": "1", "code": 7}}}
        report = validate_response(SCHEMA, query, response)
        reported = {issue.path for issue in report.issues}
        assert report.issues[0].kind == (
            ResponseValidationIssueKind.UNDECIDABLE_BRANCH
        )
        assert ("node", "code") in reported
        assert ("node",) in reported

    def does_not_pass_an_undecidable_branch_by_picking_an_implementation():
        # The response only carries Bot's code. Without __typename the
        # checker must not simply pick Bot and pass: the branch is reported
        # as undecidable instead of accepting the Bot implementation.
        query = """
          {
            node(id: "1") {
              ... on User { name nickname }
              ... on Bot { code }
            }
          }
        """
        response = {"data": {"node": {"code": 7}}}
        report = validate_response(SCHEMA, query, response)
        undecidable = {
            issue.path
            for issue in report.issues
            if issue.kind == ResponseValidationIssueKind.UNDECIDABLE_BRANCH
        }
        assert ("node", "code") in undecidable
        assert ("node",) in undecidable
        # The report must not contain a plain success nor treat code as
        # unexpected.
        assert report.issues

    def validates_common_fields_of_unions():
        query = '{ search(term: "x") { ... on Node { id } } }'
        response = {"data": {"search": [{"id": "1"}, {"id": "2"}]}}
        report = validate_response(SCHEMA, query, response)
        assert report.valid

    def typename_disambiguates_union_members():
        query = """
          {
            search(term: "x") {
              __typename
              ... on User { name }
              ... on Bot { code }
            }
          }
        """
        response = {
            "data": {
                "search": [
                    {"__typename": "User", "name": "Ada"},
                    {"__typename": "Bot", "code": 9},
                ]
            }
        }
        report = validate_response(SCHEMA, query, response)
        assert report.valid


def describe_scalars_and_enums():
    def validates_builtin_scalars_and_enums():
        good = {"data": {"me": {"id": 1, "name": "x"}, "status": "ACTIVE"}}
        assert validate_response(
            SCHEMA, "{ me { id name } status }", good
        ).valid

        bad_enum = {"data": {"status": "WAITING"}}
        report = validate_response(SCHEMA, "{ status }", bad_enum)
        assert kinds(report) == [("invalid-scalar", ("status",))]

        report = validate_response(
            SCHEMA,
            '{ node(id: "1") { __typename ... on Bot { code } } }',
            {"data": {"node": {"__typename": "Bot", "code": True}}},
        )
        assert kinds(report) == [("invalid-scalar", ("node", "code"))]

    def custom_scalar_checkers_decide_custom_leaf_values():
        def checker(_scalar, value):
            if not isinstance(value, str) or not value.endswith("Z"):
                return "must be an ISO timestamp ending in Z"
            return None

        query = "{ when }"
        good = {"data": {"when": "2026-10-05T00:00:00Z"}}
        report = validate_response(
            SCHEMA, query, good, scalar_checkers={"DateTime": checker}
        )
        assert report.valid

        bad = {"data": {"when": "yesterday"}}
        report = validate_response(
            SCHEMA, query, bad, scalar_checkers={"DateTime": checker}
        )
        assert kinds(report) == [("invalid-scalar", ("when",))]

    def custom_scalars_without_checker_skip_leaf_checks():
        report = validate_response(
            SCHEMA, "{ when }", {"data": {"when": {"weird": True}}}
        )
        assert report.valid


def describe_error_paths():
    QUERY = """
      {
        me { id name nickname }
        tagged: tags
      }
    """

    def accepts_a_legal_partial_failure_recording():
        # nickname is nullable: its field error only nulls nickname.
        # tagged/tags items are non-null: an item error nulls the whole list.
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": None},
                "tagged": None,
            },
            "errors": [
                {"message": "nickname failed", "path": ["me", "nickname"]},
                {"message": "item failed", "path": ["tagged", 2]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert report.valid
        assert report.nulled_positions == {
            ("me", "nickname"),
            ("tagged", 2),
            ("tagged",),
        }

    def rejects_wrong_alias_in_error_path_but_keeps_other_findings():
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": None},
                "tagged": ["a"],
            },
            "errors": [
                {"message": "old alias", "path": ["tags", 0]},
                {"message": "nickname failed", "path": ["me", "nickname"]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("invalid-error-path", ("tags",)),
        ]

    def rejects_out_of_range_list_index_in_error_path():
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": "Ada"},
                "tagged": ["a"],
            },
            "errors": [{"message": "idx", "path": ["tagged", 9]}],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("error-path-data-mismatch", ("tagged", 9)),
        ]

    def rejects_unrelated_error_paths_item_by_item():
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": "Ada"},
                "tagged": [],
            },
            "errors": [
                {"message": "no such root"},
                {"message": "unrelated", "path": ["other", "field"]},
                {"message": "deep", "path": ["me", "id", "x"]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("invalid-error-path", ("other",)),
            ("error-path-data-mismatch", ("me", "id", "x")),
        ]

    def one_error_does_not_excuse_the_whole_data():
        # A valid nickname error must not hide a missing non-null name and an
        # unrelated bad value elsewhere.
        response = {
            "data": {
                "me": {"id": "1", "nickname": None},
                "tagged": "not-a-list",
            },
            "errors": [
                {"message": "nickname failed", "path": ["me", "nickname"]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("missing-key", ("me", "name")),
            ("list-level-error", ("tagged",)),
        ]

    def nullable_error_terminal_must_be_null():
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": "still here"},
                "tagged": [],
            },
            "errors": [
                {"message": "nickname failed", "path": ["me", "nickname"]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("error-without-null", ("me", "nickname")),
        ]

    def non_null_propagation_reaches_the_nearest_nullable_ancestor():
        # deep.b.cn.value is non-null, b is nullable, cn is a non-null
        # object. An error at value propagates through cn and lands on b.
        response = {
            "data": {"deep": {"b": None}},
            "errors": [
                {"message": "boom", "path": ["deep", "b", "cn", "value"]},
            ],
        }
        report = validate_response(
            SCHEMA,
            "{ deep { b { cn { value } } } }",
            response,
        )
        assert report.valid
        assert ("deep", "b") in report.nulled_positions

    def nullable_ancestor_keeps_a_non_null_child():
        # With c nullable, the value error is caught at c; b must remain a
        # present object, so b: null cannot be explained by this error.
        response = {
            "data": {"deep": {"b": {"c": None}}},
            "errors": [
                {"message": "boom", "path": ["deep", "b", "c", "value"]},
            ],
        }
        report = validate_response(
            SCHEMA,
            "{ deep { b { c { value } } } }",
            response,
        )
        assert report.valid

    def nullable_ancestor_cannot_be_nulled_from_a_caught_error():
        response = {
            "data": {"deep": {"b": None}},
            "errors": [
                {"message": "boom", "path": ["deep", "b", "c", "value"]},
            ],
        }
        report = validate_response(
            SCHEMA,
            "{ deep { b { c { value } } } }",
            response,
        )
        assert kinds(report) == [
            ("error-path-data-mismatch", ("deep", "b")),
        ]

    def item_error_propagating_to_a_nullable_root_list():
        response = {
            "data": {"tags": None},
            "errors": [{"message": "boom", "path": ["tags", 0]}],
        }
        report = validate_response(SCHEMA, "{ tags }", response)
        assert report.valid
        assert ("tags",) in report.nulled_positions

    def error_paths_inside_abstract_types_use_typename():
        query = """
          {
            node(id: "1") {
              __typename
              ... on User { nickname }
              ... on Bot { code }
            }
          }
        """
        good = {
            "data": {
                "node": {"__typename": "User", "nickname": None},
            },
            "errors": [
                {"message": "x", "path": ["node", "nickname"]},
            ],
        }
        assert validate_response(SCHEMA, query, good).valid

        bad_branch = {
            "data": {"node": {"__typename": "User"}},
            "errors": [{"message": "x", "path": ["node", "code"]}],
        }
        report = validate_response(SCHEMA, query, bad_branch)
        assert ("invalid-error-path", ("node", "code")) in kinds(report)

    def recorded_partial_failure_corpus_is_itemized():
        # A realistic recording: legitimate partial failures mixed with a
        # stale alias, an out-of-range list index and an unrelated path.
        query = """
          query ($withAvatar: Boolean!) {
            viewer: me {
              id
              nickname
            }
            tagged: tags
          }
        """
        response = {
            "data": {
                "viewer": {"id": "1", "nickname": None},
                "tagged": ["a", None],
            },
            "errors": [
                {"message": "ok", "path": ["viewer", "nickname"]},
                {"message": "ok", "path": ["tagged", 1]},
                {"message": "stale alias", "path": ["me", "id"]},
                {"message": "bad index", "path": ["tagged", 7]},
                {"message": "global"},
            ],
        }
        report = validate_response(
            SCHEMA,
            query,
            response,
            {"withAvatar": True},
        )
        assert sorted(kinds(report)) == sorted([
            ("invalid-error-path", ("me",)),
            ("error-path-data-mismatch", ("tagged", 7)),
        ])
        # The legitimate errors still explain the nulls in data.
        assert ("viewer", "nickname") in report.nulled_positions
        assert ("tagged",) in report.nulled_positions
        # The global error and the bad paths never excuse the rest of data:
        # every selected, healthy key was still validated.
        assert report.valid is False

    def malformed_error_paths_are_reported_at_their_index():
        response = {
            "data": {
                "me": {"id": "1", "name": "Ada", "nickname": None},
                "tagged": [],
            },
            "errors": [
                {"message": "ok"},
                {"message": "bad", "path": ["me", True]},
            ],
        }
        report = validate_response(SCHEMA, QUERY, response)
        assert kinds(report) == [
            ("malformed-response", ("errors", 1)),
        ]


def describe_envelope():
    def rejects_responses_without_data_or_errors():
        report = validate_response(SCHEMA, "{ me { id } }", {"nope": True})
        assert kinds(report) == [("malformed-response", None)]

    def rejects_non_object_data():
        report = validate_response(SCHEMA, "{ me { id } }", {"data": [1]})
        assert kinds(report) == [("malformed-response", ())]

    def rejects_non_list_errors():
        report = validate_response(
            SCHEMA,
            "{ me { id } }",
            {"data": {"me": {"id": "1"}}, "errors": {}},
        )
        assert kinds(report) == [("malformed-response", None)]
