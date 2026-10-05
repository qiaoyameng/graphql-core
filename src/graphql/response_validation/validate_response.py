"""Validate recorded GraphQL responses against the schema and the request.

Gateway replay does not have access to the upstream resolvers. Given only the
schema, the selected operation, the variable values and a response JSON, this
module checks whether the response fulfills the GraphQL contract:

- every selected response key (respecting aliases and ``@skip``/``@include``)
  is present, with missing keys, explicit nulls, list nesting errors and
  non-null violations reported separately;
- values at interface and union positions are validated without a
  ``resolve_type`` function: if the response does not reveal the runtime type
  (for example via ``__typename``), the branches that cannot be decided are
  reported instead of accepting an arbitrary implementation;
- leaf values of custom scalars are checked with functions provided by the
  caller, while built-in scalars and enums get structural checks;
- ``errors[].path`` entries are resolved against the request, and the null
  positions permitted by non-null error propagation are computed. Errors never
  excuse the remainder of ``data``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

from ..error import GraphQLError
from ..language import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    OperationDefinitionNode,
    parse,
)
from ..pyutils import inspect, print_path_list
from ..type import (
    GraphQLBoolean,
    GraphQLDeferDirective,
    GraphQLFloat,
    GraphQLID,
    GraphQLIncludeDirective,
    GraphQLInt,
    GraphQLObjectType,
    GraphQLSkipDirective,
    GraphQLString,
    is_abstract_type,
    is_enum_type,
    is_list_type,
    is_non_null_type,
    is_scalar_type,
)
from ..type.introspection import (
    SchemaMetaFieldDef,
    TypeMetaFieldDef,
    TypeNameMetaFieldDef,
)
from ..utilities import build_schema, get_operation_ast
from ..execution.values import (
    VariableValues,
    get_directive_values,
    get_variable_values,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..type import (
        GraphQLAbstractType,
        GraphQLField,
        GraphQLNamedType,
        GraphQLOutputType,
        GraphQLScalarType,
        GraphQLSchema,
    )
__all__ = [
    "ResponseValidationIssue",
    "ResponseValidationIssueKind",
    "ResponseValidationReport",
    "ScalarOutputChecker",
    "validate_response",
]


class ResponseValidationIssueKind(str, Enum):
    """Kinds of issues reported by response validation."""

    MALFORMED_RESPONSE = "malformed-response"
    """The response envelope is not a valid GraphQL result (wrong ``data``/
    ``errors`` shape)."""

    MISSING_KEY = "missing-key"
    """A key selected by the operation is absent from a response object."""

    UNEXPECTED_KEY = "unexpected-key"
    """The response contains a key not selected by the operation."""

    NON_NULL_VIOLATION = "non-null-violation"
    """A non-null position is null without a justifying propagated error."""

    LIST_LEVEL_ERROR = "list-level-error"
    """A list position does not have the list nesting required by the type."""

    INVALID_SCALAR = "invalid-scalar"
    """A leaf scalar or enum value is not a valid output value of the type."""

    INVALID_TYPENAME = "invalid-typename"
    """A ``__typename`` value does not name a possible type of the position."""

    UNDECIDABLE_BRANCH = "undecidable-branch"
    """An abstract position lacks the runtime type information needed to decide
    whether one of its type-conditioned branches is valid."""

    INVALID_ERROR_PATH = "invalid-error-path"
    """An ``errors[].path`` cannot correspond to a selected request position."""

    ERROR_PATH_DATA_MISMATCH = "error-path-data-mismatch"
    """A valid ``errors[].path`` is inconsistent with the ``data`` payload."""

    ERROR_WITHOUT_NULL = "error-without-null"
    """A field error lands on a nullable position whose value is not null."""


class ResponseValidationIssue(NamedTuple):
    """A single response validation finding.

    :param kind: the kind of the issue
    :param path: the response path of the finding, or ``None`` for findings
        that are not tied to a response position (such as an ``errors`` entry
        without a ``path``)
    :param message: a human readable description of the finding
    """

    kind: ResponseValidationIssueKind
    """Kind of the finding, from :class:`ResponseValidationIssueKind`."""
    path: tuple[str | int, ...] | None
    """Response path of the finding, or ``None`` when position-less."""
    message: str
    """Human readable description of the finding."""

    def __str__(self) -> str:
        location = print_path_list(list(self.path)) if self.path else ""
        return f"{self.message}{location}"


class ResponseValidationReport:
    """The result of validating a recorded response.

    :param issues: all findings, in the order they were detected
    :param nulled_positions: response positions that are null (or missing)
        because of non-null error propagation

    >>> report = ResponseValidationReport()
    >>> report.valid
    True
    """

    def __init__(
        self,
        issues: list[ResponseValidationIssue] | None = None,
        nulled_positions: set[tuple[str | int, ...]] | None = None,
    ) -> None:
        self.issues: list[ResponseValidationIssue] = issues or []
        self.nulled_positions: set[tuple[str | int, ...]] = nulled_positions or set()

    @property
    def valid(self) -> bool:
        """Whether the response satisfies the GraphQL contract."""
        return not self.issues

    def __bool__(self) -> bool:
        return self.valid

    def __iter__(self):
        return iter(self.issues)

    def __len__(self) -> int:
        return len(self.issues)


#: Signature of a custom scalar output checker. The checker receives the
#: scalar type and the JSON value, returning ``None`` when acceptable or a
#: human readable reason otherwise.
ScalarOutputChecker = Callable[..., str | None]


class FieldEntry(NamedTuple):
    """A selected field together with the type conditions guarding it.

    :param node: the field AST node
    :param conditions: type-condition names of the enclosing fragments, in
        nesting order; an empty tuple means the field is selected unconditionally
    """

    node: FieldNode
    conditions: tuple[str, ...]


GroupedFields = dict[str, list[FieldEntry]]

#: Sentinel: the error path does not match the recorded data.
_MISSING = object()
#: Sentinel: the error path continues below a null value in the data.
_NULL_BELOW = object()


def validate_response(
    schema: GraphQLSchema,
    document: DocumentNode | str,
    response: Mapping[str, Any] | Any,
    variables: Mapping[str, Any] | None = None,
    *,
    operation_name: str | None = None,
    scalar_checkers: Mapping[str, ScalarOutputChecker] | None = None,
) -> ResponseValidationReport:
    """Validate a recorded response against the schema and the request.

    :param schema: the schema describing the service contract
    :param document: the executed query document, as parsed AST or SDL string
    :param response: the full response JSON (``{"data": ..., "errors": ...}``),
        or an :class:`~graphql.execution.ExecutionResult`
    :param variables: the variable values of the request
    :param operation_name: name of the operation to validate when the document
        contains several operations
    :param scalar_checkers: output value checkers for custom scalars, keyed by
        scalar type name
    :returns: a :class:`ResponseValidationReport` with every finding

    The check does not execute resolvers. It walks the selection of the
    selected operation (honoring aliases as well as ``@skip`` and
    ``@include``) and compares it with the recorded JSON. Response errors are
    accounted for: their ``path`` entries must correspond to the request and
    their non-null propagation determines which nulls in ``data`` are legal.

    >>> schema = build_schema(
    ...     "type Query { me: User } type User { id: ID! name: String }"
    ... )
    >>> response = {"data": {"me": {"id": "1", "name": None}}}
    >>> report = validate_response(schema, "{ me { id name } }", response)
    >>> report.valid
    True
    >>> bad = {"data": {"me": {"name": "Ada"}}}
    >>> report = validate_response(schema, "{ me { id name } }", bad)
    >>> report.valid
    False
    >>> report.issues[0].kind.value
    'missing-key'
    """
    report = ResponseValidationReport()
    if isinstance(document, str):
        document = parse(document)

    operation = get_operation_ast(document, operation_name)
    if operation is None:
        report.issues.append(
            ResponseValidationIssue(
                ResponseValidationIssueKind.MALFORMED_RESPONSE,
                None,
                "Could not determine the operation to validate.",
            )
        )
        return report

    root_type = schema.get_root_type(operation.operation)
    if root_type is None:
        report.issues.append(
            ResponseValidationIssue(
                ResponseValidationIssueKind.MALFORMED_RESPONSE,
                None,
                f"Schema does not define a {operation.operation.value} root type.",
            )
        )
        return report

    variable_values = _prepare_variable_values(schema, operation, variables)

    result = _as_result_mapping(response)
    if result is None:
        report.issues.append(
            ResponseValidationIssue(
                ResponseValidationIssueKind.MALFORMED_RESPONSE,
                None,
                "Response must be an object with optional 'data' and 'errors'.",
            )
        )
        return report

    validator = _ResponseValidator(
        schema,
        document,
        variable_values,
        dict(scalar_checkers or {}),
        report,
    )

    data = result.get("data")
    root_fields = validator.collect_fields(operation.selection_set)

    errors = result.get("errors")
    if errors is not None:
        if not isinstance(errors, list):
            validator.issue(
                ResponseValidationIssueKind.MALFORMED_RESPONSE,
                None,
                "'errors' must be a list.",
            )
        else:
            for index, error in enumerate(errors):
                validator.validate_error(
                    error, index, root_type, root_fields, data
                )
        validator.check_error_data(data)

    if data is None:
        if () not in validator.nulled_positions:
            validator.issue(
                ResponseValidationIssueKind.NON_NULL_VIOLATION,
                (),
                "Response data is null without an error propagated to the root.",
            )
    elif not isinstance(data, dict):
        validator.issue(
            ResponseValidationIssueKind.MALFORMED_RESPONSE,
            (),
            "Response data must be an object.",
        )
    else:
        validator.validate_object(root_type, root_fields, data, ())

    return report


def _prepare_variable_values(
    schema: GraphQLSchema,
    operation: OperationDefinitionNode,
    variables: Mapping[str, Any] | None,
) -> VariableValues:
    """Coerce request variables, falling back to the raw mapping.

    Only the boolean ``if`` arguments of ``@skip`` and ``@include`` are read
    here, so coercibility problems of other variables must not prevent the
    response check.
    """
    raw = dict(variables or {})
    var_defs = operation.variable_definitions or ()
    prepared = get_variable_values(schema, var_defs, raw)
    if isinstance(prepared, list):
        return VariableValues(sources={}, coerced=raw)
    return prepared


def _as_result_mapping(response: Any) -> dict[str, Any] | None:
    """Normalize the response envelope into a plain dictionary."""
    formatted = getattr(type(response), "formatted", None)
    if isinstance(formatted, property):
        response = formatted.fget(response)
    elif callable(getattr(response, "formatted", None)):
        response = response.formatted()
    if hasattr(response, "errors") and hasattr(response, "data"):
        return {"data": response.data, "errors": response.errors}
    if isinstance(response, dict):
        if "data" not in response and "errors" not in response:
            return None
        return response
    return None


class _ResponseValidator:
    """Walks selection and response, accumulating issues."""

    def __init__(
        self,
        schema: GraphQLSchema,
        document: DocumentNode,
        variable_values: VariableValues,
        scalar_checkers: dict[str, ScalarOutputChecker],
        report: ResponseValidationReport,
    ) -> None:
        self.schema = schema
        self.variable_values = variable_values
        self.scalar_checkers = scalar_checkers
        self.report = report
        self.fragments: dict[str, FragmentDefinitionNode] = {
            definition.name.value: definition
            for definition in document.definitions
            if isinstance(definition, FragmentDefinitionNode)
        }
        self._subfields_cache: dict[
            tuple[GraphQLObjectType, tuple[int, ...]], GroupedFields
        ] = {}
        # Error paths that matched the request, for a data-consistency pass
        # after all propagation positions are known.
        self._pending_error_paths: list[
            tuple[
                list[str | int],
                list[bool],
                bool,
                tuple[str | int, ...] | None,
                str | None,
            ]
        ] = []

    @property
    def nulled_positions(self) -> set[tuple[str | int, ...]]:
        return self.report.nulled_positions

    def issue(
        self,
        kind: ResponseValidationIssueKind,
        path: tuple[str | int, ...] | None,
        message: str,
    ) -> None:
        self.report.issues.append(ResponseValidationIssue(kind, path, message))

    # -- Selection collection ---------------------------------------------

    def collect_fields(
        self,
        selection_set: Any,
        conditions: tuple[str, ...] = (),
    ) -> GroupedFields:
        """Collect selected fields keyed by response key.

        Conditions are the names of the type conditions of enclosing
        fragments. ``@skip``/``@include`` are evaluated against the variables;
        deferred selections are omitted because they arrive in later payloads.
        """
        grouped: GroupedFields = {}
        self._collect_selections(
            selection_set.selections, conditions, grouped, set()
        )
        return grouped

    def _collect_selections(
        self,
        selections: Any,
        conditions: tuple[str, ...],
        grouped: GroupedFields,
        visited_fragments: set[str],
    ) -> None:
        for selection in selections:
            if isinstance(selection, FieldNode):
                if self._is_included(selection):
                    key = (
                        selection.alias.value
                        if selection.alias
                        else selection.name.value
                    )
                    grouped.setdefault(key, []).append(
                        FieldEntry(selection, conditions)
                    )
            elif isinstance(selection, InlineFragmentNode):
                if not self._is_included(selection):
                    continue
                if self._is_deferred(selection):
                    continue
                nested_conditions = self._extend_conditions(
                    conditions, selection.type_condition
                )
                self._collect_selections(
                    selection.selection_set.selections,
                    nested_conditions,
                    grouped,
                    visited_fragments,
                )
            elif isinstance(selection, FragmentSpreadNode):
                if not self._is_included(selection):
                    continue
                fragment_name = selection.name.value
                if fragment_name in visited_fragments:
                    continue
                fragment = self.fragments.get(fragment_name)
                if fragment is None or self._is_deferred(selection):
                    continue
                nested_conditions = self._extend_conditions(
                    conditions, fragment.type_condition
                )
                visited_fragments.add(fragment_name)
                self._collect_selections(
                    fragment.selection_set.selections,
                    nested_conditions,
                    grouped,
                    visited_fragments,
                )

    def _extend_conditions(
        self,
        conditions: tuple[str, ...],
        type_condition: Any,
    ) -> tuple[str, ...]:
        if type_condition is None:
            return conditions
        name = type_condition.name.value
        if conditions and conditions[-1] == name:
            return conditions
        return (*conditions, name)

    def _is_included(self, node: Any) -> bool:
        skip = get_directive_values(
            GraphQLSkipDirective, node, self.variable_values
        )
        if skip and skip.get("if"):
            return False
        include = get_directive_values(
            GraphQLIncludeDirective, node, self.variable_values
        )
        return not (include and not include.get("if"))

    def _is_deferred(self, node: Any) -> bool:
        defer = get_directive_values(
            GraphQLDeferDirective, node, self.variable_values
        )
        return bool(defer and defer.get("if") is not False)

    # -- Schema helpers ----------------------------------------------------

    def _condition_type(self, name: str) -> GraphQLNamedType | None:
        return self.schema.get_type(name)

    def _condition_applies(
        self, conditions: tuple[str, ...], object_type: GraphQLObjectType
    ) -> bool:
        for name in conditions:
            if name == object_type.name:
                continue
            condition_type = self._condition_type(name)
            if condition_type is None:
                return False
            if is_abstract_type(condition_type):
                if not self.schema.is_sub_type(condition_type, object_type):
                    return False
            elif condition_type is not object_type:
                return False
        return True

    def _field_definition(
        self, object_type: GraphQLObjectType, field_name: str
    ) -> GraphQLField | None:
        if field_name == "__typename":
            return TypeNameMetaFieldDef
        if field_name.startswith("__"):
            if object_type is self.schema.query_type:
                if field_name == "__schema":
                    return SchemaMetaFieldDef
                if field_name == "__type":
                    return TypeMetaFieldDef
            return None
        return self.schema.get_field(object_type, field_name)

    def _subfields(
        self, object_type: GraphQLObjectType, entries: list[FieldEntry]
    ) -> GroupedFields:
        cache_key = (object_type, tuple(id(entry.node) for entry in entries))
        cached = self._subfields_cache.get(cache_key)
        if cached is not None:
            return cached
        grouped: GroupedFields = {}
        visited: set[str] = set()
        for entry in entries:
            selection_set = entry.node.selection_set
            if selection_set is None:
                continue
            self._collect_selections(
                selection_set.selections, entry.conditions, grouped, visited
            )
        self._subfields_cache[cache_key] = grouped
        return grouped

    # -- Object validation -------------------------------------------------

    def validate_object(
        self,
        object_type: GraphQLObjectType,
        fields: GroupedFields,
        data: dict[str, Any],
        path: tuple[str | int, ...],
    ) -> None:
        for key, entries in fields.items():
            field_path = (*path, key)
            applicable = [
                entry
                for entry in entries
                if self._condition_applies(entry.conditions, object_type)
            ]
            if not applicable:
                if key in data:
                    self.issue(
                        ResponseValidationIssueKind.UNEXPECTED_KEY,
                        field_path,
                        f"Key '{key}' is not selected on type '{object_type.name}'.",
                    )
                continue

            field_name = applicable[0].node.name.value
            field_def = self._field_definition(object_type, field_name)
            if field_def is None:
                if key in data and not field_name.startswith("__"):
                    self.issue(
                        ResponseValidationIssueKind.UNEXPECTED_KEY,
                        field_path,
                        f"Field '{field_name}' does not exist on type"
                        f" '{object_type.name}'.",
                    )
                continue

            if key not in data:
                self.issue(
                    ResponseValidationIssueKind.MISSING_KEY,
                    field_path,
                    f"Missing key '{key}' for field '{field_name}' on type"
                    f" '{object_type.name}'.",
                )
                continue

            self.validate_value(
                field_def.type, applicable, data[key], field_path
            )

        for key in data.keys() - fields.keys():
            self.issue(
                ResponseValidationIssueKind.UNEXPECTED_KEY,
                (*path, key),
                f"Unexpected key '{key}' on type '{object_type.name}'.",
            )

    # -- Value validation --------------------------------------------------

    def validate_value(
        self,
        return_type: GraphQLOutputType,
        entries: list[FieldEntry],
        value: Any,
        path: tuple[str | int, ...],
    ) -> None:
        if _get_named_type(return_type).name.startswith("__"):
            # Introspection positions are not part of the service contract.
            return
        if is_non_null_type(return_type):
            inner_type: GraphQLOutputType = return_type.of_type
            if value is None:
                if path not in self.nulled_positions:
                    self.issue(
                        ResponseValidationIssueKind.NON_NULL_VIOLATION,
                        path,
                        "Non-null position is explicitly null without a"
                        " propagated error.",
                    )
                return
            self.validate_value(inner_type, entries, value, path)
            return

        if value is None:
            return

        if is_list_type(return_type):
            if not isinstance(value, list):
                self.issue(
                    ResponseValidationIssueKind.LIST_LEVEL_ERROR,
                    path,
                    f"Expected a list for type '{return_type}', got"
                    f" {inspect(value)}.",
                )
                return
            item_type = return_type.of_type
            for index, item in enumerate(value):
                item_path = (*path, index)
                if item is None:
                    if is_non_null_type(item_type):
                        if item_path not in self.nulled_positions:
                            self.issue(
                                ResponseValidationIssueKind.NON_NULL_VIOLATION,
                                item_path,
                                "Non-null list item is explicitly null without"
                                " a propagated error.",
                            )
                    continue
                if is_non_null_type(item_type):
                    self.validate_value(item_type.of_type, entries, item, item_path)
                else:
                    self.validate_value(item_type, entries, item, item_path)
            return

        if is_scalar_type(return_type):
            self.validate_scalar(return_type, value, path)
            return

        if is_enum_type(return_type):
            if value not in return_type.values:
                self.issue(
                    ResponseValidationIssueKind.INVALID_SCALAR,
                    path,
                    f"Value {inspect(value)} is not a valid enum value of"
                    f" '{return_type.name}'.",
                )
            return

        if not isinstance(value, dict):
            self.issue(
                ResponseValidationIssueKind.LIST_LEVEL_ERROR,
                path,
                f"Expected an object for type '{return_type}', got"
                f" {inspect(value)}.",
            )
            return

        if is_abstract_type(return_type):
            self.validate_abstract(return_type, entries, value, path)
            return

        subfields = self._subfields(return_type, entries)
        self.validate_object(return_type, subfields, value, path)

    def validate_scalar(
        self,
        scalar_type: GraphQLScalarType,
        value: Any,
        path: tuple[str | int, ...],
    ) -> None:
        checker = self.scalar_checkers.get(scalar_type.name)
        if checker is not None:
            reason = checker(scalar_type, value)
            if reason is not None:
                self.issue(
                    ResponseValidationIssueKind.INVALID_SCALAR,
                    path,
                    f"Invalid value for scalar '{scalar_type.name}': {reason}",
                )
            return

        reason = _check_builtin_scalar(scalar_type, value)
        if reason is not None:
            self.issue(
                ResponseValidationIssueKind.INVALID_SCALAR,
                path,
                f"Invalid value for scalar '{scalar_type.name}': {reason}",
            )

    def validate_abstract(
        self,
        abstract_type: GraphQLAbstractType,
        entries: list[FieldEntry],
        value: dict[str, Any],
        path: tuple[str | int, ...],
    ) -> None:
        """Validate an interface/union position without a type resolver.

        The runtime type can only be trusted when the response reveals it via
        a ``__typename`` matching a possible type. Without that information a
        key compatible with several implementations is not accepted by
        guessing one: unconditionally selected fields that every candidate
        shares are still checked, while type-conditioned branches and
        ambiguous common fields are reported as undecidable.
        """
        fields = self._collect_grouped(entries)
        possible_types = self.schema.get_possible_types(abstract_type)

        if "__typename" in value:
            typename = value["__typename"]
            resolved = next(
                (
                    candidate
                    for candidate in possible_types
                    if candidate.name == typename
                ),
                None,
            )
            if resolved is None:
                self.issue(
                    ResponseValidationIssueKind.INVALID_TYPENAME,
                    (*path, "__typename"),
                    f"{inspect(typename)} is not a possible type of"
                    f" '{abstract_type.name}'.",
                )
                return
            subfields = self._subfields(resolved, entries)
            self.validate_object(resolved, subfields, value, path)
            return

        candidates = list(possible_types)
        candidate_names = ", ".join(sorted(c.name for c in candidates))

        # Every present key must be explainable by at least one candidate;
        # the intersection of explaining candidates identifies the runtime
        # type only when __typename already did. Missing conditional keys do
        # not shrink the candidates (the branch may simply be inapplicable).
        explaining: dict[str, list[GraphQLObjectType]] = {}
        unexplainable: list[str] = []
        for key, key_entries in fields.items():
            if key not in value:
                continue
            explainers = self._explaining_candidates(
                candidates, key_entries
            )
            if explainers:
                explaining[key] = explainers
            else:
                unexplainable.append(key)

        for key in unexplainable:
            self.issue(
                ResponseValidationIssueKind.UNEXPECTED_KEY,
                (*path, key),
                f"Key '{key}' cannot be produced by any possible type of"
                f" '{abstract_type.name}'.",
            )
        for key in value.keys() - fields.keys():
            self.issue(
                ResponseValidationIssueKind.UNEXPECTED_KEY,
                (*path, key),
                f"Unexpected key '{key}' on abstract type"
                f" '{abstract_type.name}'.",
            )
        if unexplainable or any(
            key not in fields for key in value
        ):
            return

        undecidable_keys: list[str] = []
        for key, key_entries in fields.items():
            field_name = key_entries[0].node.name.value
            if field_name == "__typename":
                continue

            applicable = self._candidates_for_entries(
                candidates, key_entries
            )
            field_types = {
                field_def.type
                for candidate in applicable
                if (field_def := self._field_definition(candidate, field_name))
                is not None
            }
            common_to_all = len(applicable) == len(candidates)
            if not common_to_all:
                # Type-conditioned branch: only the response can tell which
                # implementation applies, and it does not.
                if key in value:
                    undecidable_keys.append(key)
                    names = ", ".join(sorted(c.name for c in applicable))
                    self.issue(
                        ResponseValidationIssueKind.UNDECIDABLE_BRANCH,
                        (*path, key),
                        f"Branch for key '{key}' only applies to {names} and"
                        " the response does not identify the runtime type of"
                        f" '{abstract_type.name}'; it cannot be decided.",
                    )
                continue

            if key not in value:
                # A field common to every possible type is mandatory when
                # selected unconditionally.
                self.issue(
                    ResponseValidationIssueKind.MISSING_KEY,
                    (*path, key),
                    f"Missing key '{key}' for field '{field_name}'.",
                )
                continue

            if len({_type_signature(field_type) for field_type in field_types}) != 1:
                undecidable_keys.append(key)
                self.issue(
                    ResponseValidationIssueKind.UNDECIDABLE_BRANCH,
                    (*path, key),
                    f"Field '{field_name}' has different types across"
                    " possible implementations; it cannot be decided"
                    " without a runtime type.",
                )
                continue

            self.validate_value(
                next(iter(field_types)),
                key_entries,
                value[key],
                (*path, key),
            )

        if undecidable_keys:
            self.issue(
                ResponseValidationIssueKind.UNDECIDABLE_BRANCH,
                path,
                f"Runtime type of '{abstract_type.name}' is unknown"
                f" (candidates: {candidate_names}); branches for keys"
                f" {', '.join(undecidable_keys)} could not be decided.",
            )

    def _collect_grouped(self, entries: list[FieldEntry]) -> GroupedFields:
        grouped: GroupedFields = {}
        visited: set[str] = set()
        for entry in entries:
            selection_set = entry.node.selection_set
            if selection_set is None:
                continue
            self._collect_selections(
                selection_set.selections, entry.conditions, grouped, visited
            )
        return grouped

    def _explaining_candidates(
        self,
        candidates: list[GraphQLObjectType],
        entries: list[FieldEntry],
    ) -> list[GraphQLObjectType]:
        """Candidates on which a present selected key could legitimately be."""
        field_name = entries[0].node.name.value
        explaining = []
        for candidate in candidates:
            if not any(
                self._condition_applies(entry.conditions, candidate)
                for entry in entries
            ):
                continue
            if field_name == "__typename" or self._field_definition(
                candidate, field_name
            ) is not None:
                explaining.append(candidate)
        return explaining

    def _candidates_for_entries(
        self,
        candidates: list[GraphQLObjectType],
        entries: list[FieldEntry],
    ) -> list[GraphQLObjectType]:
        return [
            candidate
            for candidate in candidates
            if any(
                self._condition_applies(entry.conditions, candidate)
                for entry in entries
            )
        ]

    # -- Error path validation --------------------------------------------

    def validate_error(
        self,
        error: Any,
        index: int,
        root_type: GraphQLObjectType,
        root_fields: GroupedFields,
        data: Any,
    ) -> None:
        if isinstance(error, GraphQLError):
            error_path = list(error.path or [])
        elif isinstance(error, dict):
            raw_path = error.get("path")
            if raw_path is None:
                return
            if not isinstance(raw_path, list) or not all(
                isinstance(segment, (str, int))
                and not isinstance(segment, bool)
                for segment in raw_path
            ):
                self.issue(
                    ResponseValidationIssueKind.MALFORMED_RESPONSE,
                    ("errors", index),
                    f"errors[{index}].path must be a list of strings and"
                    " integers.",
                )
                return
            error_path = raw_path
        else:
            self.issue(
                ResponseValidationIssueKind.MALFORMED_RESPONSE,
                ("errors", index),
                f"errors[{index}] must be an error object.",
            )
            return

        if not error_path:
            return

        request = self._walk_request_path(
            root_type, root_fields, error_path, data
        )
        if request is not None:
            self._pending_error_paths.append(
                (error_path, request[0], request[1], request[2], request[3])
            )

    def check_error_data(self, data: Any) -> None:
        """Second pass: match every structurally valid error path to data."""
        null_terminals: list[tuple[list[str | int], list[bool], bool]] = []
        nonnull_terminals: list[
            tuple[list[str | int], list[bool], bool]
        ] = []
        below_null: list[
            tuple[list[str | int], list[bool], tuple[str | int, ...]]
        ] = []
        stopped: list[
            tuple[list[str | int], tuple[str | int, ...], str]
        ] = []
        for (
            error_path,
            slot_nonnulls,
            terminal_slot_nonnull,
            stop_prefix,
            stop_reason,
        ) in self._pending_error_paths:
            if stop_prefix is not None:
                stopped.append((error_path, stop_prefix, stop_reason or "missing"))
                continue
            data_node = self._walk_data_path(
                error_path, data, slot_nonnulls, terminal_slot_nonnull
            )
            if (
                isinstance(data_node, tuple)
                and data_node
                and data_node[0] is _NULL_BELOW
            ):
                below_null.append((error_path, slot_nonnulls, data_node[1]))
            elif data_node is not _MISSING:
                record = (error_path, slot_nonnulls, terminal_slot_nonnull)
                if data_node is None:
                    null_terminals.append(record)
                else:
                    nonnull_terminals.append(record)

        # Propagate null terminals first. A path ending at a non-null
        # terminal below a propagated null cannot be read from data, so such
        # paths seed additional propagations (their terminal is
        # non-observable). Iterate until the set of nulled positions is
        # stable.
        for error_path, slot_nonnulls, terminal_slot_nonnull in null_terminals:
            self._record_propagation(
                error_path, slot_nonnulls, terminal_slot_nonnull
            )

        pending = list(below_null)
        while pending:
            remaining: list[
                tuple[list[str | int], list[bool], tuple[str | int, ...]]
            ] = []
            for error_path, slot_nonnulls, null_prefix in pending:
                if self._position_is_nulled(null_prefix):
                    continue
                remaining.append((error_path, slot_nonnulls, null_prefix))
            progressed = False
            still_remaining: list[
                tuple[list[str | int], list[bool], tuple[str | int, ...]]
            ] = []
            for error_path, slot_nonnulls, null_prefix in remaining:
                # The terminal is non-observable; if it is non-null it must
                # have propagated to the observed null position.
                if self._propagate_from_path(
                    error_path, slot_nonnulls, null_prefix
                ):
                    progressed = True
                else:
                    still_remaining.append(
                        (error_path, slot_nonnulls, null_prefix)
                    )
            pending = still_remaining
            if not progressed:
                break

        for _error_path, _slot_nonnulls, null_prefix in pending:
            self.issue(
                ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                null_prefix,
                "Error path continues below a null value that no error"
                " propagates to.",
            )

        # Paths that stop on a selected position but continue beyond it only
        # fail when the recorded data actually contains that position: the
        # continuation is then unaddressable by the request.
        for error_path, stop_prefix, stop_reason in stopped:
            if stop_reason == "missing":
                segment = (
                    error_path[len(stop_prefix) - 1]
                    if len(stop_prefix) <= len(error_path)
                    else None
                )
                self.issue(
                    ResponseValidationIssueKind.INVALID_ERROR_PATH,
                    stop_prefix,
                    f"Error path segment {inspect(segment)} does not"
                    " correspond to a selected field of the request.",
                )
                continue

            node = data
            reachable = True
            for segment in stop_prefix:
                if isinstance(segment, int) and not isinstance(segment, bool):
                    if not isinstance(node, list) or segment >= len(node):
                        reachable = False
                        break
                    node = node[segment]
                else:
                    if not isinstance(node, dict) or segment not in node:
                        reachable = False
                        break
                    node = node[segment]
            continuation_depth = len(stop_prefix) + 1
            continuation = tuple(error_path[:continuation_depth])
            if not reachable:
                self.issue(
                    ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                    stop_prefix,
                    "Error path does not reach a position selected by the"
                    " request.",
                )
            elif continuation_depth < len(error_path):
                self.issue(
                    ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                    continuation,
                    "Error path continues beyond a position that the"
                    " request does not select.",
                )
            else:
                # The final segment names a child of a leaf position.
                self.issue(
                    ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                    continuation,
                    "Error path ends below a position that the request"
                    " does not select.",
                )

        # Verify terminals. A non-null nullable terminal with no propagated
        # descendant is an error the response did not reflect.
        for error_path, _slot_nonnulls, terminal_slot_nonnull in (
            nonnull_terminals
        ):
            path_tuple = tuple(error_path)
            if self._position_is_nulled(path_tuple):
                continue
            if not terminal_slot_nonnull:
                self.issue(
                    ResponseValidationIssueKind.ERROR_WITHOUT_NULL,
                    path_tuple,
                    "Error at a nullable position but the response value is"
                    " not null and no descendant error propagates to it.",
                )

    def _propagate_from_path(
        self,
        error_path: list[str | int],
        slot_nonnulls: list[bool],
        null_prefix: tuple[str | int, ...],
    ) -> bool:
        """Justify an observed null above an unobservable terminal.

        The terminal slot is assumed to have errored to null. Propagate
        upward across consecutive non-null slots to the nearest nullable
        slot; the observed null must be a position on that chain.
        """
        # Non-null propagation mapping. The terminal is null. Walking
        # upward, a non-null segment i pushes the null to its parent
        # position ``[:i]``; the nearest nullable segment is the landing
        # (a nullable field key nulls its own position ``[:i+1]``, a
        # nullable list position nulls the parent list ``[:i]``). With only
        # non-null wrappers the root ``()`` is null. An observed null is
        # explained when it is any position on this chain.
        chain: list[tuple[str | int, ...]] = [tuple(error_path)]
        for index in range(len(error_path) - 1, -1, -1):
            if not slot_nonnulls[index]:
                if isinstance(error_path[index], str):
                    chain.append(tuple(error_path[: index + 1]))
                else:
                    chain.append(tuple(error_path[:index]))
                break
            chain.append(tuple(error_path[:index]))
        if null_prefix not in chain:
            return False
        cutoff = chain.index(null_prefix)
        for position in chain[: cutoff + 1]:
            self.nulled_positions.add(position)
        return True

    def _walk_request_path(
        self,
        root_type: GraphQLObjectType,
        root_fields: GroupedFields,
        error_path: list[str | int],
        data: Any,
    ) -> tuple[
        list[bool], bool, tuple[str | int, ...] | None, str | None
    ] | None:
        """Validate the path structurally.

        Returns the slot non-null wrappers, the terminal slot nullability and
        the response prefix at which the path stops following a selected
        field, together with the stop reason (``"missing"`` when the segment
        itself is not selected, or ``"leaf"`` when a continuation follows a
        leaf position). A ``leaf`` stop is not necessarily invalid: it can
        still be a field error if the data walk shows it cannot reach that
        position.
        """
        possible_types: set[GraphQLObjectType] = {root_type}
        fields = root_fields
        type_states: set[GraphQLOutputType] | None = None
        subfield_entries: list[FieldEntry] | None = None
        slot_nonnulls: list[bool] = []
        stop: tuple[tuple[str | int, ...], str] | None = None
        data_node: Any = data

        for depth, segment in enumerate(error_path):
            path_prefix = tuple(error_path[: depth + 1])
            is_terminal = depth == len(error_path) - 1

            if isinstance(segment, int) and not isinstance(segment, bool):
                if stop is not None:
                    return slot_nonnulls, False, stop[0], stop[1]
                if type_states is not None and not self._list_item_types(type_states):
                    # A previous field position was a leaf; this index is a
                    # continuation of an unaddressable position.
                    leaf_prefix = tuple(error_path[:depth])
                    return slot_nonnulls, False, leaf_prefix, "leaf"
                if segment < 0:
                    self.issue(
                        ResponseValidationIssueKind.INVALID_ERROR_PATH,
                        path_prefix,
                        f"Error path index {segment} must not be negative.",
                    )
                    return None
                item_types = self._list_item_types(type_states)
                if item_types is None:
                    reason = "leaf" if stop is None else stop[1]
                    return slot_nonnulls, False, path_prefix, reason
                slot_nonnull = all(
                    is_non_null_type(item_type) for item_type in item_types
                )
                slot_nonnulls.append(slot_nonnull)
                if is_terminal:
                    return slot_nonnulls, slot_nonnull, None, None
                type_states = self._unwrap_item_types(item_types)
                possible_types = self._possible_types_from_states(type_states)
                if not possible_types:
                    stop = (path_prefix, "leaf")
                    continue
                if isinstance(data_node, list) and segment < len(data_node):
                    data_node = data_node[segment]
                    possible_types = self._narrow_with_typename(
                        possible_types, data_node
                    )
                fields = self._merged_subfields(
                    possible_types, subfield_entries or []
                )
                continue

            if stop is not None:
                return slot_nonnulls, False, stop[0], stop[1]
            if depth > 0 and type_states is not None and not (
                self._possible_types_from_states(type_states)
            ):
                # Previous segment was a leaf scalar/enum; this field segment
                # cannot be selected, but the data walk decides whether the
                # leaf position actually exists in the recorded payload.
                leaf_prefix = tuple(error_path[:depth])
                return slot_nonnulls, False, leaf_prefix, "leaf"
            entries = fields.get(segment) if fields is not None else None
            if entries is None:
                return slot_nonnulls, False, path_prefix, "missing"

            field_name = entries[0].node.name.value
            field_types = self._applicable_field_types(
                possible_types, field_name, entries
            )
            if not field_types:
                return slot_nonnulls, False, path_prefix, "missing"
            slot_nonnull = all(
                is_non_null_type(field_type) for field_type in field_types
            )
            possible_types_after = self._possible_types_from_states(field_types)
            slot_nonnulls.append(slot_nonnull)
            if is_terminal:
                return slot_nonnulls, slot_nonnull, None, None
            type_states = field_types
            possible_types = possible_types_after
            if isinstance(data_node, dict) and segment in data_node:
                data_node = data_node[segment]
                possible_types = self._narrow_with_typename(
                    possible_types, data_node
                )
            subfield_entries = entries
            fields = self._merged_subfields(possible_types, entries)

        return None  # pragma: no cover

    def _possible_types_from_states(
        self, type_states: set[GraphQLOutputType]
    ) -> set[GraphQLObjectType]:
        possible: set[GraphQLObjectType] = set()
        for output_type in type_states:
            named_type = _get_named_type(output_type)
            if isinstance(named_type, GraphQLObjectType):
                possible.add(named_type)
            elif is_abstract_type(named_type):
                possible.update(self.schema.get_possible_types(named_type))
        return possible

    def _narrow_with_typename(
        self,
        possible_types: set[GraphQLObjectType],
        data_node: Any,
    ) -> set[GraphQLObjectType]:
        """Narrow abstract candidates when data reveals ``__typename``."""
        if not isinstance(data_node, dict):
            return possible_types
        typename = data_node.get("__typename")
        if typename is None:
            return possible_types
        narrowed = {
            candidate
            for candidate in possible_types
            if candidate.name == typename
        }
        return narrowed or possible_types


    def _record_propagation(
        self,
        error_path: list[str | int],
        slot_nonnulls: list[bool],
        terminal_slot_nonnull: bool,
    ) -> None:
        """Record the positions nulled by non-null propagation.

        Purely determined by the request types: a terminal non-null slot
        propagates upward across non-null wrappers to the nearest nullable
        parent; a terminal nullable slot only nulls itself.
        """
        if not terminal_slot_nonnull:
            self.nulled_positions.add(tuple(error_path))
            return
        landing = 0
        for wrapper in reversed(slot_nonnulls):
            if not wrapper:
                break
            landing += 1
        for level in range(landing + 1):
            self.nulled_positions.add(
                tuple(error_path[: len(error_path) - level])
            )

    def _walk_data_path(
        self,
        error_path: list[str | int],
        data: Any,
        slot_nonnulls: list[bool],
        terminal_slot_nonnull: bool,
    ) -> Any:
        """Structurally match an error path against the recorded data.

        Returns the terminal value, or ``_MISSING`` when a mismatch was
        reported. Null values below the terminal are not judged here: they
        may be justified by another error whose propagation is only known
        once every terminal has been inspected.
        """
        data_node: Any = data
        for depth, segment in enumerate(error_path):
            path_prefix = tuple(error_path[: depth + 1])
            is_terminal = depth == len(error_path) - 1

            if isinstance(segment, int) and not isinstance(segment, bool):
                if not isinstance(data_node, list):
                    self.issue(
                        ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                        path_prefix,
                        "Error path index does not correspond to a list in"
                        " data.",
                    )
                    return _MISSING
                if segment >= len(data_node):
                    self.issue(
                        ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                        path_prefix,
                        f"Error path index {segment} is out of range for the"
                        f" list of length {len(data_node)}.",
                    )
                    return _MISSING
                data_node = data_node[segment]
            else:
                if not isinstance(data_node, dict):
                    self.issue(
                        ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                        path_prefix,
                        "Error path continues through a non-object value in"
                        " data.",
                    )
                    return _MISSING
                if segment not in data_node:
                    self.issue(
                        ResponseValidationIssueKind.ERROR_PATH_DATA_MISMATCH,
                        path_prefix,
                        f"Error path segment '{segment}' is absent from"
                        " data.",
                    )
                    return _MISSING
                data_node = data_node[segment]

            if not is_terminal and data_node is None:
                # Recorded below a null; may be caused by another error.
                return _NULL_BELOW, path_prefix

        return data_node

    def _position_is_nulled(
        self, path_tuple: tuple[str | int, ...]
    ) -> bool:
        return any(
            len(path_tuple) >= len(nulled)
            and path_tuple[: len(nulled)] == nulled
            for nulled in self.nulled_positions
        )

    def _list_item_types(
        self,
        type_states: set[GraphQLOutputType] | None,
    ) -> set[GraphQLOutputType] | None:
        if type_states is None:
            return None
        item_types: set[GraphQLOutputType] = set()
        found_list = False
        for output_type in type_states:
            unwrapped = (
                output_type.of_type
                if is_non_null_type(output_type)
                else output_type
            )
            if not is_list_type(unwrapped):
                continue
            found_list = True
            item_types.add(unwrapped.of_type)
        return item_types if found_list else None

    def _unwrap_item_types(
        self, item_types: set[GraphQLOutputType]
    ) -> set[GraphQLOutputType]:
        return {
            output_type.of_type
            if is_non_null_type(output_type)
            else output_type
            for output_type in item_types
        }

    def _applicable_field_types(
        self,
        possible_types: set[GraphQLObjectType],
        field_name: str,
        entries: list[FieldEntry],
    ) -> set[GraphQLOutputType]:
        types: set[GraphQLOutputType] = set()
        for candidate in possible_types:
            if not any(
                self._condition_applies(entry.conditions, candidate)
                for entry in entries
            ):
                continue
            field_def = self._field_definition(candidate, field_name)
            if field_def is not None:
                types.add(field_def.type)
        return types

    def _merged_subfields(
        self,
        possible_types: set[GraphQLObjectType],
        entries: list[FieldEntry],
    ) -> GroupedFields | None:
        if not possible_types or not entries:
            return None
        merged: GroupedFields = {}
        for candidate in possible_types:
            entries_by_key = self._subfields(candidate, entries)
            for key, key_entries in entries_by_key.items():
                applicable = [
                    entry
                    for entry in key_entries
                    if self._condition_applies(entry.conditions, candidate)
                ]
                if applicable:
                    merged.setdefault(key, []).extend(applicable)
        return merged


def _type_signature(output_type: GraphQLOutputType) -> tuple:
    """Structural signature of an output type for cross-type comparison.

    Fields selected on several possible implementations can share the same
    shape even though their named type objects differ (for example the same
    ``id: ID!`` declared on several object types). The signature records the
    wrapper nullability at every level and the named type name.
    """
    wrappers: list[str] = []
    current = output_type
    while is_non_null_type(current) or is_list_type(current):
        if is_non_null_type(current):
            wrappers.append("!")
            current = current.of_type
        else:
            wrappers.append("[]")
            current = current.of_type
    return tuple(wrappers), current.name


def _get_named_type(
    output_type: GraphQLOutputType,
) -> GraphQLNamedType:
    while is_non_null_type(output_type) or is_list_type(output_type):
        output_type = output_type.of_type
    return output_type


def _check_builtin_scalar(scalar_type: GraphQLScalarType, value: Any) -> str | None:
    """Check JSON-level shape of the specified built-in scalars."""
    if scalar_type is GraphQLBoolean:
        if not isinstance(value, bool):
            return "expected a boolean"
    elif scalar_type is GraphQLInt:
        if isinstance(value, bool) or not isinstance(value, int):
            return "expected an integer"
        if not -(2**31) <= value <= 2**31 - 1:
            return "integer outside the Int range"
    elif scalar_type is GraphQLFloat:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a number"
    elif scalar_type is GraphQLString:
        if not isinstance(value, str):
            return "expected a string"
    elif scalar_type is GraphQLID:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return "expected a string or integer"
    return None
