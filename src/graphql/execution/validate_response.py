"""Validate GraphQL responses against a schema and a request document."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any, NamedTuple

from ..error import GraphQLError
from ..language import (
    DocumentNode,
    FragmentDefinitionNode,
    OperationDefinitionNode,
    OperationType,
)
from ..pyutils import Undefined, inspect
from ..type import (
    GraphQLAbstractType,
    GraphQLBoolean,
    GraphQLField,
    GraphQLFloat,
    GraphQLID,
    GraphQLInt,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    GraphQLType,
    SchemaMetaFieldDef,
    TypeMetaFieldDef,
    TypeNameMetaFieldDef,
    is_abstract_type,
    is_enum_type,
    is_interface_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
    is_scalar_type,
    is_specified_scalar_type,
)
from ..utilities import get_operation_ast
from .collect_fields import (
    FieldDetails,
    FragmentDetails,
    GroupedFieldSet,
    collect_fields,
    collect_subfields,
)
from .get_variable_signature import get_variable_signature
from .values import VariableValues, get_variable_values

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ResponseIssueKind",
    "ResponseValidationIssue",
    "ResponseValidationResult",
    "validate_response",
]


class ResponseIssueKind(Enum):
    """Kinds of issues reported by :func:`validate_response`."""

    MISSING_DATA = "missing_data"
    """The response does not contain a ``data`` entry."""

    MISSING_FIELD = "missing_field"
    """A selected response key is missing from the response."""

    UNEXPECTED_FIELD = "unexpected_field"
    """The response contains a key that was not selected."""

    UNEXPECTED_NULL = "unexpected_null"
    """An explicit null violates a non-null constraint."""

    EXPECTED_NULL = "expected_null"
    """An error was reported for this position, so it must be null."""

    EXPECTED_LIST = "expected_list"
    """A list was expected, but a non-list value was found."""

    UNEXPECTED_LIST = "unexpected_list"
    """A list was found where a non-list value was expected."""

    EXPECTED_OBJECT = "expected_object"
    """An object was expected, but a non-object value was found."""

    INVALID_SCALAR = "invalid_scalar"
    """A scalar value does not match its expected output type."""

    INVALID_ENUM = "invalid_enum"
    """An enum value is not one of the defined enum values."""

    INVALID_TYPENAME = "invalid_typename"
    """The ``__typename`` value is not a possible runtime type."""

    UNDETERMINED_ABSTRACT_TYPE = "undetermined_abstract_type"
    """The runtime type of an abstract type could not be determined."""

    UNMATCHED_ERROR_PATH = "unmatched_error_path"
    """An error path does not correspond to the request."""

    INVALID_VARIABLES = "invalid_variables"
    """The provided variable values cannot be coerced."""

    INVALID_RESPONSE = "invalid_response"
    """The response envelope itself is malformed."""


class ResponseValidationIssue(NamedTuple):
    """A single problem found while validating a GraphQL response."""

    kind: ResponseIssueKind
    """The kind of the issue."""

    message: str
    """A human-readable description of the issue."""

    path: tuple[str | int, ...] | None = None
    """The response path where the issue was found, if any."""


class ResponseValidationResult:
    """The outcome of validating a GraphQL response.

    :param issues: All issues found while validating the response.

    >>> from graphql import ResponseValidationResult
    >>> result = ResponseValidationResult([])
    >>> result.is_valid
    True
    """

    __slots__ = ("issues",)

    issues: list[ResponseValidationIssue]
    """All issues found while validating the response."""

    def __init__(self, issues: list[ResponseValidationIssue]) -> None:
        self.issues = issues

    @property
    def errors(self) -> list[ResponseValidationIssue]:
        """The issues that make the response invalid."""
        return [
            issue
            for issue in self.issues
            if issue.kind is not ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE
        ]

    @property
    def undetermined(self) -> list[ResponseValidationIssue]:
        """Branches that could not be checked for lack of type information."""
        return [
            issue
            for issue in self.issues
            if issue.kind is ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE
        ]

    @property
    def is_valid(self) -> bool:
        """Whether the response conforms to the request and the schema."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.is_valid

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} valid={self.is_valid}"
            f" issues={len(self.issues)}>"
        )


def validate_response(
    schema: GraphQLSchema,
    document: DocumentNode,
    response: Mapping[str, Any],
    operation_name: str | None = None,
    variable_values: dict[str, Any] | None = None,
    scalar_checkers: Mapping[str, Callable[[Any], bool]] | None = None,
) -> ResponseValidationResult:
    """Validate a GraphQL response against a schema and a request document.

    This is useful for gateways and testing tools that record or replay
    responses and want to verify them against the GraphQL contract without
    executing any resolvers. Only the schema, the request document, the
    variable values and the response itself are needed.

    :param schema: The GraphQL schema the request was executed against.
    :param document: The parsed GraphQL request document.
    :param response: The parsed JSON response with ``data`` and ``errors``.
    :param operation_name: The operation to validate against, needed if the
        document contains more than one operation.
    :param variable_values: The raw variable values of the request, used to
        evaluate ``@skip`` and ``@include`` directives.
    :param scalar_checkers: Output check functions for custom scalars, keyed
        by scalar type name. Each function receives the JSON value and returns
        whether it is a valid output value. Custom scalars without a checker
        are accepted without further inspection.
    :returns: A result object listing all issues that were found.

    The validator checks the actual response keys against the fields selected
    by the chosen operation, distinguishing missing keys, explicit null
    values, list nesting mistakes and non-null constraint violations. For
    abstract types, the runtime type is taken from the ``__typename`` value in
    the response; if it is missing, the branches that cannot be decided are
    reported as undetermined instead of being checked against an arbitrary
    possible type.

    If the response contains errors, each error path is matched against the
    request, and the null positions in ``data`` are checked against the
    non-null propagation of those errors. A single error only excuses the
    null position it propagates to, never the whole ``data`` value.

    >>> from graphql import build_schema, parse, validate_response
    >>> schema = build_schema('''
    ...     type Query {
    ...       user: User
    ...     }
    ...     type User {
    ...       id: ID!
    ...       name: String!
    ...     }
    ... ''')
    >>> document = parse('{ user { id name } }')
    >>> result = validate_response(
    ...     schema, document, {'data': {'user': {'id': '1'}}}
    ... )
    >>> result.is_valid
    False
    >>> result.issues[0].kind
    <ResponseIssueKind.MISSING_FIELD: 'missing_field'>
    >>> result.issues[0].path
    ('user', 'name')
    """
    operation = get_operation_ast(document, operation_name)
    if operation is None:
        if operation_name is not None:
            msg = f"Unknown operation named '{operation_name}'."
            raise GraphQLError(msg)
        msg = "Must provide an operation."
        raise GraphQLError(msg)

    fragments: dict[str, FragmentDetails] = {}
    for definition in document.definitions:
        if isinstance(definition, FragmentDefinitionNode):
            variable_signatures = None
            if definition.variable_definitions:
                variable_signatures = {}
                for var_def in definition.variable_definitions:
                    signature = get_variable_signature(schema, var_def)
                    if isinstance(signature, GraphQLError):
                        raise signature
                    variable_signatures[signature.name] = signature
            fragments[definition.name.value] = FragmentDetails(
                definition, variable_signatures
            )

    coerced_variables = get_variable_values(
        schema, operation.variable_definitions or (), variable_values or {}
    )
    if isinstance(coerced_variables, list):
        return ResponseValidationResult(
            [
                ResponseValidationIssue(
                    ResponseIssueKind.INVALID_VARIABLES, error.message
                )
                for error in coerced_variables
            ]
        )

    validator = _ResponseValidator(
        schema, operation, fragments, coerced_variables, scalar_checkers
    )
    return validator.validate(response)


def _format_path(path: tuple[str | int, ...]) -> str:
    """Format a response path for human-readable messages."""
    result = "data"
    for segment in path:
        if isinstance(segment, int):
            result += f"[{segment}]"
        else:
            result += f".{segment}"
    return result


def _is_valid_specified_scalar(type_: GraphQLType, value: Any) -> bool:
    """Check a JSON value against one of the built-in scalar types."""
    if type_ is GraphQLBoolean:
        return isinstance(value, bool)
    if type_ is GraphQLInt:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2**31) <= value < 2**31
        )
    if type_ is GraphQLFloat:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        )
    if type_ is GraphQLString:
        return isinstance(value, str)
    if type_ is GraphQLID:
        return isinstance(value, str) or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    return True


class _ResponseValidator:
    """Validate one response against one operation (internal)."""

    schema: GraphQLSchema
    operation: OperationDefinitionNode
    fragments: dict[str, FragmentDetails]
    variable_values: VariableValues
    scalar_checkers: Mapping[str, Callable[[Any], bool]]
    issues: list[ResponseValidationIssue]
    allowed_null_positions: set[tuple[str | int, ...]]
    has_pathless_errors: bool
    data: Any

    def __init__(
        self,
        schema: GraphQLSchema,
        operation: OperationDefinitionNode,
        fragments: dict[str, FragmentDetails],
        variable_values: VariableValues,
        scalar_checkers: Mapping[str, Callable[[Any], bool]] | None,
    ) -> None:
        self.schema = schema
        self.operation = operation
        self.fragments = fragments
        self.variable_values = variable_values
        self.scalar_checkers = scalar_checkers or {}
        self.issues = []
        self.allowed_null_positions = set()
        self.has_pathless_errors = False
        self.data = Undefined

    # -- entry point -------------------------------------------------

    def validate(self, response: Any) -> ResponseValidationResult:
        """Validate the response and return the validation result."""
        if not isinstance(response, Mapping):
            self._report(
                ResponseIssueKind.INVALID_RESPONSE,
                f"Response must be a JSON object, got {inspect(response)}.",
            )
            return ResponseValidationResult(self.issues)

        data = response.get("data", Undefined)
        self.data = data

        errors = response.get("errors")
        if errors is not None:
            if not isinstance(errors, list):
                self._report(
                    ResponseIssueKind.INVALID_RESPONSE,
                    f"Response 'errors' must be a list, got {inspect(errors)}.",
                )
            else:
                for error in errors:
                    self._check_error(error)

        if data is Undefined:
            if not errors:
                self._report(
                    ResponseIssueKind.MISSING_DATA,
                    "Response is missing the 'data' entry.",
                )
        elif data is None:
            if () not in self.allowed_null_positions and not self.has_pathless_errors:
                self._report(
                    ResponseIssueKind.UNEXPECTED_NULL,
                    "Unexpected null value for 'data'; no error explains it.",
                    (),
                )
        elif isinstance(data, Mapping):
            self._visit_object(data, self._root_type, None, ())
        else:
            self._report(
                ResponseIssueKind.INVALID_RESPONSE,
                f"Response 'data' must be an object or null,"
                f" got {inspect(data)}.",
                (),
            )
        return ResponseValidationResult(self.issues)

    # -- helpers -----------------------------------------------------

    def _report(
        self,
        kind: ResponseIssueKind,
        message: str,
        path: tuple[str | int, ...] | None = None,
    ) -> None:
        self.issues.append(ResponseValidationIssue(kind, message, path))

    @property
    def _root_type(self) -> GraphQLObjectType:
        operation = self.operation.operation
        schema = self.schema
        if operation is OperationType.QUERY:
            root_type = schema.query_type
        elif operation is OperationType.MUTATION:
            root_type = schema.mutation_type
        else:
            root_type = schema.subscription_type
        if root_type is None:
            msg = (
                "Schema is not configured to execute"
                f" {operation.value} operation."
            )
            raise GraphQLError(msg)
        return root_type

    def _collect_grouped_fields(
        self,
        runtime_type: GraphQLObjectType,
        field_details_list: list[FieldDetails] | None,
    ) -> GroupedFieldSet:
        if field_details_list is None:
            return collect_fields(
                self.schema,
                self.fragments,
                self.variable_values,
                runtime_type,
                self.operation,
            ).grouped_field_set
        return collect_subfields(
            self.schema,
            self.fragments,
            self.variable_values,
            self.operation,
            runtime_type,
            field_details_list,
        ).grouped_field_set

    def _get_field_def(
        self, parent_type: GraphQLType, field_name: str
    ) -> GraphQLField | None:
        if field_name == "__typename":
            return TypeNameMetaFieldDef
        if parent_type is self.schema.query_type:
            if field_name == "__schema":
                return SchemaMetaFieldDef
            if field_name == "__type":
                return TypeMetaFieldDef
        if is_object_type(parent_type) or is_interface_type(parent_type):
            return parent_type.fields.get(field_name)
        return None

    def _find_field(
        self,
        composite_type: GraphQLType,
        key: str,
        field_details_list: list[FieldDetails] | None,
    ) -> tuple[GraphQLField, list[FieldDetails]] | None:
        """Find the field with the given response key on a composite type."""
        if is_object_type(composite_type):
            candidates = [composite_type]
        elif is_abstract_type(composite_type):
            candidates = list(self.schema.get_possible_types(composite_type))
        else:
            return None
        for candidate in candidates:
            grouped = self._collect_grouped_fields(candidate, field_details_list)
            details = grouped.get(key)
            if details:
                field_def = self._get_field_def(candidate, details[0].node.name.value)
                if field_def is not None:
                    return field_def, details
        return None

    # -- error paths -------------------------------------------------

    def _check_error(self, error: Any) -> None:
        if not isinstance(error, Mapping):
            self._report(
                ResponseIssueKind.INVALID_RESPONSE,
                f"Response error entries must be objects, got {inspect(error)}.",
            )
            return
        path = error.get("path")
        if path is None:
            # Errors without a path are request errors. They excuse no
            # particular position in the data.
            self.has_pathless_errors = True
            return
        self._check_error_path(path)

    def _check_error_path(self, path: Any) -> None:
        if (
            not isinstance(path, (list, tuple))
            or not path
            or any(
                isinstance(segment, bool)
                or not isinstance(segment, (str, int))
                for segment in path
            )
        ):
            self._report(
                ResponseIssueKind.UNMATCHED_ERROR_PATH,
                f"Error path {inspect(path)} is not a valid response path.",
            )
            return

        positions: list[tuple[str | int, GraphQLType]] = []
        current_type: GraphQLType = self._root_type
        field_details_list: list[FieldDetails] | None = None
        data_value = self.data
        reason = None
        for segment in path:
            nullable_type = (
                current_type.of_type
                if is_non_null_type(current_type)
                else current_type
            )
            if is_list_type(nullable_type):
                if not isinstance(segment, int) or isinstance(segment, bool):
                    reason = (
                        f"expected a list index for type"
                        f" '{inspect(current_type)}', got {inspect(segment)}"
                    )
                    break
                if isinstance(data_value, list):
                    if not 0 <= segment < len(data_value):
                        reason = (
                            f"list index {segment} is out of range for the"
                            f" list of length {len(data_value)}"
                        )
                        break
                    data_value = data_value[segment]
                else:
                    data_value = Undefined
                current_type = nullable_type.of_type
                positions.append((segment, current_type))
                continue
            if is_object_type(nullable_type) or is_abstract_type(nullable_type):
                if not isinstance(segment, str):
                    reason = (
                        f"expected a response key for type"
                        f" '{inspect(nullable_type)}', got index {segment}"
                    )
                    break
                found = self._find_field(
                    nullable_type, segment, field_details_list
                )
                if found is None:
                    reason = (
                        f"no field with response key '{segment}' is selected"
                        f" on type '{inspect(nullable_type)}'"
                    )
                    break
                field_def, field_details_list = found
                current_type = field_def.type
                positions.append((segment, current_type))
                data_value = (
                    data_value.get(segment, Undefined)
                    if isinstance(data_value, Mapping)
                    else Undefined
                )
                continue
            reason = (
                f"cannot select {inspect(segment)} into leaf type"
                f" '{inspect(current_type)}'"
            )
            break

        if reason is not None:
            self._report(
                ResponseIssueKind.UNMATCHED_ERROR_PATH,
                f"Error path {list(path)!r} does not match the request:"
                f" {reason}.",
                tuple(path),
            )
            return

        # The error positions the null value at the errored field. While the
        # type at that position is non-null, the null propagates to the
        # parent position. The final position is where the data may be null.
        index = len(positions) - 1
        while index >= 0 and is_non_null_type(positions[index][1]):
            index -= 1
        null_path = tuple(segment for segment, _ in positions[: index + 1])
        self.allowed_null_positions.add(null_path)

    # -- data validation -----------------------------------------------

    def _visit_object(
        self,
        value: Mapping[str, Any],
        runtime_type: GraphQLObjectType,
        field_details_list: list[FieldDetails] | None,
        path: tuple[str | int, ...],
    ) -> None:
        grouped = self._collect_grouped_fields(runtime_type, field_details_list)
        for key, details in grouped.items():
            field_def = self._get_field_def(runtime_type, details[0].node.name.value)
            if field_def is not None:
                self._visit_object_entry(value, key, field_def, details, path)
        for key in value:
            if key not in grouped and key != "__typename":
                self._report(
                    ResponseIssueKind.UNEXPECTED_FIELD,
                    f"Unexpected response key '{key}' at {_format_path(path)}.",
                    path + (key,),
                )
        self._check_typename_value(value, runtime_type, path)

    def _visit_object_entry(
        self,
        value: Mapping[str, Any],
        key: str,
        field_def: GraphQLField,
        details: list[FieldDetails],
        path: tuple[str | int, ...],
    ) -> None:
        field_path = path + (key,)
        if key not in value:
            if field_path not in self.allowed_null_positions:
                self._report(
                    ResponseIssueKind.MISSING_FIELD,
                    f"Missing response key '{key}' at {_format_path(path)}.",
                    field_path,
                )
            return
        self._visit_value(value[key], field_def.type, details, field_path)

    def _check_typename_value(
        self,
        value: Mapping[str, Any],
        runtime_type: GraphQLObjectType,
        path: tuple[str | int, ...],
    ) -> None:
        if "__typename" not in value or value["__typename"] is None:
            return
        typename = value["__typename"]
        if not isinstance(typename, str) or typename != runtime_type.name:
            self._report(
                ResponseIssueKind.INVALID_TYPENAME,
                f"Invalid __typename {inspect(typename)} at"
                f" {_format_path(path)}; expected '{runtime_type.name}'.",
                path + ("__typename",),
            )

    def _visit_value(
        self,
        value: Any,
        type_: GraphQLType,
        field_details_list: list[FieldDetails],
        path: tuple[str | int, ...],
    ) -> None:
        if is_non_null_type(type_):
            if value is None:
                if path not in self.allowed_null_positions:
                    self._report(
                        ResponseIssueKind.UNEXPECTED_NULL,
                        f"Unexpected null for non-null type"
                        f" '{inspect(type_)}' at {_format_path(path)}.",
                        path,
                    )
                return
            self._visit_value(value, type_.of_type, field_details_list, path)
            return
        if value is None:
            # An explicit null is always allowed at a nullable position.
            return
        if path in self.allowed_null_positions:
            self._report(
                ResponseIssueKind.EXPECTED_NULL,
                f"Expected null at {_format_path(path)} because an error was"
                f" reported for this position.",
                path,
            )
        if is_list_type(type_):
            if not isinstance(value, list):
                self._report(
                    ResponseIssueKind.EXPECTED_LIST,
                    f"Expected a list for type '{inspect(type_)}' at"
                    f" {_format_path(path)}, got {inspect(value)}.",
                    path,
                )
                return
            item_type = type_.of_type
            for index, item in enumerate(value):
                self._visit_value(
                    item, item_type, field_details_list, path + (index,)
                )
            return
        if isinstance(value, list):
            self._report(
                ResponseIssueKind.UNEXPECTED_LIST,
                f"Unexpected list for non-list type '{inspect(type_)}' at"
                f" {_format_path(path)}.",
                path,
            )
            return
        if is_object_type(type_) or is_abstract_type(type_):
            if not isinstance(value, Mapping):
                self._report(
                    ResponseIssueKind.EXPECTED_OBJECT,
                    f"Expected an object for type '{inspect(type_)}' at"
                    f" {_format_path(path)}, got {inspect(value)}.",
                    path,
                )
                return
            if is_object_type(type_):
                self._visit_object(value, type_, field_details_list, path)
            else:
                self._visit_abstract(value, type_, field_details_list, path)
            return
        self._visit_leaf(value, type_, path)

    def _visit_leaf(
        self, value: Any, type_: GraphQLType, path: tuple[str | int, ...]
    ) -> None:
        if is_enum_type(type_):
            if not isinstance(value, str) or value not in type_.values:
                self._report(
                    ResponseIssueKind.INVALID_ENUM,
                    f"Invalid value {inspect(value)} for enum"
                    f" '{type_.name}' at {_format_path(path)}.",
                    path,
                )
            return
        if not is_scalar_type(type_):
            return
        if is_specified_scalar_type(type_):
            valid = _is_valid_specified_scalar(type_, value)
        else:
            checker = self.scalar_checkers.get(type_.name)
            if checker is None:
                # Custom scalars cannot be verified without a checker
                # provided by the caller.
                return
            try:
                valid = bool(checker(value))
            except Exception:
                valid = False
        if not valid:
            self._report(
                ResponseIssueKind.INVALID_SCALAR,
                f"Invalid value {inspect(value)} for scalar"
                f" '{type_.name}' at {_format_path(path)}.",
                path,
            )

    # -- abstract types --------------------------------------------------

    def _visit_abstract(
        self,
        value: Mapping[str, Any],
        abstract_type: GraphQLAbstractType,
        field_details_list: list[FieldDetails],
        path: tuple[str | int, ...],
    ) -> None:
        typename = value.get("__typename")
        if isinstance(typename, str):
            runtime_type = self.schema.get_type(typename)
            if (
                not is_object_type(runtime_type)
                or not self.schema.is_sub_type(abstract_type, runtime_type)
            ):
                self._report(
                    ResponseIssueKind.INVALID_TYPENAME,
                    f"Invalid __typename {inspect(typename)} at"
                    f" {_format_path(path)}; not a possible type of"
                    f" '{abstract_type.name}'.",
                    path + ("__typename",),
                )
                return
            self._visit_object(value, runtime_type, field_details_list, path)
            return
        if "__typename" in value:
            self._report(
                ResponseIssueKind.INVALID_TYPENAME,
                f"Invalid __typename at {_format_path(path)};"
                f" expected a string.",
                path + ("__typename",),
            )
            return
        possible_types = list(self.schema.get_possible_types(abstract_type))
        names = ", ".join(type_.name for type_ in possible_types)
        self._report(
            ResponseIssueKind.UNDETERMINED_ABSTRACT_TYPE,
            f"Cannot determine the runtime type of '{abstract_type.name}' at"
            f" {_format_path(path)} without __typename; undetermined"
            f" branches: {names}.",
            path,
        )
        self._visit_abstract_undetermined(
            value, possible_types, field_details_list, path
        )

    def _visit_abstract_undetermined(
        self,
        value: Mapping[str, Any],
        possible_types: list[GraphQLObjectType],
        field_details_list: list[FieldDetails],
        path: tuple[str | int, ...],
    ) -> None:
        """Validate what can be validated without a runtime type.

        Only the response keys that are selected for every possible type are
        checked strictly. Keys that are specific to some of the branches are
        only checked if they are present, and missing keys of such branches
        are not reported, since the runtime type is unknown.
        """
        grouped_per_type = [
            (type_, self._collect_grouped_fields(type_, field_details_list))
            for type_ in possible_types
        ]
        key_sets = [set(grouped) for _, grouped in grouped_per_type]
        union_keys = set().union(*key_sets) if key_sets else set()
        common_keys = set.intersection(*key_sets) if key_sets else set()
        for key in sorted(common_keys):
            runtime_type, grouped = grouped_per_type[0]
            details = grouped[key]
            field_def = self._get_field_def(runtime_type, details[0].node.name.value)
            if field_def is not None:
                self._visit_object_entry(value, key, field_def, details, path)
        for key in sorted(union_keys - common_keys):
            if key not in value:
                continue
            for runtime_type, grouped in grouped_per_type:
                details = grouped.get(key)
                if details:
                    field_def = self._get_field_def(
                        runtime_type, details[0].node.name.value
                    )
                    if field_def is not None:
                        self._visit_object_entry(
                            value, key, field_def, details, path
                        )
                    break
        for key in value:
            if key not in union_keys and key != "__typename":
                self._report(
                    ResponseIssueKind.UNEXPECTED_FIELD,
                    f"Unexpected response key '{key}' at {_format_path(path)}.",
                    path + (key,),
                )
