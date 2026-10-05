"""Pruning of GraphQL schemas to the needs of a set of operations.

This module provides :func:`prune_schema`, which takes a schema and a set of
already validated operations and produces a smaller, independently usable schema
that only contains what those operations need:

* the root fields selected by the operations and their selected sub-fields,
* the arguments actually used as well as all required arguments and input
  fields (the full input contract is preserved, independent of which fields
  happen to appear in the sample variable values),
* the directives used by the operations (plus the standard ``@include`` and
  ``@skip`` directives),
* and all type dependencies of those elements, including interface
  implementation relationships, union memberships, recursive types and types
  referenced by default values.

The pruned schema is guaranteed to be valid: it never contains empty object,
interface or input object types, empty union types, dangling type references or
broken interface implementations. Whenever extra elements have to be retained
beyond what the operations directly use to guarantee this, the accompanying
:class:`PruningReport` explains why.

The tool only performs static analysis. It does not run any resolvers and it
does not change the executor; field and type resolver functions are copied to
the pruned schema so that the operations remain executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..language import (
    DocumentNode,
    FragmentDefinitionNode,
    ListValueNode,
    ObjectValueNode,
    OperationDefinitionNode,
    OperationType,
    Visitor,
    parse,
)
from ..language.visitor import visit
from ..pyutils import Undefined
from ..type import (
    GraphQLArgument,
    GraphQLDirective,
    GraphQLField,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNamedType,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLUnionType,
    is_composite_type,
    is_input_object_type,
    is_interface_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
    is_required_argument,
    is_required_input_field,
    is_union_type,
)
from ..type.definition import get_named_type, get_nullable_type
from .type_from_ast import type_from_ast
from .type_info import TypeInfo, TypeInfoVisitor

__all__ = ["PrunedSchema", "PruningReport", "prune_schema"]

ROOT_OPERATION_TYPES = {
    OperationType.QUERY: "query_type",
    OperationType.MUTATION: "mutation_type",
    OperationType.SUBSCRIPTION: "subscription_type",
}

STANDARD_DIRECTIVES = ("include", "skip")

MarkInputField = Callable[[str, str], None]
MarkType = Callable[[Any], None]


@dataclass
class PruningReport:
    """Report describing how a schema was pruned.

    All dictionaries map type (or directive) names to the names of the elements
    that were removed. ``notes`` explains any elements that had to be retained
    beyond what the operations directly use in order to keep the pruned schema
    valid and self-contained.

    :param removed_root_fields: removed fields of the root operation types,
        keyed by root type name
    :param removed_fields: removed fields of other object and interface types,
        keyed by type name
    :param removed_arguments: removed field arguments, keyed by type name; each
        entry is formatted as ``fieldName(argName)``
    :param removed_input_fields: removed input object fields, keyed by input
        object type name
    :param removed_union_members: removed union member types, keyed by union
        type name
    :param removed_types: names of all removed named types
    :param removed_directives: names of all removed executable directives
    :param notes: explanations of additional elements retained to keep the
        pruned schema valid

    >>> from graphql import PruningReport
    >>> report = PruningReport()
    >>> report.removed_types
    []
    >>> report.removed_type_count
    0
    """

    removed_root_fields: dict[str, list[str]] = field(default_factory=dict)
    """Removed fields of the root operation types, keyed by root type name."""

    removed_fields: dict[str, list[str]] = field(default_factory=dict)
    """Removed fields of other object and interface types."""

    removed_arguments: dict[str, list[str]] = field(default_factory=dict)
    """Removed field arguments as ``fieldName(argName)``, keyed by type name."""

    removed_input_fields: dict[str, list[str]] = field(default_factory=dict)
    """Removed input object fields, keyed by input object type name."""

    removed_union_members: dict[str, list[str]] = field(default_factory=dict)
    """Removed union member types, keyed by union type name."""

    removed_types: list[str] = field(default_factory=list)
    """Names of all removed named types."""

    removed_directives: list[str] = field(default_factory=list)
    """Names of all removed executable directives."""

    notes: list[str] = field(default_factory=list)
    """Explanations of additional elements retained to keep the schema valid."""

    @property
    def removed_type_count(self) -> int:
        """Number of named types that were removed."""
        return len(self.removed_types)


@dataclass
class PrunedSchema:
    """The result of pruning a schema.

    :param schema: the pruned schema
    :param report: the pruning report

    >>> from graphql import PrunedSchema, PruningReport
    >>> result = PrunedSchema(schema=None, report=PruningReport())
    >>> result.report.removed_type_count
    0
    """

    schema: GraphQLSchema
    """The pruned schema."""

    report: PruningReport
    """The pruning report."""


def prune_schema(
    schema: GraphQLSchema,
    operations: str | DocumentNode | Iterable[str | DocumentNode],
    variable_values: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
    *,
    validate_operations: bool = True,
) -> PrunedSchema:
    """Prune a schema to the parts used by a set of validated operations.

    :param schema: the source schema
    :param operations: one operation document (as SDL string or parsed document)
        or an iterable of operation documents
    :param variable_values: sample variable values, either a single dictionary
        shared by all documents or an iterable of dictionaries aligned with the
        documents. Used to detect optional input fields that are actually used;
        required input fields are always retained regardless of the samples.
    :param validate_operations: whether the operations shall be validated against
        the source schema before pruning (enabled by default)
    :returns: a :class:`PrunedSchema` containing the pruned, independently
        usable schema and a pruning report

    >>> from graphql import build_schema, print_schema
    >>> source = build_schema('''
    ...     type User { id: ID! name: String email: String }
    ...     type Query { user(id: ID!): User unusedRoot: Boolean }
    ... ''')
    >>> operation = '{ user(id: "4") { id name } }'
    >>> result = prune_schema(source, operation)
    >>> "unusedRoot" in print_schema(result.schema)
    False
    >>> "email" in print_schema(result.schema)
    False
    >>> result.report.removed_root_fields
    {'Query': ['unusedRoot']}
    """
    documents = _coerce_documents(operations)
    if not documents:
        msg = "prune_schema requires at least one operation document."
        raise ValueError(msg)

    variables_per_document = _coerce_variables(variable_values, len(documents))

    if validate_operations:
        _validate_operation_documents(schema, documents)

    state = _collect_usage(schema, documents, variables_per_document)

    selection = _close_selection(schema, state)

    return _build_pruned_schema(schema, state, selection)


def _coerce_documents(
    operations: str | DocumentNode | Iterable[str | DocumentNode],
) -> list[DocumentNode]:
    if isinstance(operations, (str, DocumentNode)):
        raw_documents: Iterable[str | DocumentNode] = [operations]
    else:
        raw_documents = operations
    return [
        parse(operation) if isinstance(operation, str) else operation
        for operation in raw_documents
    ]


def _coerce_variables(
    variable_values: dict[str, Any] | Iterable[dict[str, Any]] | None,
    document_count: int,
) -> list[dict[str, Any]]:
    if variable_values is None:
        return [{} for _ in range(document_count)]
    if isinstance(variable_values, dict):
        return [dict(variable_values) for _ in range(document_count)]
    variables = [dict(values) for values in variable_values]
    if len(variables) != document_count:
        msg = (
            "The number of variable value dictionaries must match the number"
            " of operation documents."
        )
        raise ValueError(msg)
    return variables


def _validate_operation_documents(
    schema: GraphQLSchema, documents: list[DocumentNode]
) -> None:
    # Imported lazily to avoid an import cycle between the packages.
    from .. import validate
    from ..validation import specified_rules

    errors = []
    for document in documents:
        errors.extend(validate(schema, document, specified_rules))
    if errors:
        messages = "\n".join(f"- {error.message}" for error in errors)
        msg = f"Cannot prune schema with invalid operations:\n{messages}"
        raise ValueError(msg)


class _UsageState:
    """Usage information collected from the operation documents."""

    def __init__(self) -> None:
        self.root_operations: set[OperationType] = set()
        # Object/interface fields selected by the operations.
        self.used_fields: set[tuple[str, str]] = set()
        # Arguments used on object/interface fields.
        self.used_field_arguments: set[tuple[str, str, str]] = set()
        # Arguments used on directives, keyed by directive name.
        self.used_directive_arguments: dict[str, set[str]] = {}
        # Optional input fields seen in literals or sample variable values.
        self.used_input_fields: dict[str, set[str]] = {}
        # Directives applied by the operations.
        self.used_directives: set[str] = set()
        # Every named type referenced from a type position in the operations.
        self.referenced_types: set[GraphQLNamedType] = set()
        # Input object types referenced by a variable (or fragment variable)
        # definition. Variables are opaque to validation, so these input types
        # keep their complete field contract.
        self.variable_input_types: set[str] = set()
        # Required interface implementation edges:
        # (name of implementing type, name of interface).
        self.required_interfaces: set[tuple[str, str]] = set()
        # Required union memberships: (name of union, name of member).
        self.required_union_members: set[tuple[str, str]] = set()
        # Interface fragments applied directly to a union: (union, interface).
        self.required_union_interfaces: set[tuple[str, str]] = set()

    def mark_input_field(self, input_type_name: str, field_name: str) -> None:
        self.used_input_fields.setdefault(input_type_name, set()).add(field_name)

    def mark_type_reference(self, type_: Any) -> None:
        named_type = get_named_type(type_)
        if named_type is not None:
            self.referenced_types.add(named_type)


class _OperationCollector(Visitor):
    """Visitor recording the schema elements used by one document.

    Must be wrapped in a :class:`TypeInfoVisitor` when traversing a document.
    """

    def __init__(
        self,
        type_info: TypeInfo,
        state: _UsageState,
        fragment_conditions: dict[str, GraphQLNamedType],
        variables: dict[str, Any],
    ) -> None:
        super().__init__()
        self.type_info = type_info
        self.state = state
        self.fragment_conditions = fragment_conditions
        self.variables = variables
        # Composite type of the current selection context.
        self.composite_stack: list[GraphQLNamedType] = []
        # For each entered field, whether it returned a composite type and the
        # name of the field (for argument tracking).
        self.field_pushes: list[bool] = []
        self.field_name_stack: list[str] = []

    def enter_operation_definition(
        self, node: OperationDefinitionNode, *_args: Any
    ) -> None:
        root_type = get_named_type(self.type_info.get_type())
        if root_type is not None:
            self.composite_stack.append(root_type)
            self.state.referenced_types.add(root_type)
        self.state.root_operations.add(node.operation)

    def leave_operation_definition(self, *_args: Any) -> None:
        if self.composite_stack:
            self.composite_stack.pop()

    def enter_fragment_definition(
        self, node: FragmentDefinitionNode, *_args: Any
    ) -> None:
        condition = self.fragment_conditions[node.name.value]
        self.composite_stack.append(condition)
        self.state.referenced_types.add(condition)

    def leave_fragment_definition(self, *_args: Any) -> None:
        if self.composite_stack:
            self.composite_stack.pop()

    def enter_inline_fragment(self, node, *_args: Any) -> None:
        condition = get_named_type(self.type_info.get_type())
        if condition is not None:
            self._record_type_condition(condition)
            self.composite_stack.append(condition)

    def leave_inline_fragment(self, *_args: Any) -> None:
        if self.composite_stack:
            self.composite_stack.pop()

    def enter_fragment_spread(self, node, *_args: Any) -> None:
        condition = self.fragment_conditions.get(node.name.value)
        if condition is not None:
            self._record_type_condition(condition)

    def _record_type_condition(self, condition: GraphQLNamedType) -> None:
        self.state.referenced_types.add(condition)
        enclosing = self.composite_stack[-1] if self.composite_stack else None
        if enclosing is None or condition is enclosing:
            return
        self.state.referenced_types.add(enclosing)
        if is_union_type(enclosing) and is_object_type(condition):
            self.state.required_union_members.add((enclosing.name, condition.name))
        elif is_union_type(enclosing) and is_interface_type(condition):
            self.state.required_union_interfaces.add(
                (enclosing.name, condition.name)
            )
        elif is_interface_type(enclosing) and (
            is_object_type(condition) or is_interface_type(condition)
        ):
            self.state.required_interfaces.add((condition.name, enclosing.name))

    def enter_field(self, node, *_args: Any) -> None:
        parent_type = self.type_info.get_parent_type()
        field_def = self.type_info.get_field_def()
        field_name = node.name.value
        is_composite_field = False
        if (
            parent_type is not None
            and field_def is not None
            and not field_name.startswith("__")
        ):
            self.state.used_fields.add((parent_type.name, field_name))
            self.state.referenced_types.add(parent_type)
            self.state.mark_type_reference(field_def.type)
            is_composite_field = (
            not field_name.startswith("__")
            and is_composite_type(get_named_type(field_def.type))
        )
            if is_composite_field:
                self.composite_stack.append(get_named_type(field_def.type))
        self.field_pushes.append(is_composite_field)
        self.field_name_stack.append(field_name)

    def leave_field(self, *_args: Any) -> None:
        if self.field_pushes.pop():
            if self.composite_stack:
                self.composite_stack.pop()
        self.field_name_stack.pop()

    def enter_argument(self, node, *_args: Any) -> None:
        directive = self.type_info.get_directive()
        if directive is not None:
            self.state.used_directive_arguments.setdefault(
                directive.name, set()
            ).add(node.name.value)
        else:
            parent_type = self.type_info.get_parent_type()
            if parent_type is not None and self.field_name_stack:
                self.state.used_field_arguments.add(
                    (parent_type.name, self.field_name_stack[-1], node.name.value)
                )
        argument_def = self.type_info.get_argument()
        if argument_def is not None:
            self.state.mark_type_reference(argument_def.type)

    def enter_object_field(self, node, *_args: Any) -> None:
        input_type = get_named_type(self.type_info.get_parent_input_type())
        if is_input_object_type(input_type):
            self.state.mark_input_field(input_type.name, node.name.value)
            field_def = input_type.fields.get(node.name.value)
            if field_def is not None:
                self.state.mark_type_reference(field_def.type)

    def enter_directive(self, node, *_args: Any) -> None:
        self.state.used_directives.add(node.name.value)
        directive = self.type_info.get_directive()
        if directive is not None:
            self.state.referenced_types.update(
                get_named_type(argument.type)
                for argument in directive.args.values()
            )

    def enter_variable_definition(self, node, *_args: Any) -> None:
        variable_type = self.type_info.get_input_type()
        if variable_type is None:
            return
        self.state.mark_type_reference(variable_type)
        named_type = get_named_type(variable_type)
        if is_input_object_type(named_type):
            self.state.variable_input_types.add(named_type.name)
        if node.default_value is not None:
            _walk_input_literal(
                node.default_value, variable_type, self.state.mark_input_field
            )
        variable_name = node.variable.name.value
        sample_value = self.variables.get(variable_name)
        if sample_value is not None:
            _walk_runtime_value(
                sample_value, variable_type, self.state.mark_input_field
            )


def _collect_usage(
    schema: GraphQLSchema,
    documents: list[DocumentNode],
    variables_per_document: list[dict[str, Any]],
) -> _UsageState:
    state = _UsageState()

    for document, variables in zip(documents, variables_per_document):
        fragment_conditions: dict[str, GraphQLNamedType] = {}
        for definition in document.definitions:
            if isinstance(definition, FragmentDefinitionNode):
                fragment_conditions[definition.name.value] = type_from_ast(
                    schema, definition.type_condition
                )

        type_info = TypeInfo(schema)
        collector = _OperationCollector(
            type_info, state, fragment_conditions, variables
        )
        visit(document, TypeInfoVisitor(type_info, collector))

    # @include and @skip belong to every executable schema.
    for directive in schema.directives:
        if directive.name in STANDARD_DIRECTIVES:
            state.used_directives.add(directive.name)

    return state


def _walk_runtime_value(
    value: Any, type_: Any, mark: MarkInputField
) -> None:
    """Record optional input fields used by a Python variable value."""
    named_type = get_named_type(type_)
    if is_input_object_type(named_type):
        if isinstance(value, dict):
            for field_name, field_value in value.items():
                if field_name not in named_type.fields:
                    continue
                mark(named_type.name, field_name)
                _walk_runtime_value(
                    field_value, named_type.fields[field_name].type, mark
                )
    elif is_list_type(get_nullable_type(type_)) and isinstance(value, (list, tuple)):
        item_type = get_nullable_type(type_).of_type
        for item in value:
            if item is not None:
                _walk_runtime_value(item, item_type, mark)


def _walk_input_literal(
    node: Any, type_: Any, mark: MarkInputField
) -> None:
    """Record optional input fields used by a GraphQL literal value."""
    named_type = get_named_type(type_)
    if is_input_object_type(named_type):
        if isinstance(node, ObjectValueNode):
            for field_node in node.fields:
                field_name = field_node.name.value
                if field_name not in named_type.fields:
                    continue
                mark(named_type.name, field_name)
                _walk_input_literal(
                    field_node.value, named_type.fields[field_name].type, mark
                )
    elif is_list_type(get_nullable_type(type_)) and isinstance(node, ListValueNode):
        item_type = get_nullable_type(type_).of_type
        for item_node in node.values:
            _walk_input_literal(item_node, item_type, mark)




@dataclass
class _Selection:
    """Final decision about which schema elements are retained."""

    types: set[str]
    fields: dict[str, set[str]]
    arguments: dict[str, set[str]]
    input_fields: dict[str, set[str]]
    union_members: dict[str, set[str]]
    directives: set[str]
    notes: list[str]


def _close_selection(
    schema: GraphQLSchema, state: _UsageState
) -> _Selection:
    notes: list[str] = []

    kept_types: set[str] = {named_type.name for named_type in state.referenced_types}
    kept_fields: dict[str, set[str]] = {}
    kept_arguments: dict[str, set[str]] = {}
    kept_input_fields: dict[str, set[str]] = {}
    kept_union_members: dict[str, set[str]] = {}

    # Seed the fields selected on each composite type.
    for owner_name, field_name in state.used_fields:
        if not field_name.startswith("__"):
            kept_fields.setdefault(owner_name, set()).add(field_name)

    directives_in_progress = {
        directive.name
        for directive in schema.directives
        if directive.name in state.used_directives
    }

    worklist: list[Any] = [("type", name) for name in kept_types]
    for owner_name, field_set in kept_fields.items():
        for field_name in field_set:
            worklist.append(("field", owner_name, field_name))
    done: set[Any] = set()

    def enqueue(item: Any) -> None:
        if item not in done:
            worklist.append(item)

    def keep_type(named_type: Any) -> None:
        if named_type is not None and named_type.name not in kept_types:
            kept_types.add(named_type.name)
            enqueue(("type", named_type.name))

    def keep_field(type_name: str, field_name: str) -> None:
        field_set = kept_fields.setdefault(type_name, set())
        if not field_name.startswith("__") and field_name not in field_set:
            field_set.add(field_name)
            enqueue(("field", type_name, field_name))

    def keep_argument(type_name: str, field_name: str, arg_name: str) -> None:
        key = f"{type_name}.{field_name}"
        arg_set = kept_arguments.setdefault(key, set())
        if arg_name not in arg_set:
            arg_set.add(arg_name)
            enqueue(("argument", type_name, field_name, arg_name))

    def keep_input_field(type_name: str, field_name: str) -> None:
        field_set = kept_input_fields.setdefault(type_name, set())
        if field_name not in field_set:
            field_set.add(field_name)
            enqueue(("input_field", type_name, field_name))

    while True:
        while worklist:
            item = worklist.pop()
            if item in done:
                continue
            done.add(item)
            kind = item[0]

            if kind == "type":
                _process_type(
                    schema,
                    state,
                    item[1],
                    keep_type,
                    keep_input_field,
                    kept_union_members,
                )
            elif kind == "field":
                _process_field(
                    schema, state, item[1], item[2], keep_type, keep_argument
                )
            elif kind == "argument":
                _process_argument(
                    schema,
                    state,
                    item[1],
                    item[2],
                    item[3],
                    keep_type,
                )
            elif kind == "input_field":
                _process_input_field(
                    schema, state, item[1], item[2], keep_type, keep_input_field
                )

        # Close over the argument types of every retained directive.
        for directive in schema.directives:
            if directive.name not in directives_in_progress:
                continue
            used_args = state.used_directive_arguments.get(directive.name, set())
            for arg_name, arg_def in directive.args.items():
                if is_required_argument(arg_def) or arg_name in used_args:
                    keep_type(get_named_type(arg_def.type))

        # Close interface implementation relationships: every retained
        # interface keeps its fields and arguments on every retained implementor,
        # and explicitly requested fragments force transitive edges to stay.
        # An interface fragment applied to a union is only possible on union
        # members that implement that interface; those members (and the
        # interface) must survive.
        for union_name, interface_name in state.required_union_interfaces:
            union_type = schema.get_type(union_name)
            interface_type = schema.get_type(interface_name)
            if not is_union_type(union_type) or not is_interface_type(
                interface_type
            ):
                continue
            if interface_name in kept_types:
                members = kept_union_members.setdefault(union_name, set())
                for member in union_type.types:
                    if any(
                        member_interface.name == interface_name
                        for member_interface in member.interfaces
                    ):
                        keep_type(member)
                        members.add(member.name)

        changed = _close_interface_edges(
            schema,
            state,
            kept_types,
            kept_fields,
            kept_arguments,
            keep_type,
            keep_field,
            keep_argument,
        )

        # Guarantee a valid schema: no empty objects, interfaces, input objects
        # or unions.
        changed |= _apply_backstops(
            schema,
            state,
            kept_types,
            kept_fields,
            kept_input_fields,
            kept_union_members,
            keep_type,
            keep_field,
            keep_input_field,
            enqueue,
            notes,
        )

        if not changed and not worklist:
            break

    directives = {
        directive.name
        for directive in schema.directives
        if directive.name in state.used_directives
    }

    return _Selection(
        types=kept_types,
        fields=kept_fields,
        arguments=kept_arguments,
        input_fields=kept_input_fields,
        union_members=kept_union_members,
        directives=directives,
        notes=notes,
    )



def _has_default(definition: Any) -> bool:
    return (
        definition.default is not None
        or definition.default_value is not Undefined
    )

def _process_type(
    schema: GraphQLSchema,
    state: _UsageState,
    type_name: str,
    keep_type: Callable,
    keep_input_field: Callable,
    kept_union_members: dict[str, set[str]],
) -> None:
    named_type = schema.get_type(type_name)
    if named_type is None:
        return

    if is_input_object_type(named_type):
        used = state.used_input_fields.get(type_name, set())
        full_contract = type_name in state.variable_input_types
        for input_name, input_def in named_type.fields.items():
            if (
                full_contract
                or is_required_input_field(input_def)
                or input_name in used
                or _has_default(input_def)
            ):
                keep_input_field(type_name, input_name)
    elif is_union_type(named_type):
        members = kept_union_members.setdefault(type_name, set())
        for union_name, member_name in state.required_union_members:
            if union_name == type_name:
                members.add(member_name)


def _process_field(
    schema: GraphQLSchema,
    state: _UsageState,
    type_name: str,
    field_name: str,
    keep_type: Callable,
    keep_argument: Callable,
) -> None:
    named_type = schema.get_type(type_name)
    if not (is_object_type(named_type) or is_interface_type(named_type)):
        return
    field_def = named_type.fields.get(field_name)
    if field_def is None:
        return

    keep_type(get_named_type(field_def.type))

    used_arg_names = {
        arg_name
        for owner, used_field, arg_name in state.used_field_arguments
        if owner == type_name and used_field == field_name
    }
    for arg_name, arg_def in field_def.args.items():
        if (
            is_required_argument(arg_def)
            or arg_name in used_arg_names
            or arg_def.default is not None
            or arg_def.default_value is not Undefined
        ):
            # Arguments with default values are part of the field contract:
            # the default value (and the types it references) must survive.
            keep_argument(type_name, field_name, arg_name)


def _process_argument(
    schema: GraphQLSchema,
    state: _UsageState,
    type_name: str,
    field_name: str,
    arg_name: str,
    keep_type: Callable,
) -> None:
    named_type = schema.get_type(type_name)
    if not (is_object_type(named_type) or is_interface_type(named_type)):
        return
    field_def = named_type.fields.get(field_name)
    if field_def is None:
        return
    arg_def = field_def.args.get(arg_name)
    if arg_def is None:
        return

    keep_type(get_named_type(arg_def.type))
    _mark_used_fields_in_default(arg_def, arg_def.type, state)


def _process_input_field(
    schema: GraphQLSchema,
    state: _UsageState,
    type_name: str,
    field_name: str,
    keep_type: Callable,
    keep_input_field: Callable,
) -> None:
    named_type = schema.get_type(type_name)
    if not is_input_object_type(named_type):
        return
    input_def = named_type.fields.get(field_name)
    if input_def is None:
        return

    keep_type(get_named_type(input_def.type))
    _mark_used_fields_in_default(input_def, input_def.type, state)

    nested_type = get_named_type(input_def.type)
    if is_input_object_type(nested_type):
        if type_name in state.variable_input_types:
            state.variable_input_types.add(nested_type.name)
        used = state.used_input_fields.get(nested_type.name, set())
        full_contract = nested_type.name in state.variable_input_types
        for nested_name, nested_def in nested_type.fields.items():
            if (
                full_contract
                or is_required_input_field(nested_def)
                or nested_name in used
                or _has_default(nested_def)
            ):
                keep_input_field(nested_type.name, nested_name)


def _mark_used_fields_in_default(
    definition: Any, type_: Any, state: _UsageState
) -> None:
    """Record input fields referenced by a default value (literal or runtime)."""
    if definition.default is not None:
        literal = definition.default.literal
        if literal is not None:
            _walk_input_literal(literal, type_, state.mark_input_field)
        elif definition.default.value is not Undefined:
            _walk_runtime_value(
                definition.default.value, type_, state.mark_input_field
            )
    elif definition.default_value is not Undefined:
        _walk_runtime_value(
            definition.default_value, type_, state.mark_input_field
        )


def _close_interface_edges(
    schema: GraphQLSchema,
    state: _UsageState,
    kept_types: set[str],
    kept_fields: dict[str, set[str]],
    kept_arguments: dict[str, set[str]],
    keep_type: Callable,
    keep_field: Callable,
    keep_argument: Callable,
) -> bool:
    changed = False

    # Explicitly requested implements edges (from interface/union fragments)
    # force every interface on a transitive implements path to be retained.
    for implementor_name, interface_name in list(state.required_interfaces):
        if implementor_name not in kept_types or interface_name not in kept_types:
            continue
        for ancestor_name in _transitive_interface_path(
            schema, kept_types, implementor_name, interface_name
        ):
            ancestor = schema.get_type(ancestor_name)
            if ancestor_name not in kept_types:
                keep_type(ancestor)
                changed = True

    # Every retained interface keeps its object implementors: the runtime
    # result of an interface field may be any of them, even without a fragment
    # selecting that concrete type.
    for type_name in list(kept_types):
        named_type = schema.get_type(type_name)
        if not is_interface_type(named_type):
            continue
        for implementor in schema.get_implementations(named_type).objects:
            if implementor.name not in kept_types:
                keep_type(implementor)
                changed = True

    # For every retained interface, propagate its retained fields and arguments
    # onto every retained object or interface that implements it.
    for type_name in list(kept_types):
        named_type = schema.get_type(type_name)
        if not (is_object_type(named_type) or is_interface_type(named_type)):
            continue
        for iface in named_type.interfaces:
            if iface.name not in kept_types:
                continue
            before = changed
            for field_name in kept_fields.get(iface.name, ()):
                keep_field(type_name, field_name)
                _sync_interface_arguments(
                    schema,
                    state,
                    kept_arguments,
                    type_name,
                    iface.name,
                    field_name,
                    keep_argument,
                )
            changed = before or changed

    return changed


def _sync_interface_arguments(
    schema: GraphQLSchema,
    state: _UsageState,
    kept_arguments: dict[str, set[str]],
    implementor_name: str,
    interface_name: str,
    field_name: str,
    keep_argument: Callable,
) -> None:
    # The arguments kept on the interface field must also be kept on every
    # implementing field, since implementing argument sets have to be
    # supersets with identical argument types.
    interface_args = kept_arguments.get(
        f"{interface_name}.{field_name}", set()
    )
    for arg_name in interface_args:
        keep_argument(implementor_name, field_name, arg_name)


def _transitive_interface_path(
    schema: GraphQLSchema,
    kept_types: set[str],
    implementor_name: str,
    interface_name: str,
) -> list[str]:
    """Find the interfaces on a retained transitive implements path."""
    implementor = schema.get_type(implementor_name)
    if implementor is None:
        return []
    queue: list[tuple[str, list[str]]] = []
    for iface in implementor.interfaces:
        if iface.name in kept_types:
            queue.append((iface.name, [iface.name]))
    visited = {implementor_name}
    while queue:
        current_name, path = queue.pop(0)
        if current_name in visited:
            continue
        visited.add(current_name)
        if current_name == interface_name:
            return path
        current_type = schema.get_type(current_name)
        if not is_interface_type(current_type):
            continue
        for iface in current_type.interfaces:
            queue.append((iface.name, [*path, iface.name]))
    return []


def _apply_backstops(
    schema: GraphQLSchema,
    state: _UsageState,
    kept_types: set[str],
    kept_fields: dict[str, set[str]],
    kept_input_fields: dict[str, set[str]],
    kept_union_members: dict[str, set[str]],
    keep_type: Callable,
    keep_field: Callable,
    keep_input_field: Callable,
    enqueue: Callable,
    notes: list[str],
) -> bool:
    changed = False

    for type_name in list(kept_types):
        named_type = schema.get_type(type_name)
        if named_type is None:
            continue

        if is_object_type(named_type) or is_interface_type(named_type):
            if not kept_fields.get(type_name):
                retained = set(named_type.fields)
                kept_fields[type_name] = retained
                for field_name in retained:
                    enqueue(("field", type_name, field_name))
                notes.append(
                    f"Kept all fields of '{type_name}': an object or interface"
                    " type must define at least one field."
                )
                changed = True

        elif is_input_object_type(named_type):
            if not kept_input_fields.get(type_name):
                retained = set(named_type.fields)
                kept_input_fields[type_name] = retained
                state.used_input_fields.setdefault(type_name, set()).update(retained)
                for field_name in retained:
                    enqueue(("input_field", type_name, field_name))
                notes.append(
                    f"Kept all input fields of '{type_name}': an input object"
                    " type must define at least one field."
                )
                changed = True

        elif is_union_type(named_type):
            members = kept_union_members.setdefault(type_name, set())
            surviving = [
                member.name
                for member in named_type.types
                if member.name in kept_types and member.name in members
            ]
            if not surviving:
                retained = [
                    member.name
                    for member in named_type.types
                    if member.name in kept_types
                ]
                added_new_types = False
                if not retained:
                    retained = [member.name for member in named_type.types]
                    for member in named_type.types:
                        if member.name not in kept_types:
                            keep_type(member)
                            added_new_types = True
                members.update(retained)
                notes.append(
                    f"Kept member(s) {', '.join(retained)} of union"
                    f" '{type_name}': a union type must define at least one"
                    " member type."
                )
                changed = True
                if added_new_types:
                    continue

    return changed


def _build_pruned_schema(
    schema: GraphQLSchema, state: _UsageState, selection: _Selection
) -> PrunedSchema:
    old_type_map = schema.type_map
    new_type_map: dict[str, GraphQLNamedType] = {}

    def remap_type(type_: Any) -> Any:
        if is_non_null_type(type_):
            return GraphQLNonNull(remap_type(type_.of_type))
        if is_list_type(type_):
            return GraphQLList(remap_type(type_.of_type))
        return new_type_map[type_.name]

    def filter_arguments(
        field: Any, type_name: str, field_name: str
    ) -> dict | None:
        retained_names = selection.arguments.get(f"{type_name}.{field_name}")
        if retained_names is None:
            return None
        args = {}
        used_directive_names = state.used_directive_arguments  # noqa: F841
        for arg_name, arg_def in field.args.items():
            if arg_name in retained_names:
                args[arg_name] = _copy_argument(arg_def, remap_type)
        return args

    def build_fields_for(
        named_type: Any, field_names: set[str]
    ) -> dict[str, GraphQLField]:
        fields = {}
        for field_name in field_names:
            field_def = named_type.fields.get(field_name)
            if field_def is None:
                continue
            fields[field_name] = GraphQLField(
                remap_type(field_def.type),
                args=filter_arguments(field_def, named_type.name, field_name),
                resolve=field_def.resolve,
                subscribe=field_def.subscribe,
                description=field_def.description,
                deprecation_reason=field_def.deprecation_reason,
                extensions=field_def.extensions,
                ast_node=field_def.ast_node,
            )
        return fields

    def build_input_fields_for(
        named_type: GraphQLInputObjectType, field_names: set[str]
    ) -> dict[str, GraphQLInputField]:
        fields = {}
        for field_name in field_names:
            input_def = named_type.fields.get(field_name)
            if input_def is None:
                continue
            kwargs = input_def.to_kwargs()
            kwargs["type_"] = remap_type(input_def.type)
            fields[field_name] = GraphQLInputField(**kwargs)
        return fields

    # First create every retained named type. Scalars and enums are reused as
    # they are leaf types without prunable structure; composite, input object
    # and union types are rebuilt with thunks so recursive references resolve
    # lazily after all types exist.
    ordered_types = [
        named_type
        for name, named_type in old_type_map.items()
        if name in selection.types and not name.startswith("__")
    ]

    for named_type in ordered_types:
        if is_object_type(named_type):
            new_type = GraphQLObjectType(
                name=named_type.name,
                fields=lambda nt=named_type: build_fields_for(
                    nt, _field_names_for(selection, nt)
                ),
                interfaces=lambda nt=named_type: tuple(
                    new_type_map[iface.name]
                    for iface in nt.interfaces
                    if iface.name in selection.types
                ),
                is_type_of=named_type.is_type_of,
                description=named_type.description,
                extensions=named_type.extensions,
                ast_node=named_type.ast_node,
                extension_ast_nodes=named_type.extension_ast_nodes,
            )
            new_type_map[named_type.name] = new_type
        elif is_interface_type(named_type):
            new_type = GraphQLInterfaceType(
                name=named_type.name,
                fields=lambda nt=named_type: build_fields_for(
                    nt, _field_names_for(selection, nt)
                ),
                interfaces=lambda nt=named_type: tuple(
                    new_type_map[iface.name]
                    for iface in nt.interfaces
                    if iface.name in selection.types
                ),
                resolve_type=named_type.resolve_type,
                description=named_type.description,
                extensions=named_type.extensions,
                ast_node=named_type.ast_node,
                extension_ast_nodes=named_type.extension_ast_nodes,
            )
            new_type_map[named_type.name] = new_type
        elif is_input_object_type(named_type):
            field_names = selection.input_fields.get(named_type.name)
            new_type = GraphQLInputObjectType(
                name=named_type.name,
                fields=lambda nt=named_type, fn=field_names: (
                    build_input_fields_for(nt, fn or set(nt.fields))
                ),
                description=named_type.description,
                out_type=getattr(named_type, "out_type", None),
                extensions=named_type.extensions,
                ast_node=named_type.ast_node,
                extension_ast_nodes=named_type.extension_ast_nodes,
                is_one_of=named_type.is_one_of,
            )
            new_type_map[named_type.name] = new_type
        elif is_union_type(named_type):
            member_names = selection.union_members.get(named_type.name, set())

            def build_members(nt=named_type, members=member_names):
                names = members or {m.name for m in nt.types}
                return tuple(
                    new_type_map[member.name]
                    for member in nt.types
                    if member.name in names and member.name in new_type_map
                )

            new_type = GraphQLUnionType(
                name=named_type.name,
                types=build_members,
                resolve_type=named_type.resolve_type,
                description=named_type.description,
                extensions=named_type.extensions,
                ast_node=named_type.ast_node,
                extension_ast_nodes=named_type.extension_ast_nodes,
            )
            new_type_map[named_type.name] = new_type
        else:
            # Scalars and enums are immutable leaves and safe to reuse.
            new_type_map[named_type.name] = named_type

    directives = _build_directives(schema, state, selection, remap_type)

    roots = {}
    for operation, attr in ROOT_OPERATION_TYPES.items():
        root_type = getattr(schema, attr)
        if root_type is not None and operation in state.root_operations:
            roots[attr] = new_type_map[root_type.name]
        else:
            roots[attr] = None

    new_schema = GraphQLSchema(
        query=roots["query_type"],
        mutation=roots["mutation_type"],
        subscription=roots["subscription_type"],
        types=tuple(new_type_map.values()),
        directives=tuple(directives),
        description=schema.description,
        extensions=schema.extensions,
        ast_node=schema.ast_node,
        extension_ast_nodes=schema.extension_ast_nodes,
    )

    report = _build_report(schema, state, selection)
    return PrunedSchema(schema=new_schema, report=report)


def _field_names_for(selection: _Selection, named_type: Any) -> set[str]:
    return selection.fields.get(named_type.name, set())


def _copy_argument(arg_def: Any, remap_type: Callable) -> GraphQLArgument:
    kwargs = arg_def.to_kwargs()
    kwargs["type_"] = remap_type(arg_def.type)
    return GraphQLArgument(**kwargs)


def _build_directives(
    schema: GraphQLSchema,
    state: _UsageState,
    selection: _Selection,
    remap_type: Callable,
) -> list[GraphQLDirective]:
    directives = []
    for directive in schema.directives:
        if directive.name not in selection.directives:
            continue
        used_args = state.used_directive_arguments.get(directive.name, set())
        args = {}
        for arg_name, arg_def in directive.args.items():
            if is_required_argument(arg_def) or arg_name in used_args:
                args[arg_name] = _copy_argument(arg_def, remap_type)
        directives.append(
            GraphQLDirective(
                name=directive.name,
                locations=directive.locations,
                args=args,
                is_repeatable=directive.is_repeatable,
                description=directive.description,
                extensions=directive.extensions,
                ast_node=directive.ast_node,
                extension_ast_nodes=directive.extension_ast_nodes,
            )
        )
    return directives


def _build_report(
    schema: GraphQLSchema, state: _UsageState, selection: _Selection
) -> PruningReport:
    report = PruningReport()

    root_names = {
        root.name
        for root in (
            schema.query_type,
            schema.mutation_type,
            schema.subscription_type,
        )
        if root is not None
    }

    for type_name in selection.types:
        named_type = schema.get_type(type_name)
        if named_type is None:
            continue

        if is_object_type(named_type) or is_interface_type(named_type):
            retained_fields = selection.fields.get(type_name, set())
            removed_field_names = [
                field_name
                for field_name in named_type.fields
                if field_name not in retained_fields
            ]
            if removed_field_names:
                target = (
                    report.removed_root_fields
                    if type_name in root_names
                    else report.removed_fields
                )
                target[type_name] = removed_field_names

            removed_args: list[str] = []
            for field_name in retained_fields:
                field_def = named_type.fields.get(field_name)
                if field_def is None:
                    continue
                retained_args = selection.arguments.get(
                    f"{type_name}.{field_name}", set()
                )
                for arg_name in field_def.args:
                    if arg_name not in retained_args:
                        removed_args.append(f"{field_name}({arg_name})")
            if removed_args:
                report.removed_arguments[type_name] = removed_args

        elif is_input_object_type(named_type):
            retained = selection.input_fields.get(type_name, set())
            removed = [
                field_name
                for field_name in named_type.fields
                if field_name not in retained
            ]
            if removed:
                report.removed_input_fields[type_name] = removed

        elif is_union_type(named_type):
            retained = selection.union_members.get(type_name, set())
            removed = [
                member.name
                for member in named_type.types
                if member.name not in retained
            ]
            if removed:
                report.removed_union_members[type_name] = removed

    report.removed_types = [
        name
        for name in schema.type_map
        if not name.startswith("__") and name not in selection.types
    ]
    # @deprecated, @specifiedBy and @oneOf are type-system directives that are
    # applied during schema construction rather than by operations, so they are
    # not part of the executable directive set and are not reported as removed.
    type_system_directives = {"deprecated", "specifiedBy", "oneOf"}
    report.removed_directives = [
        directive.name
        for directive in schema.directives
        if directive.name not in selection.directives
        and directive.name not in type_system_directives
    ]
    report.notes = list(selection.notes)
    return report
