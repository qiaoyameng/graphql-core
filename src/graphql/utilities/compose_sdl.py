"""Compose multiple SDL documents from different sources into a single schema.

When several teams each deliver a fragment of the schema in the GraphQL
schema definition language (SDL), naively concatenating the documents fails
as soon as two fragments declare the same type. The :func:`compose_sdl`
function instead merges the fragments semantically:

* Fields, enum values, union members, interfaces, input fields and directive
  applications that do not conflict are merged into a single declaration.
* Declarations that are repeated with identical semantics are kept only once.
  Comparison of arguments, wrapped types, default values, enum values and
  directive applications is structural, not textual.
* Declarations that only differ in their description text are not treated as
  conflicts; they are reported separately as description differences.
* The order in which definitions and extensions appear (``type`` vs.
  ``extend type``) does not affect the result.
* Genuinely conflicting declarations are reported with a schema coordinate
  and the source name, line and column of *both* sides, and no partial
  schema is returned.
* The merged output is checked with the regular schema validation, and
  interfaces, union memberships and root operation type specifications are
  preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Mapping, NamedTuple

from ..error import GraphQLError
from ..language import (
    ConstDirectiveNode,
    DirectiveDefinitionNode,
    DocumentNode,
    EnumTypeDefinitionNode,
    EnumTypeExtensionNode,
    EnumValueDefinitionNode,
    FieldDefinitionNode,
    InputObjectTypeDefinitionNode,
    InputObjectTypeExtensionNode,
    InputValueDefinitionNode,
    InterfaceTypeDefinitionNode,
    InterfaceTypeExtensionNode,
    ListTypeNode,
    NamedTypeNode,
    NameNode,
    Node,
    NonNullTypeNode,
    ObjectTypeDefinitionNode,
    ObjectTypeExtensionNode,
    OperationTypeDefinitionNode,
    ScalarTypeDefinitionNode,
    ScalarTypeExtensionNode,
    SchemaDefinitionNode,
    SchemaExtensionNode,
    Source,
    StringValueNode,
    TypeNode,
    UnionTypeDefinitionNode,
    UnionTypeExtensionNode,
    ValueNode,
    get_location,
    parse,
    print_ast,
)
from ..type import GraphQLSchema, validate_schema
from .build_ast_schema import build_ast_schema

__all__ = [
    "SdlCompositionResult",
    "SdlConflict",
    "SdlDescriptionDifference",
    "SdlSourceLocation",
    "compose_sdl",
]


class SdlSourceLocation(NamedTuple):
    """A pointer back into one of the composed SDL source documents."""

    source: str
    """The name of the source document, as passed to :func:`compose_sdl`."""

    line: int
    """The one-indexed line within the source document."""

    column: int
    """The one-indexed column within the source document."""

    def __str__(self) -> str:
        return f"{self.source}:{self.line}:{self.column}"


@dataclass(frozen=True)
class SdlConflict:
    """A semantic conflict between two SDL declarations.

    :param coordinate: the schema coordinate of the conflicting element
    :param message: a description of the conflict
    :param first: the location of the first declaration
    :param second: the location of the conflicting declaration

    >>> from graphql.utilities import SdlConflict, SdlSourceLocation
    >>> loc = SdlSourceLocation("a.graphql", 1, 7)
    >>> print(SdlConflict("Query.x", "types differ", loc, loc))
    Conflicting definitions of Query.x: types differ
      first:  a.graphql:1:7
      second: a.graphql:1:7
    """

    coordinate: str
    """The schema coordinate of the conflicting element."""

    message: str
    """A description of the conflict."""

    first: SdlSourceLocation
    """The location of the first declaration."""

    second: SdlSourceLocation
    """The location of the conflicting declaration."""

    def __str__(self) -> str:
        return (
            f"Conflicting definitions of {self.coordinate}: {self.message}\n"
            f"  first:  {self.first}\n"
            f"  second: {self.second}"
        )


@dataclass(frozen=True)
class SdlDescriptionDifference:
    """Two declarations that only differ in their description text.

    Description differences are not conflicts: the first description is kept
    and the difference is merely reported.

    :param coordinate: the schema coordinate of the element
    :param first: the location of the description that is kept
    :param first_description: the description text that is kept
    :param second: the location of the description that is discarded
    :param second_description: the description text that is discarded

    >>> from graphql.utilities import (
    ...     SdlDescriptionDifference, SdlSourceLocation)
    >>> loc = SdlSourceLocation("a.graphql", 1, 7)
    >>> diff = SdlDescriptionDifference("Query", loc, "One.", loc, "Two.")
    >>> print(diff)
    Differing descriptions for Query:
      first (a.graphql:1:7): 'One.'
      second (a.graphql:1:7): 'Two.'
    """

    coordinate: str
    """The schema coordinate of the element."""

    first: SdlSourceLocation
    """The location of the description that is kept."""

    first_description: str
    """The description text that is kept."""

    second: SdlSourceLocation
    """The location of the description that is discarded."""

    second_description: str
    """The description text that is discarded."""

    def __str__(self) -> str:
        return (
            f"Differing descriptions for {self.coordinate}:\n"
            f"  first ({self.first}): {self.first_description!r}\n"
            f"  second ({self.second}): {self.second_description!r}"
        )


@dataclass(frozen=True)
class SdlCompositionResult:
    """The outcome of composing SDL documents with :func:`compose_sdl`.

    :param schema: the composed schema, or ``None`` if composition failed
    :param conflicts: semantic conflicts found while merging
    :param description_differences: differing descriptions, reported but not
        treated as conflicts
    :param validation_errors: errors reported when validating the composed
        schema

    >>> from graphql.utilities import compose_sdl
    >>> result = compose_sdl([("a.graphql", "type Query { a: Int }")])
    >>> result.ok
    True
    """

    schema: GraphQLSchema | None
    """The composed schema, or ``None`` if composition failed."""

    conflicts: tuple[SdlConflict, ...] = ()
    """Semantic conflicts found while merging, with both source locations."""

    description_differences: tuple[SdlDescriptionDifference, ...] = ()
    """Differing descriptions, reported but not treated as conflicts."""

    validation_errors: tuple[GraphQLError, ...] = ()
    """Errors reported when validating the composed schema."""

    @property
    def ok(self) -> bool:
        """Whether composition produced a valid schema."""
        return self.schema is not None


def compose_sdl(
    documents: Collection[tuple[str, str]] | Mapping[str, str],
    *,
    assume_valid: bool = False,
) -> SdlCompositionResult:
    """Compose SDL documents from multiple sources into a single schema.

    Each document is given as a ``(source_name, sdl)`` pair (or as a mapping
    from source names to SDL strings). The source names are used in the
    reported locations of conflicts and description differences.

    Documents are merged semantically: declarations of the same type are
    combined, identical re-declarations are kept only once, and genuinely
    conflicting declarations are reported as :class:`SdlConflict` entries
    that point back to both source documents. If any conflict is found, or
    the merged document does not pass schema validation, no schema is
    returned (``result.schema`` is ``None``); a partial schema is never
    produced.

    :param documents: the SDL documents to compose, as ``(source_name, sdl)``
        pairs or as a mapping from source names to SDL strings
    :param assume_valid: skip validation of the composed schema
    :returns: an :class:`SdlCompositionResult` holding either the composed
        schema or the conflicts and validation errors that were found

    Example:

        >>> from graphql.utilities import compose_sdl
        >>> result = compose_sdl([
        ...     ("base.graphql", "type Query { a: Int }"),
        ...     ("ext.graphql", "extend type Query { b: String }"),
        ... ])
        >>> result.ok
        True
    """
    if isinstance(documents, Mapping):
        items = list(documents.items())
    else:
        items = list(documents)
    composer = _Composer()
    for source_name, sdl in items:
        if not isinstance(sdl, str):
            raise TypeError(
                "compose_sdl() expects (source_name, sdl) pairs with SDL"
                f" strings, but source {source_name!r} is a {type(sdl)}."
            )
        composer.add_document(parse(Source(sdl, source_name)))
    conflicts = tuple(composer.conflicts)
    description_differences = tuple(composer.description_differences)
    if conflicts:
        return SdlCompositionResult(
            schema=None,
            conflicts=conflicts,
            description_differences=description_differences,
        )
    document = composer.build_document()
    schema = build_ast_schema(
        document, assume_valid=assume_valid, assume_valid_sdl=assume_valid
    )
    validation_errors: tuple[GraphQLError, ...] = ()
    if not assume_valid:
        validation_errors = tuple(validate_schema(schema))
    if validation_errors:
        return SdlCompositionResult(
            schema=None,
            description_differences=description_differences,
            validation_errors=validation_errors,
        )
    return SdlCompositionResult(
        schema=schema, description_differences=description_differences
    )


_TYPE_NODE_CLASSES = (
    ScalarTypeDefinitionNode,
    ScalarTypeExtensionNode,
    ObjectTypeDefinitionNode,
    ObjectTypeExtensionNode,
    InterfaceTypeDefinitionNode,
    InterfaceTypeExtensionNode,
    UnionTypeDefinitionNode,
    UnionTypeExtensionNode,
    EnumTypeDefinitionNode,
    EnumTypeExtensionNode,
    InputObjectTypeDefinitionNode,
    InputObjectTypeExtensionNode,
)

_DEFINITION_NODE_CLASSES = {
    "scalar": ScalarTypeDefinitionNode,
    "object": ObjectTypeDefinitionNode,
    "interface": InterfaceTypeDefinitionNode,
    "union": UnionTypeDefinitionNode,
    "enum": EnumTypeDefinitionNode,
    "input": InputObjectTypeDefinitionNode,
}


@dataclass
class _TypeState:
    """All declarations of one named type, accumulated across documents."""

    kind: str
    name: str
    first_node: Node
    description: StringValueNode | None = None
    description_carrier: Node | None = None
    directives: list[ConstDirectiveNode] = field(default_factory=list)
    directive_keys: set = field(default_factory=set)
    interfaces: dict[str, NamedTypeNode] = field(default_factory=dict)
    fields: dict[str, FieldDefinitionNode | InputValueDefinitionNode] = field(
        default_factory=dict
    )
    enum_values: dict[str, EnumValueDefinitionNode] = field(default_factory=dict)
    union_members: dict[str, NamedTypeNode] = field(default_factory=dict)


class _Composer:
    """Merge the definitions of parsed SDL documents into one document."""

    def __init__(self) -> None:
        self.conflicts: list[SdlConflict] = []
        self.description_differences: list[SdlDescriptionDifference] = []
        self.has_schema_definition = False
        self.schema_description: StringValueNode | None = None
        self.schema_description_carrier: Node | None = None
        self.schema_directives: list[ConstDirectiveNode] = []
        self.schema_directive_keys: set = set()
        self.operation_types: dict[str, OperationTypeDefinitionNode] = {}
        self.directive_definitions: dict[str, DirectiveDefinitionNode] = {}
        self.types: dict[str, _TypeState] = {}

    def add_document(self, document: DocumentNode) -> None:
        for definition in document.definitions:
            if isinstance(
                definition, (SchemaDefinitionNode, SchemaExtensionNode)
            ):
                self._merge_schema(definition)
            elif isinstance(definition, DirectiveDefinitionNode):
                self._merge_directive_definition(definition)
            elif isinstance(definition, _TYPE_NODE_CLASSES):
                self._merge_type(definition)
            else:
                kind = getattr(definition, "kind", type(definition).__name__)
                raise GraphQLError(
                    "compose_sdl() only accepts schema definition language"
                    f" (SDL) documents, but found a {kind!r} definition."
                )

    def build_document(self) -> DocumentNode:
        definitions: list = []
        if self.has_schema_definition:
            definitions.append(
                SchemaDefinitionNode(
                    description=self.schema_description,
                    directives=tuple(self.schema_directives),
                    operation_types=tuple(self.operation_types.values()),
                )
            )
        definitions.extend(self.directive_definitions.values())
        for name, state in self.types.items():
            node_class = _DEFINITION_NODE_CLASSES[state.kind]
            kwargs: dict = {
                "name": NameNode(value=name),
                "description": state.description,
                "directives": tuple(state.directives),
            }
            if state.kind in ("object", "interface"):
                kwargs["interfaces"] = tuple(state.interfaces.values())
                kwargs["fields"] = tuple(state.fields.values())
            elif state.kind == "input":
                kwargs["fields"] = tuple(state.fields.values())
            elif state.kind == "enum":
                kwargs["values"] = tuple(state.enum_values.values())
            elif state.kind == "union":
                kwargs["types"] = tuple(state.union_members.values())
            definitions.append(node_class(**kwargs))
        return DocumentNode(definitions=tuple(definitions))

    def _merge_schema(self, node: Node) -> None:
        self.has_schema_definition = True
        description = getattr(node, "description", None)
        if description is not None:
            if self.schema_description is None:
                self.schema_description = description
                self.schema_description_carrier = node
            elif self.schema_description.value != description.value:
                self._add_description_difference(
                    "schema", self.schema_description_carrier, node
                )
        self._merge_directives(
            self.schema_directives, self.schema_directive_keys, node.directives
        )
        for operation_type in node.operation_types or ():
            operation = operation_type.operation.value
            existing = self.operation_types.get(operation)
            if existing is None:
                self.operation_types[operation] = operation_type
            elif existing.type.name.value != operation_type.type.name.value:
                self._add_conflict(
                    f"schema.{operation}",
                    f"root operation type {existing.type.name.value!r}"
                    f" conflicts with {operation_type.type.name.value!r}",
                    existing,
                    operation_type,
                )

    def _merge_directive_definition(self, node: DirectiveDefinitionNode) -> None:
        name = node.name.value
        existing = self.directive_definitions.get(name)
        if existing is None:
            self.directive_definitions[name] = node
            return
        reasons = []
        if existing.repeatable != node.repeatable:
            reasons.append("one declaration is repeatable, the other is not")
        existing_locations = {loc.value for loc in existing.locations}
        node_locations = {loc.value for loc in node.locations}
        if existing_locations != node_locations:
            reasons.append(
                f"locations {sorted(existing_locations)}"
                f" differ from {sorted(node_locations)}"
            )
        equal, reason = _arguments_equal(existing.arguments, node.arguments)
        if not equal:
            reasons.append(reason)
        if reasons:
            self._add_conflict(f"@{name}", "; ".join(reasons), existing, node)
        else:
            self._check_description(f"@{name}", existing, node)

    def _merge_type(self, node: Node) -> None:
        name = node.name.value
        kind = _type_kind(node)
        state = self.types.get(name)
        if state is None:
            state = _TypeState(kind=kind, name=name, first_node=node)
            self.types[name] = state
        elif state.kind != kind:
            self._add_conflict(
                name,
                f"declared as a {state.kind} type and as a {kind} type",
                state.first_node,
                node,
            )
            return
        description = getattr(node, "description", None)
        if description is not None:
            if state.description is None:
                state.description = description
                state.description_carrier = node
            elif state.description.value != description.value:
                self._add_description_difference(
                    name, state.description_carrier, node
                )
        self._merge_directives(
            state.directives, state.directive_keys, node.directives
        )
        if kind in ("object", "interface"):
            for interface in node.interfaces or ():
                state.interfaces.setdefault(interface.name.value, interface)
            self._merge_fields(state, node)
        elif kind == "input":
            self._merge_input_fields(state, node)
        elif kind == "enum":
            self._merge_enum_values(state, node)
        elif kind == "union":
            for member in node.types or ():
                state.union_members.setdefault(member.name.value, member)

    def _merge_fields(self, state: _TypeState, node: Node) -> None:
        for field_node in node.fields or ():
            field_name = field_node.name.value
            coordinate = f"{state.name}.{field_name}"
            existing = state.fields.get(field_name)
            if existing is None:
                state.fields[field_name] = field_node
                continue
            equal, reason = _field_definitions_equal(existing, field_node)
            if not equal:
                self._add_conflict(coordinate, reason, existing, field_node)
                continue
            self._check_description(coordinate, existing, field_node)
            self._check_argument_descriptions(
                coordinate, existing.arguments, field_node.arguments
            )

    def _merge_input_fields(self, state: _TypeState, node: Node) -> None:
        for field_node in node.fields or ():
            field_name = field_node.name.value
            coordinate = f"{state.name}.{field_name}"
            existing = state.fields.get(field_name)
            if existing is None:
                state.fields[field_name] = field_node
                continue
            equal, reason = _input_values_equal(existing, field_node)
            if not equal:
                self._add_conflict(coordinate, reason, existing, field_node)
                continue
            self._check_description(coordinate, existing, field_node)

    def _merge_enum_values(self, state: _TypeState, node: Node) -> None:
        for value_node in node.values or ():
            value_name = value_node.name.value
            coordinate = f"{state.name}.{value_name}"
            existing = state.enum_values.get(value_name)
            if existing is None:
                state.enum_values[value_name] = value_node
                continue
            if not _directives_equal(existing.directives, value_node.directives):
                self._add_conflict(
                    coordinate,
                    "directive applications differ:"
                    f" {_print_directives(existing.directives)} vs"
                    f" {_print_directives(value_node.directives)}",
                    existing,
                    value_node,
                )
                continue
            self._check_description(coordinate, existing, value_node)

    def _merge_directives(
        self, directives: list, directive_keys: set, new_directives: tuple
    ) -> None:
        for directive in new_directives or ():
            key = _directive_key(directive)
            if key not in directive_keys:
                directive_keys.add(key)
                directives.append(directive)

    def _check_description(
        self, coordinate: str, kept_node: Node, incoming_node: Node
    ) -> None:
        kept = getattr(kept_node, "description", None)
        incoming = getattr(incoming_node, "description", None)
        if (
            kept is not None
            and incoming is not None
            and kept.value != incoming.value
        ):
            self._add_description_difference(coordinate, kept_node, incoming_node)

    def _check_argument_descriptions(
        self, coordinate: str, kept_args: tuple, incoming_args: tuple
    ) -> None:
        incoming_map = {arg.name.value: arg for arg in incoming_args or ()}
        for kept_arg in kept_args or ():
            incoming_arg = incoming_map.get(kept_arg.name.value)
            if incoming_arg is not None:
                self._check_description(
                    f"{coordinate}({kept_arg.name.value}:)",
                    kept_arg,
                    incoming_arg,
                )

    def _add_conflict(
        self, coordinate: str, message: str, first_node: Node, second_node: Node
    ) -> None:
        self.conflicts.append(
            SdlConflict(
                coordinate=coordinate,
                message=message,
                first=_loc_of(first_node),
                second=_loc_of(second_node),
            )
        )

    def _add_description_difference(
        self, coordinate: str, kept_node: Node, incoming_node: Node
    ) -> None:
        kept = kept_node.description
        incoming = incoming_node.description
        self.description_differences.append(
            SdlDescriptionDifference(
                coordinate=coordinate,
                first=_loc_of(kept_node),
                first_description=kept.value if kept else "",
                second=_loc_of(incoming_node),
                second_description=incoming.value if incoming else "",
            )
        )


def _loc_of(node: Node) -> SdlSourceLocation:
    loc = node.loc
    name = getattr(node, "name", None)
    if isinstance(name, NameNode) and name.loc is not None:
        loc = name.loc
    if loc is None:
        return SdlSourceLocation("<unknown>", 0, 0)
    position = get_location(loc.source, loc.start)
    return SdlSourceLocation(loc.source.name, position.line, position.column)


def _type_kind(node: Node) -> str:
    if isinstance(node, (ObjectTypeDefinitionNode, ObjectTypeExtensionNode)):
        return "object"
    if isinstance(node, (InterfaceTypeDefinitionNode, InterfaceTypeExtensionNode)):
        return "interface"
    if isinstance(node, (UnionTypeDefinitionNode, UnionTypeExtensionNode)):
        return "union"
    if isinstance(node, (EnumTypeDefinitionNode, EnumTypeExtensionNode)):
        return "enum"
    if isinstance(
        node, (InputObjectTypeDefinitionNode, InputObjectTypeExtensionNode)
    ):
        return "input"
    return "scalar"


def _type_equal(first: TypeNode, second: TypeNode) -> bool:
    """Structurally compare two type references, including wrappers."""
    if isinstance(first, NonNullTypeNode) and isinstance(second, NonNullTypeNode):
        return _type_equal(first.type, second.type)
    if isinstance(first, ListTypeNode) and isinstance(second, ListTypeNode):
        return _type_equal(first.type, second.type)
    if isinstance(first, NamedTypeNode) and isinstance(second, NamedTypeNode):
        return first.name.value == second.name.value
    return False


def _value_key(value: ValueNode) -> tuple:
    """A hashable, order-insensitive key for a value literal."""
    kind = value.kind
    if kind in (
        "int_value",
        "float_value",
        "string_value",
        "boolean_value",
        "enum_value",
        "variable",
    ):
        return kind, value.value
    if kind == "list_value":
        return kind, tuple(_value_key(item) for item in value.values)
    if kind == "object_value":
        return kind, tuple(
            sorted(
                (field.name.value, _value_key(field.value))
                for field in value.fields
            )
        )
    return (kind,)


def _value_equal(first: ValueNode | None, second: ValueNode | None) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return _value_key(first) == _value_key(second)


def _directive_key(directive: ConstDirectiveNode) -> tuple:
    arguments = directive.arguments or ()
    return directive.name.value, tuple(
        sorted((arg.name.value, _value_key(arg.value)) for arg in arguments)
    )


def _directives_equal(first: tuple, second: tuple) -> bool:
    first_keys = sorted(_directive_key(directive) for directive in first or ())
    second_keys = sorted(_directive_key(directive) for directive in second or ())
    return first_keys == second_keys


def _print_directives(directives: tuple) -> str:
    if not directives:
        return "(none)"
    return " ".join(print_ast(directive) for directive in directives)


def _describe_input_value(node: InputValueDefinitionNode) -> str:
    text = print_ast(node.type)
    if node.default_value is not None:
        text += f" = {print_ast(node.default_value)}"
    return text


def _input_values_equal(
    first: InputValueDefinitionNode, second: InputValueDefinitionNode
) -> tuple:
    """Semantically compare two input value definitions."""
    if not _type_equal(first.type, second.type):
        return False, (
            f"type {print_ast(first.type)!r} conflicts with"
            f" {print_ast(second.type)!r}"
        )
    if not _value_equal(first.default_value, second.default_value):
        first_default = (
            print_ast(first.default_value)
            if first.default_value is not None
            else "(none)"
        )
        second_default = (
            print_ast(second.default_value)
            if second.default_value is not None
            else "(none)"
        )
        return (
            False,
            f"default value {first_default} conflicts with {second_default}",
        )
    if not _directives_equal(first.directives, second.directives):
        return False, (
            "directive applications differ:"
            f" {_print_directives(first.directives)} vs"
            f" {_print_directives(second.directives)}"
        )
    return True, ""


def _arguments_equal(first_args: tuple, second_args: tuple) -> tuple:
    """Semantically compare two argument definition lists."""
    first_map = {arg.name.value: arg for arg in first_args or ()}
    second_map = {arg.name.value: arg for arg in second_args or ()}
    if first_map.keys() != second_map.keys():
        details = []
        only_first = sorted(first_map.keys() - second_map.keys())
        only_second = sorted(second_map.keys() - first_map.keys())
        if only_first:
            details.append(f"only in first: {', '.join(only_first)}")
        if only_second:
            details.append(f"only in second: {', '.join(only_second)}")
        return False, f"argument list differs ({'; '.join(details)})"
    for name, first_arg in first_map.items():
        second_arg = second_map[name]
        equal, reason = _input_values_equal(first_arg, second_arg)
        if not equal:
            return (
                False,
                f"argument {name!r} differs:"
                f" {_describe_input_value(first_arg)} vs"
                f" {_describe_input_value(second_arg)} ({reason})",
            )
    return True, ""


def _field_definitions_equal(
    first: FieldDefinitionNode, second: FieldDefinitionNode
) -> tuple:
    """Semantically compare two field definitions."""
    if not _type_equal(first.type, second.type):
        return False, (
            f"field type {print_ast(first.type)!r} conflicts with"
            f" {print_ast(second.type)!r}"
        )
    equal, reason = _arguments_equal(first.arguments, second.arguments)
    if not equal:
        return False, reason
    if not _directives_equal(first.directives, second.directives):
        return False, (
            "directive applications differ:"
            f" {_print_directives(first.directives)} vs"
            f" {_print_directives(second.directives)}"
        )
    return True, ""
