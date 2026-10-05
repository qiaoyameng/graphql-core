"""Compose a GraphQL schema from several named SDL source documents.

Teams often deliver their parts of a schema as separate SDL fragments. Simply
concatenating those fragments fails as soon as one type is declared in more than
one fragment (``Field 'X.y' can only be defined once.``), even when the
declarations agree with each other.

This module implements a merge step that understands the SDL semantics:

* The same named type may be declared (and extended) by several fragments.
  Non-conflicting fields, arguments, enum values, union members, implemented
  interfaces and directives are merged, regardless of whether a fragment uses a
  type definition or a type extension, and regardless of fragment order.
* Identical declarations are deduplicated once, not compared as printed text.
  Arguments, wrapped types, default values, enum values and directive usages are
  compared by their semantic value, so ``[Int!]`` vs. ``[Int !]`` or differently
  formatted list literals do not produce false conflicts.
* Different description strings are reported separately as *description notes*
  instead of masquerading as type conflicts.
* Real conflicts name both involved sources together with schema coordinates and
  precise locations. The batch is atomic: when any conflict exists, no partial

"""

from __future__ import annotations

from dataclasses import replace as dataclasses_replace

from typing import NamedTuple

from ..language import (
    DocumentNode,
    EnumTypeDefinitionNode,
    EnumTypeExtensionNode,
    InputObjectTypeDefinitionNode,
    InputObjectTypeExtensionNode,
    InterfaceTypeDefinitionNode,
    InterfaceTypeExtensionNode,
    NamedTypeNode,
    NameNode,
    ObjectTypeDefinitionNode,
    ObjectTypeExtensionNode,
    ScalarTypeDefinitionNode,
    ScalarTypeExtensionNode,
    SchemaDefinitionNode,
    SchemaExtensionNode,
    Source,
    UnionTypeDefinitionNode,
    UnionTypeExtensionNode,
    get_location,
    parse,
    print_ast,
)
from ..type import assert_valid_schema, specified_directives
from .build_ast_schema import build_ast_schema
from .value_from_ast_untyped import value_from_ast_untyped

__all__ = [
    "SdlDocument",
    "ComposeLocation",
    "ComposeIssue",
    "ComposedSchema",
    "ComposeSdlError",
    "compose_sdl",
]


class SdlDocument(NamedTuple):
    """An SDL fragment together with the name of its delivering team/source."""

    source_name: str
    """Name of the team or file that delivered the fragment."""

    body: str | Source | DocumentNode
    """SDL text, a parsed :class:`Source`, or an already parsed document AST."""


class ComposeLocation(NamedTuple):
    """A coordinate position inside one of the composed source documents."""

    source_name: str
    """Name of the source document containing the position."""

    line: int
    """One-indexed line in the source document."""

    column: int
    """One-indexed column in the source document."""

    def __str__(self) -> str:
        return f"{self.source_name}:{self.line}:{self.column}"


class ComposeIssue(NamedTuple):
    """A finding produced while composing SDL documents.

    ``severity`` is either ``"conflict"`` (fatal; the batch produces no schema) or
    ``"note"`` (the schema is still produced, e.g. for differing descriptions).
    """

    severity: str
    """``"conflict"`` (fatal) or ``"note"`` (non-fatal finding)."""

    coordinate: str
    """Schema coordinate of the finding, such as ``Query.media`` or ``E.A``."""

    message: str
    """Human-readable description of the finding."""

    locations: tuple[ComposeLocation, ...] = ()
    """Precise positions of both (or all) involved declarations."""

    def __str__(self) -> str:
        head = f"{self.severity.upper()} at {self.coordinate}: {self.message}"
        if self.locations:
            head += "\n  " + "\n  ".join(str(loc) for loc in self.locations)
        return head


class ComposedSchema(NamedTuple):
    """Result of a successful composition."""

    schema: object
    """The composed and fully validated :class:`graphql.GraphQLSchema`."""

    document_ast: DocumentNode
    """The merged document AST used to build the schema."""

    notes: tuple[ComposeIssue, ...]
    """Non-fatal findings, such as differing descriptions."""

    @property
    def sdl(self) -> str:
        """The merged schema rendered as SDL, after full schema validation.

        Rendered from the merged document AST so that custom directive usage is
        preserved in the output.
        """
        return print_ast(self.document_ast)


class ComposeSdlError(Exception):
    """Raised when SDL fragments cannot be composed into one schema.

    The composition is atomic: on any conflict neither a partial schema nor a
    partial document is returned. Non-fatal findings (such as differing
    descriptions) are available as :attr:`notes`; fatal findings as
    :attr:`conflicts`.

    :param conflicts: The fatal conflicts that prevented composition.
    :param notes: Non-fatal findings collected while composing.

    Raised when two fragments declare the same field with different types:

    >>> from graphql import compose_sdl
    >>> try:
    ...     compose_sdl([
    ...         ("a", "type Query { f: String }"),
    ...         ("b", "type Query { f: Int }"),
    ...     ])
    ... except Exception as error:
    ...     error.conflicts[0].coordinate
    'Query.f'
    """

    def __init__(self, conflicts: list[ComposeIssue], notes: list[ComposeIssue]):
        self.conflicts = tuple(conflicts)
        self.notes = tuple(notes)
        lines = [f"Cannot compose SDL documents: {len(conflicts)} conflict(s)."]
        lines.extend(f"  - {issue}" for issue in conflicts)
        if notes:
            lines.append(f"{len(notes)} non-conflicting note(s):")
            lines.extend(f"  - {issue}" for issue in notes)
        super().__init__("\n".join(lines))


_FIELD_BEARING_DEFINITION_KINDS = (
    ObjectTypeDefinitionNode,
    InterfaceTypeDefinitionNode,
)
_FIELD_BEARING_EXTENSION_KINDS = (
    ObjectTypeExtensionNode,
    InterfaceTypeExtensionNode,
)
_INPUT_DEFINITION_KINDS = (InputObjectTypeDefinitionNode,)
_INPUT_EXTENSION_KINDS = (InputObjectTypeExtensionNode,)
_ENUM_DEFINITION_KINDS = (EnumTypeDefinitionNode,)
_ENUM_EXTENSION_KINDS = (EnumTypeExtensionNode,)
_UNION_DEFINITION_KINDS = (UnionTypeDefinitionNode,)
_UNION_EXTENSION_KINDS = (UnionTypeExtensionNode,)
_SCALAR_DEFINITION_KINDS = (ScalarTypeDefinitionNode,)
_SCALAR_EXTENSION_KINDS = (ScalarTypeExtensionNode,)

_EXTENSION_TO_DEFINITION_KIND = {
    ObjectTypeExtensionNode: ObjectTypeDefinitionNode,
    InterfaceTypeExtensionNode: InterfaceTypeDefinitionNode,
    InputObjectTypeExtensionNode: InputObjectTypeDefinitionNode,
    EnumTypeExtensionNode: EnumTypeDefinitionNode,
    UnionTypeExtensionNode: UnionTypeDefinitionNode,
    ScalarTypeExtensionNode: ScalarTypeDefinitionNode,
}


def _values_equal(left: object, right: object) -> bool:
    """Compare untyped values produced from GraphQL value ASTs.

    Lists compare element by element and input objects compare as dictionaries,
    so the comparison never depends on how a literal was printed.
    """
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    return left == right and type(left) is type(right)


def _description_text(node: object) -> str | None:
    description = getattr(node, "description", None)
    return description.value if description is not None else None


def _print_optional_value(node: object | None) -> str:
    return print_ast(node) if node is not None else "(no default value)"


def _nodes(value) -> tuple:
    """Normalize optional tuple AST attributes (None when parsed absent)."""
    return value if value else ()


def _kind_label(node: object) -> str:
    kind = type(node).kind
    return kind.replace("_type_definition", "").replace("_type_extension", "")


def _same_named_type(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, NamedTypeNode):
        return left.name.value == right.name.value
    return _same_named_type(left.type, right.type)


def _same_value(left: object | None, right: object | None) -> bool:
    if left is None or right is None:
        return left is right
    return _values_equal(
        value_from_ast_untyped(left), value_from_ast_untyped(right)
    )


class _Occurrence(NamedTuple):
    """An AST node together with its source document provenance."""

    node: object
    source: Source
    source_name: str


class _MemberState:
    """Mutable state kept for a merged member (field/arg/enum value)."""

    __slots__ = ("node", "occurrence")

    def __init__(self, node: object, occurrence: _Occurrence):
        self.node = node
        self.occurrence = occurrence


class _Composer:
    def __init__(self) -> None:
        self.conflicts: list[ComposeIssue] = []
        self.notes: list[ComposeIssue] = []
        self._builtin_repeatable = {
            directive.name
            for directive in specified_directives
            if directive.is_repeatable
        }
        self._declared_repeatable: set[str] = set()
        self._type_directive_origin: dict[int, _Occurrence] = {}

    # -- reporting --------------------------------------------------------

    def _location(self, occurrence: _Occurrence, node: object) -> ComposeLocation:
        loc = getattr(node, "loc", None)
        source_location = get_location(occurrence.source, loc.start if loc else 0)
        return ComposeLocation(
            occurrence.source.name or occurrence.source_name,
            source_location.line,
            source_location.column,
        )

    def _report(
        self,
        severity: str,
        coordinate: str,
        message: str,
        sides: tuple[tuple[_Occurrence, object], ...],
    ) -> None:
        issue = ComposeIssue(
            severity,
            coordinate,
            message,
            tuple(self._location(occ, node) for occ, node in sides),
        )
        (self.conflicts if severity == "conflict" else self.notes).append(issue)

    # -- semantic comparison ----------------------------------------------

    @staticmethod
    def _same_directive(left: object, right: object) -> bool:
        if left.name.value != right.name.value:
            return False
        args_left = {arg.name.value: arg for arg in _nodes(left.arguments)}
        args_right = {arg.name.value: arg for arg in _nodes(right.arguments)}
        if args_left.keys() != args_right.keys():
            return False
        return all(
            _same_value(args_left[name].value, args_right[name].value)
            for name in args_left
        )

    def _is_repeatable(self, name: str) -> bool:
        return (
            name in self._builtin_repeatable
            or name in self._declared_repeatable
        )

    # -- member level -----------------------------------------------------

    def _merge_description(
        self,
        coordinate: str,
        state: _MemberState,
        candidate: _Occurrence,
    ) -> None:
        existing_text = _description_text(state.node)
        candidate_text = _description_text(candidate.node)
        if (
            existing_text is not None
            and candidate_text is not None
            and existing_text != candidate_text
        ):
            self._report(
                "note",
                coordinate,
                "descriptions differ; keeping the description from "
                f"'{state.occurrence.source_name}' and ignoring the one from "
                f"'{candidate.source_name}'",
                (
                    (state.occurrence, state.node.description),
                    (candidate, candidate.node.description),
                ),
            )

    def _merge_directives(
        self,
        coordinate: str,
        state: _MemberState,
        candidate: _Occurrence,
    ) -> tuple:
        """Return the merged directives of two semantically equal elements."""
        merged = list(_nodes(state.node.directives))
        for directive in _nodes(candidate.node.directives):
            same_name = [d for d in merged if d.name.value == directive.name.value]
            if not same_name:
                merged.append(directive)
                continue
            if self._is_repeatable(directive.name.value):
                if not any(self._same_directive(directive, d) for d in same_name):
                    merged.append(directive)
                continue
            other = same_name[0]
            if not self._same_directive(other, directive):
                self._report(
                    "conflict",
                    coordinate,
                    f"directive @{directive.name.value} used with different "
                    f"arguments: {print_ast(other)} vs {print_ast(directive)}",
                    ((state.occurrence, other), (candidate, directive)),
                )
        return tuple(merged)

    def _merge_input_value_members(
        self,
        coordinate: str,
        states: dict[str, _MemberState],
        order: list[str],
        occurrences: list[_Occurrence],
    ) -> None:
        for occurrence in occurrences:
            node = occurrence.node
            name = node.name.value
            item_coordinate = f"{coordinate}({name}:)"
            state = states.get(name)
            if state is None:
                states[name] = _MemberState(node, occurrence)
                order.append(name)
                continue
            if not _same_named_type(state.node.type, node.type):
                self._report(
                    "conflict",
                    item_coordinate,
                    f"type mismatch: {print_ast(state.node.type)} vs "
                    f"{print_ast(node.type)}",
                    ((state.occurrence, state.node.type),
                     (occurrence, node.type)),
                )
            if not _same_value(state.node.default_value, node.default_value):
                self._report(
                    "conflict",
                    item_coordinate,
                    "default value mismatch: "
                    f"{_print_optional_value(state.node.default_value)} vs "
                    f"{_print_optional_value(node.default_value)}",
                    ((state.occurrence, state.node), (occurrence, node)),
                )
            merged_directives = self._merge_directives(
                item_coordinate, state, occurrence
            )
            state.node = dataclasses_replace(
                state.node, directives=merged_directives
            )
            self._merge_description(item_coordinate, state, occurrence)

    def _merge_field_members(
        self,
        type_name: str,
        states: dict[str, _MemberState],
        order: list[str],
        occurrences: list[_Occurrence],
    ) -> None:
        for occurrence in occurrences:
            node = occurrence.node
            name = node.name.value
            coordinate = f"{type_name}.{name}"
            state = states.get(name)
            if state is None:
                states[name] = _MemberState(node, occurrence)
                order.append(name)
                continue
            if not _same_named_type(state.node.type, node.type):
                self._report(
                    "conflict",
                    coordinate,
                    f"field type mismatch: {print_ast(state.node.type)} vs "
                    f"{print_ast(node.type)}",
                    ((state.occurrence, state.node.type),
                     (occurrence, node.type)),
                )
            arg_states: dict[str, _MemberState] = {}
            arg_order: list[str] = []
            for arg in _nodes(state.node.arguments):
                arg_occ = _Occurrence(arg, state.occurrence.source,
                                      state.occurrence.source_name)
                arg_states[arg.name.value] = _MemberState(arg, arg_occ)
                arg_order.append(arg.name.value)
            self._merge_input_value_members(
                coordinate,
                arg_states,
                arg_order,
                [
                    _Occurrence(arg, occurrence.source, occurrence.source_name)
                    for arg in _nodes(node.arguments)
                ],
            )
            merged_arguments = tuple(
                arg_states[arg_name].node for arg_name in arg_order
            )
            merged_directives = self._merge_directives(coordinate, state, occurrence)
            state.node = dataclasses_replace(
                state.node,
                arguments=merged_arguments,
                directives=merged_directives,
            )
            self._merge_description(coordinate, state, occurrence)

    def _merge_enum_members(
        self,
        type_name: str,
        states: dict[str, _MemberState],
        order: list[str],
        occurrences: list[_Occurrence],
    ) -> None:
        for occurrence in occurrences:
            node = occurrence.node
            name = node.name.value
            coordinate = f"{type_name}.{name}"
            state = states.get(name)
            if state is None:
                states[name] = _MemberState(node, occurrence)
                order.append(name)
                continue
            merged_directives = self._merge_directives(coordinate, state, occurrence)
            state.node = dataclasses_replace(
                state.node, directives=merged_directives
            )
            self._merge_description(coordinate, state, occurrence)

    # -- type level -------------------------------------------------------

    def _merge_type_directive(
        self,
        coordinate: str,
        directives: list,
        directive: object,
        occurrence: _Occurrence,
    ) -> None:
        same_name = [d for d in directives if d.name.value == directive.name.value]
        if not same_name:
            directives.append(directive)
            return
        if self._is_repeatable(directive.name.value):
            if not any(self._same_directive(directive, d) for d in same_name):
                directives.append(directive)
            return
        other = same_name[0]
        other_occurrence = self._type_directive_origin[id(other)]
        if not self._same_directive(other, directive):
            self._report(
                "conflict",
                coordinate,
                f"directive @{directive.name.value} used with different arguments: "
                f"{print_ast(other)} vs {print_ast(directive)}",
                ((other_occurrence, other), (occurrence, directive)),
            )

    def merge_type(
        self,
        type_name: str,
        occurrences: list[_Occurrence],
    ) -> object | None:
        definition_kind: type | None = None
        for occurrence in occurrences:
            mapped = _EXTENSION_TO_DEFINITION_KIND.get(
                type(occurrence.node), type(occurrence.node)
            )
            if definition_kind is None:
                definition_kind = mapped
            elif definition_kind is not mapped:
                self._report(
                    "conflict",
                    type_name,
                    f"kind mismatch: '{_kind_label(occurrences[0].node)}' cannot be "
                    f"merged with '{_kind_label(occurrence.node)}'",
                    (
                        (occurrences[0], occurrences[0].node),
                        (occurrence, occurrence.node),
                    ),
                )
                return None

        field_states: dict[str, _MemberState] = {}
        field_order: list[str] = []
        input_states: dict[str, _MemberState] = {}
        input_order: list[str] = []
        enum_states: dict[str, _MemberState] = {}
        enum_order: list[str] = []
        interfaces: list[NamedTypeNode] = []
        interface_names: set[str] = set()
        union_types: list[NamedTypeNode] = []
        union_names: set[str] = set()
        directives: list = []
        description: tuple[str, object, _Occurrence] | None = None

        for occurrence in occurrences:
            node = occurrence.node

            text = _description_text(node)
            if text is not None:
                if description is None:
                    description = (text, node.description, occurrence)
                elif description[0] != text:
                    self._report(
                        "note",
                        type_name,
                        "descriptions differ; keeping the description from "
                        f"'{description[2].source_name}' and ignoring the one from "
                        f"'{occurrence.source_name}'",
                        (
                            (description[2], description[1]),
                            (occurrence, node.description),
                        ),
                    )

            for directive in _nodes(node.directives):
                self._merge_type_directive(
                    type_name, directives, directive, occurrence
                )
                self._type_directive_origin.setdefault(id(directive), occurrence)

            if isinstance(
                node,
                _FIELD_BEARING_DEFINITION_KINDS + _FIELD_BEARING_EXTENSION_KINDS,
            ):
                for named_type in _nodes(node.interfaces):
                    if named_type.name.value not in interface_names:
                        interfaces.append(named_type)
                        interface_names.add(named_type.name.value)
                self._merge_field_members(
                    type_name,
                    field_states,
                    field_order,
                    [
                        _Occurrence(field, occurrence.source, occurrence.source_name)
                        for field in _nodes(node.fields)
                    ],
                )
            elif isinstance(
                node, _INPUT_DEFINITION_KINDS + _INPUT_EXTENSION_KINDS
            ):
                self._merge_input_value_members(
                    type_name,
                    input_states,
                    input_order,
                    [
                        _Occurrence(field, occurrence.source, occurrence.source_name)
                        for field in _nodes(node.fields)
                    ],
                )
            elif isinstance(node, _ENUM_DEFINITION_KINDS + _ENUM_EXTENSION_KINDS):
                self._merge_enum_members(
                    type_name,
                    enum_states,
                    enum_order,
                    [
                        _Occurrence(value, occurrence.source, occurrence.source_name)
                        for value in _nodes(node.values)
                    ],
                )
            elif isinstance(node, _UNION_DEFINITION_KINDS + _UNION_EXTENSION_KINDS):
                for named_type in _nodes(node.types):
                    if named_type.name.value not in union_names:
                        union_types.append(named_type)
                        union_names.add(named_type.name.value)

        common = dict(
            description=description[1] if description else None,
            name=NameNode(value=type_name),
            directives=tuple(directives),
        )
        if definition_kind is ObjectTypeDefinitionNode:
            return ObjectTypeDefinitionNode(
                **common,
                interfaces=tuple(interfaces),
                fields=tuple(field_states[name].node for name in field_order),
            )
        if definition_kind is InterfaceTypeDefinitionNode:
            return InterfaceTypeDefinitionNode(
                **common,
                interfaces=tuple(interfaces),
                fields=tuple(field_states[name].node for name in field_order),
            )
        if definition_kind is InputObjectTypeDefinitionNode:
            return InputObjectTypeDefinitionNode(
                **common,
                fields=tuple(input_states[name].node for name in input_order),
            )
        if definition_kind is EnumTypeDefinitionNode:
            return EnumTypeDefinitionNode(
                **common,
                values=tuple(enum_states[name].node for name in enum_order),
            )
        if definition_kind is UnionTypeDefinitionNode:
            return UnionTypeDefinitionNode(
                **common, types=tuple(union_types)
            )
        return ScalarTypeDefinitionNode(**common)

    # -- directive definitions -------------------------------------------

    def merge_directive_definition(
        self,
        name: str,
        occurrences: list[_Occurrence],
    ) -> object | None:
        first = occurrences[0].node
        state = _MemberState(first, occurrences[0])
        for occurrence in occurrences[1:]:
            node = occurrence.node
            coordinate = f"@{name}"
            if set(_nodes(node.locations)) != set(_nodes(first.locations)):
                self._report(
                    "conflict",
                    coordinate,
                    "directive locations mismatch: "
                    f"{' | '.join(_nodes(first.locations))} vs "
                    f"{' | '.join(_nodes(node.locations))}",
                    ((occurrences[0], first), (occurrence, node)),
                )
            if node.repeatable != first.repeatable:
                self._report(
                    "conflict",
                    coordinate,
                    "directive repeatability mismatch: "
                    f"{'repeatable' if first.repeatable else 'non-repeatable'} vs "
                    f"{'repeatable' if node.repeatable else 'non-repeatable'}",
                    ((occurrences[0], first), (occurrence, node)),
                )
            arg_states: dict[str, _MemberState] = {}
            arg_order: list[str] = []
            self._merge_input_value_members(
                coordinate, arg_states, arg_order,
                [_Occurrence(arg, occurrences[0].source, occurrences[0].source_name)
                 for arg in _nodes(first.arguments)],
            )
            self._merge_input_value_members(
                coordinate, arg_states, arg_order,
                [_Occurrence(arg, occurrence.source, occurrence.source_name)
                 for arg in _nodes(node.arguments)],
            )
            first = dataclasses_replace(
                first,
                arguments=tuple(arg_states[n].node for n in arg_order),
                directives=self._merge_directives(coordinate, state, occurrence),
            )
            state.node = first
            self._merge_description(coordinate, state, occurrence)
        if first.repeatable:
            self._declared_repeatable.add(name)
        return first

    # -- schema definition ------------------------------------------------

    def merge_schema(
        self,
        occurrences: list[_Occurrence],
    ) -> object | None:
        directives: list = []
        operation_types: dict[object, object] = {}
        operation_sources: dict[object, _Occurrence] = {}
        operation_order: list[object] = []
        description: tuple[str, object, _Occurrence] | None = None

        for occurrence in occurrences:
            node = occurrence.node
            if not isinstance(node, (SchemaDefinitionNode, SchemaExtensionNode)):
                continue
            text = _description_text(node)
            if isinstance(node, SchemaDefinitionNode) and text is not None:
                if description is None:
                    description = (text, node.description, occurrence)
                elif description[0] != text:
                    self._report(
                        "note",
                        "schema",
                        "descriptions differ; keeping the description from "
                        f"'{description[2].source_name}' and ignoring the one from "
                        f"'{occurrence.source_name}'",
                        ((description[2], description[1]),
                         (occurrence, node.description)),
                    )
            for directive in _nodes(node.directives):
                self._merge_type_directive(
                    "schema", directives, directive, occurrence
                )
                self._type_directive_origin.setdefault(id(directive), occurrence)
            for operation in _nodes(node.operation_types):
                operation_kind = operation.operation
                operation_name = operation_kind.value
                coordinate = f"schema.{operation_name}"
                new_type = operation.type.name.value
                existing = operation_types.get(operation_kind)
                if existing is None:
                    operation_types[operation_kind] = operation
                    operation_sources[operation_kind] = occurrence
                    operation_order.append(operation_kind)
                elif existing.type.name.value != new_type:
                    self._report(
                        "conflict",
                        coordinate,
                        f"root {operation_name} type mismatch: "
                        f"{existing.type.name.value} vs {new_type}",
                        (
                            (operation_sources[operation_kind], existing),
                            (occurrence, operation),
                        ),
                    )

        if not isinstance(occurrences[0].node, SchemaDefinitionNode):
            # Only schema extensions were given; synthesize a schema definition.
            return SchemaDefinitionNode(
                description=None,
                directives=tuple(directives),
                operation_types=tuple(
                    operation_types[key] for key in operation_order
                ),
            )
        base = occurrences[0].node
        return SchemaDefinitionNode(
            description=description[1] if description else base.description,
            directives=tuple(directives),
            operation_types=tuple(
                operation_types[key] for key in operation_order
            ),
        )

def compose_sdl(documents: list[SdlDocument | tuple[str, str]]) -> ComposedSchema:
    """Compose a single validated schema from named SDL fragments.

    :param documents: Pairs of ``(source_name, sdl_text)`` (or
        :class:`SdlDocument` objects). The source name identifies the delivering
        team/file and is used in every diagnostic.
    :returns: A :class:`ComposedSchema` holding the built and validated schema,
        the merged document AST and any non-fatal notes.
    :raises ComposeSdlError: If the fragments conflict semantically or the merged
        document fails schema validation. The batch is atomic; no schema is
        returned in that case.

    Non-conflicting declarations of the same type are merged regardless of
    whether they use a definition or an extension and regardless of order::

        >>> result = compose_sdl([
        ...     ("team-a", "type Query { a: String }"),
        ...     ("team-b", "extend type Query { b: Int }"),
        ... ])
        >>> sorted(result.schema.query_type.fields)
        ['a', 'b']
    """
    composer = _Composer()

    type_groups: dict[str, list[_Occurrence]] = {}
    directive_groups: dict[str, list[_Occurrence]] = {}
    schema_occurrences: list[_Occurrence] = []
    type_order: list[str] = []
    directive_order: list[str] = []
    has_schema = False

    for document in documents:
        if not isinstance(document, SdlDocument):
            source_name, body = document
            document = SdlDocument(source_name, body)
        source_name, body = document

        try:
            if isinstance(body, DocumentNode):
                document_ast = body
                existing = (
                    body.loc.source
                    if body.loc is not None and body.loc.source is not None
                    else None
                )
                source = (
                    existing
                    if existing is not None and existing.name == source_name
                    else Source(
                        existing.body if existing is not None else "",
                        source_name,
                    )
                )
            else:
                source = (
                    body
                    if isinstance(body, Source)
                    else Source(body, source_name)
                )
                document_ast = parse(source)
        except Exception as error:
            parse_locations = ()
            error_positions = getattr(error, "positions", None)
            if error_positions and isinstance(body, (str, Source)):
                source_location = get_location(source, error_positions[0])
                parse_locations = (
                    ComposeLocation(
                        source_name,
                        source_location.line,
                        source_location.column,
                    ),
                )
            raise ComposeSdlError(
                [
                    ComposeIssue(
                        "conflict",
                        "<document>",
                        f"failed to parse SDL from '{source_name}': {error}",
                        parse_locations,
                    )
                ],
                [],
            ) from error

        for definition in document_ast.definitions:
            occurrence = _Occurrence(definition, source, source_name)
            if isinstance(definition, (SchemaDefinitionNode, SchemaExtensionNode)):
                schema_occurrences.append(occurrence)
                if isinstance(definition, SchemaDefinitionNode):
                    has_schema = True
                continue
            name_node = getattr(definition, "name", None)
            if name_node is None:
                raise ComposeSdlError(
                    [
                        ComposeIssue(
                            "conflict",
                            "<document>",
                            f"unsupported executable definition in SDL from "
                            f"'{source_name}'",
                            (
                                ComposeLocation(
                                    source_name,
                                    *get_location(
                                        source,
                                        definition.loc.start if definition.loc else 0,
                                    ),
                                ),
                            ),
                        )
                    ],
                    [],
                )
            name = name_node.value
            from ..language import DirectiveDefinitionNode

            if isinstance(definition, DirectiveDefinitionNode):
                if name not in directive_groups:
                    directive_groups[name] = []
                    directive_order.append(name)
                directive_groups[name].append(occurrence)
            else:
                if name not in type_groups:
                    type_groups[name] = []
                    type_order.append(name)
                type_groups[name].append(occurrence)

    merged_definitions: list = []

    if has_schema or schema_occurrences:
        schema_node = composer.merge_schema(schema_occurrences)
        if schema_node is not None:
            merged_definitions.append(schema_node)

    for name in directive_order:
        merged = composer.merge_directive_definition(
            name, directive_groups[name]
        )
        if merged is not None:
            merged_definitions.append(merged)

    for name in type_order:
        merged = composer.merge_type(name, type_groups[name])
        if merged is not None:
            merged_definitions.append(merged)

    if composer.conflicts:
        raise ComposeSdlError(composer.conflicts, composer.notes)

    merged_document = DocumentNode(definitions=tuple(merged_definitions))

    try:
        schema = build_ast_schema(merged_document)
        assert_valid_schema(schema)
    except Exception as error:
        locations = []
        for node in getattr(error, "nodes", None) or []:
            loc = getattr(node, "loc", None)
            if loc is not None and loc.source is not None:
                source_location = get_location(loc.source, loc.start)
                locations.append(
                    ComposeLocation(
                        loc.source.name,
                        source_location.line,
                        source_location.column,
                    )
                )
        issue = ComposeIssue(
            "conflict",
            "<merged schema>",
            f"merged schema does not pass validation: {error}",
            tuple(locations),
        )
        raise ComposeSdlError([issue], composer.notes) from error

    return ComposedSchema(schema, merged_document, tuple(composer.notes))
