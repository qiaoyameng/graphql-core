"""Migration of GraphQL documents across renamed schema fields"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, NamedTuple, TypeAlias

from ..error import GraphQLError
from ..language import (
    DocumentNode,
    FieldNode,
    InlineFragmentNode,
    NameNode,
    SelectionSetNode,
    Visitor,
    visit,
)
from ..type import (
    GraphQLCompositeType,
    GraphQLSchema,
    is_abstract_type,
    is_interface_type,
    is_object_type,
)
from .type_info import TypeInfo, TypeInfoVisitor

__all__ = ["FieldMigrationError", "FieldMigrationResult", "migrate_fields"]


FieldMap: TypeAlias = Mapping[str, str] | Iterable[tuple[str, str]]
"""Mapping of old schema coordinates (``"Type.field"``) to new field names."""


class FieldMigrationError(GraphQLError):
    """An error raised while migrating a document to renamed schema fields.

    The :attr:`coordinate` attribute holds the old schema coordinate
    (``"Type.field"``) of the field that caused the error, if applicable.
    Positions inside the document are available via :attr:`locations`,
    as with any other :class:`~graphql.error.GraphQLError`.

    :param message: The description of the error.
    :param nodes: The AST nodes at which the error occurred, used to
        determine the locations of the error in the document.
    :param coordinate: The old schema coordinate (``"Type.field"``) of the
        field that caused the error, if applicable.

    >>> error = FieldMigrationError("Cannot migrate.", coordinate="Query.old")
    >>> error.message
    'Cannot migrate.'
    >>> error.coordinate
    'Query.old'
    """

    coordinate: str | None
    """The old schema coordinate of the field that caused the error."""

    def __init__(
        self,
        message: str,
        nodes=None,
        coordinate: str | None = None,
    ) -> None:
        super().__init__(message, nodes=nodes)
        self.coordinate = coordinate


class FieldMigrationResult(NamedTuple):
    """The result of a document migration.

    Either :attr:`document` or :attr:`errors` is set. If the migration
    failed, no half-migrated document is produced.

    >>> result = FieldMigrationResult(None, ())
    >>> result.document is None
    True
    >>> result.errors
    ()
    """

    document: DocumentNode | None
    """The migrated document, or ``None`` if the migration failed."""
    errors: tuple[FieldMigrationError, ...]
    """The diagnostics describing why the migration could not be performed."""


_COORDINATE_RE = re.compile(r"^([_A-Za-z][_0-9A-Za-z]*)\.([_A-Za-z][_0-9A-Za-z]*)$")
_NAME_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")


def migrate_fields(
    old_schema: GraphQLSchema,
    new_schema: GraphQLSchema,
    document: DocumentNode,
    field_map: FieldMap,
) -> FieldMigrationResult:
    """Rewrite a document from an old schema to a new schema with renamed fields.

    The ``field_map`` explicitly maps old schema coordinates (``"Type.field"``)
    to the new field names. Only field references that actually resolve to a
    mapped coordinate are rewritten; fields with the same name on other types
    are left untouched. Whenever a field is renamed, an alias with the old
    field name is added (unless the field already has an alias), so that the
    response keys of the document stay to be the same as before the migration.

    Fields selected through an interface or union type condition are resolved
    per possible type. If the possible types of an abstract type would be
    renamed to different targets, if a rename target does not exist in the new
    schema, or if a rename introduces a field conflict, the migration fails
    and returns diagnostics with precise document locations instead of a
    seemingly successful document.

    The given ``document`` is not modified; a new AST is returned. Existing
    aliases, directives, and variable definitions are preserved.

    :param old_schema: The schema the document was written against.
    :param new_schema: The schema with the renamed fields.
    :param document: The parsed GraphQL document to migrate.
    :param field_map: A mapping (or iterable of pairs) from old schema
        coordinates of the form ``"Type.field"`` to the new field names.
    :returns: A :class:`FieldMigrationResult` with either the migrated
        document or a tuple of :class:`FieldMigrationError` diagnostics.

    >>> from graphql import build_schema, parse, print_ast
    >>> old_schema = build_schema("type Query { oldName: String }")
    >>> new_schema = build_schema("type Query { newName: String }")
    >>> document = parse("{ oldName }")
    >>> result = migrate_fields(
    ...     old_schema, new_schema, document, {"Query.oldName": "newName"})
    >>> result.errors
    ()
    >>> print(print_ast(result.document))
    {
      oldName: newName
    }
    """
    mapping, errors = _normalize_field_map(field_map)
    errors.extend(_validate_field_map(old_schema, mapping))

    decisions: dict[int, str] = {}
    if not errors:
        errors.extend(_collect_decisions(old_schema, new_schema, document, mapping, decisions))

    if errors:
        return FieldMigrationResult(None, tuple(errors))

    migrated = visit(document, _ApplyVisitor(decisions))
    return FieldMigrationResult(migrated, ())


def _normalize_field_map(
    field_map: FieldMap,
) -> tuple[dict[tuple[str, str], str], list[FieldMigrationError]]:
    """Normalize the field map to a dict of (type, field) -> new name."""
    items = field_map.items() if isinstance(field_map, Mapping) else field_map
    mapping: dict[tuple[str, str], str] = {}
    errors: list[FieldMigrationError] = []
    for coordinate, new_name in items:
        match = _COORDINATE_RE.match(coordinate)
        if not match:
            msg = f"Invalid schema coordinate in field map: {coordinate!r}."
            raise ValueError(msg)
        if not _NAME_RE.match(new_name):
            msg = f"Invalid field name in field map: {new_name!r}."
            raise ValueError(msg)
        key = (match[1], match[2])
        previous = mapping.get(key)
        if previous is not None and previous != new_name:
            errors.append(
                FieldMigrationError(
                    f"Conflicting field map entries for '{coordinate}':"
                    f" it cannot be renamed to both '{previous}' and"
                    f" '{new_name}'.",
                    coordinate=coordinate,
                )
            )
        else:
            mapping[key] = new_name
    return mapping, errors


def _validate_field_map(
    old_schema: GraphQLSchema,
    mapping: dict[tuple[str, str], str],
) -> list[FieldMigrationError]:
    """Check that all mapped coordinates exist in the old schema."""
    errors: list[FieldMigrationError] = []
    for (type_name, field_name) in sorted(mapping):
        coordinate = f"{type_name}.{field_name}"
        type_ = old_schema.get_type(type_name)
        if type_ is None:
            errors.append(
                FieldMigrationError(
                    f"Field map refers to type '{type_name}' which does not"
                    " exist in the old schema.",
                    coordinate=coordinate,
                )
            )
        elif not (is_object_type(type_) or is_interface_type(type_)):
            errors.append(
                FieldMigrationError(
                    f"Field map refers to '{coordinate}', but type"
                    f" '{type_name}' does not define fields in the old schema.",
                    coordinate=coordinate,
                )
            )
        elif field_name not in type_.fields:
            errors.append(
                FieldMigrationError(
                    f"Field map refers to '{coordinate}' which does not exist"
                    " in the old schema.",
                    coordinate=coordinate,
                )
            )
    return errors


def _collect_decisions(
    old_schema: GraphQLSchema,
    new_schema: GraphQLSchema,
    document: DocumentNode,
    mapping: dict[tuple[str, str], str],
    decisions: dict[int, str],
) -> list[FieldMigrationError]:
    """Collect the rename decision for every field node in the document.

    Fills ``decisions`` with a mapping of field node ids to new field names
    and returns any diagnostics found along the way, including field
    conflicts that the migration would introduce.
    """
    type_info = TypeInfo(old_schema)
    collector = _CollectVisitor(old_schema, new_schema, mapping, decisions, type_info)
    visit(document, TypeInfoVisitor(type_info, collector))
    return collector.errors


def _resolve_new_name(
    old_schema: GraphQLSchema,
    parent_type: GraphQLCompositeType,
    field_name: str,
    mapping: dict[tuple[str, str], str],
    node: FieldNode,
    errors: list[FieldMigrationError],
) -> str | None:
    """Resolve the new name for a field reference, or None if it is unmoved."""
    coordinate = f"{parent_type.name}.{field_name}"
    if is_object_type(parent_type):
        exact = mapping.get((parent_type.name, field_name))
        if exact is not None:
            return exact
        inherited = {
            mapping[(interface.name, field_name)]
            for interface in parent_type.interfaces
            if (interface.name, field_name) in mapping
        }
        if len(inherited) > 1:
            names = "', '".join(sorted(inherited))
            errors.append(
                FieldMigrationError(
                    f"Field '{coordinate}' cannot be migrated because the"
                    f" interfaces of '{parent_type.name}' map it to"
                    f" incompatible names: '{names}'.",
                    nodes=[node],
                    coordinate=coordinate,
                )
            )
            return None
        return inherited.pop() if inherited else None
    if is_interface_type(parent_type):
        default = mapping.get((parent_type.name, field_name))
        resolutions: dict[str, str] = {}
        for possible_type in old_schema.get_possible_types(parent_type):
            resolutions[possible_type.name] = mapping.get(
                (possible_type.name, field_name),
                default if default is not None else field_name,
            )
        distinct = set(resolutions.values())
        if len(distinct) > 1:
            details = ", ".join(
                f"'{type_name}.{field_name}'"
                + (f" maps to '{name}'" if name != field_name else " is not mapped")
                for type_name, name in sorted(resolutions.items())
            )
            errors.append(
                FieldMigrationError(
                    f"Field '{coordinate}' cannot be migrated because the"
                    f" implementations of '{parent_type.name}' have"
                    f" incompatible mappings: {details}.",
                    nodes=[node],
                    coordinate=coordinate,
                )
            )
            return None
        new_name = distinct.pop()
        return new_name if new_name != field_name else None
    return None


def _check_target_exists(
    new_schema: GraphQLSchema,
    parent_type: GraphQLCompositeType,
    field_name: str,
    new_name: str,
    node: FieldNode,
    errors: list[FieldMigrationError],
) -> None:
    """Check that the rename target exists in the new schema."""
    coordinate = f"{parent_type.name}.{field_name}"
    new_parent = new_schema.get_type(parent_type.name)
    if new_parent is None:
        errors.append(
            FieldMigrationError(
                f"Cannot migrate field '{coordinate}' to '{new_name}': type"
                f" '{parent_type.name}' does not exist in the new schema.",
                nodes=[node],
                coordinate=coordinate,
            )
        )
        return
    exists = False
    if is_object_type(new_parent) or is_interface_type(new_parent):
        exists = new_name in new_parent.fields
    if not exists and is_abstract_type(new_parent):
        possible_types = new_schema.get_possible_types(new_parent)
        exists = bool(possible_types) and all(
            new_name in possible_type.fields for possible_type in possible_types
        )
    if not exists:
        errors.append(
            FieldMigrationError(
                f"Cannot migrate field '{coordinate}' to '{new_name}': the"
                f" target field does not exist on type '{parent_type.name}'"
                " in the new schema.",
                nodes=[node],
                coordinate=coordinate,
            )
        )


def _conditions_overlap(
    schema: GraphQLSchema,
    first: GraphQLCompositeType,
    second: GraphQLCompositeType,
) -> bool:
    """Check whether two type conditions can apply to the same object."""
    if first.name == second.name:
        return True
    first_types = (
        schema.get_possible_types(first) if is_abstract_type(first) else (first,)
    )
    second_names = {
        type_.name
        for type_ in (
            schema.get_possible_types(second) if is_abstract_type(second) else (second,)
        )
    }
    return any(type_.name in second_names for type_ in first_types)


class _CollectVisitor(Visitor):
    """Collect rename decisions and diagnostics without modifying the AST."""

    def __init__(
        self,
        old_schema: GraphQLSchema,
        new_schema: GraphQLSchema,
        mapping: dict[tuple[str, str], str],
        decisions: dict[int, str],
        type_info: TypeInfo,
    ) -> None:
        super().__init__()
        self.old_schema = old_schema
        self.new_schema = new_schema
        self.mapping = mapping
        self.decisions = decisions
        self.type_info = type_info
        self.errors: list[FieldMigrationError] = []
        # Stack of selection sets, each holding entries of the form
        # (response key, new field name, condition type, was renamed, node).
        self._frames: list[list[tuple[str, str, GraphQLCompositeType, bool, FieldNode]]] = []
        self._reported_conflicts: set[tuple[int, int]] = set()

    def enter_selection_set(self, *_args) -> None:
        self._frames.append([])

    def leave_selection_set(self, node, _key, parent, *_args) -> None:
        frame = self._frames.pop()
        self._check_conflicts(frame)
        if isinstance(parent, InlineFragmentNode) and self._frames:
            # Fields inside an inline fragment share the response level of
            # the enclosing selection set.
            self._frames[-1].extend(frame)

    def enter_field(self, node: FieldNode, *_args) -> None:
        field_name = node.name.value
        parent_type = self.type_info.get_parent_type()
        if parent_type is None or field_name.startswith("__"):
            return
        new_name = _resolve_new_name(
            self.old_schema, parent_type, field_name, self.mapping, node, self.errors
        )
        if new_name is not None and new_name != field_name:
            _check_target_exists(
                self.new_schema,
                parent_type,
                field_name,
                new_name,
                node,
                self.errors,
            )
            self.decisions[id(node)] = new_name
            renamed = True
        else:
            new_name = field_name
            renamed = False
        response_key = node.alias.value if node.alias else field_name
        self._frames[-1].append((response_key, new_name, parent_type, renamed, node))

    def _check_conflicts(self, frame) -> None:
        for index, (key, new_name, condition, renamed, node) in enumerate(frame):
            for other_key, other_new, other_condition, other_renamed, other_node in frame[
                index + 1 :
            ]:
                if (
                    key != other_key
                    or new_name == other_new
                    or not (renamed or other_renamed)
                    or not _conditions_overlap(self.old_schema, condition, other_condition)
                ):
                    continue
                pair = (id(node), id(other_node))
                if pair in self._reported_conflicts:
                    continue
                self._reported_conflicts.add(pair)
                self.errors.append(
                    FieldMigrationError(
                        f"Migrating field '{condition.name}.{node.name.value}'"
                        f" to '{new_name}' conflicts with field"
                        f" '{other_condition.name}.{other_node.name.value}':"
                        f" both selections use the response key '{key}' but"
                        " refer to different fields.",
                        nodes=[node, other_node],
                        coordinate=f"{condition.name}.{node.name.value}",
                    )
                )


class _ApplyVisitor(Visitor):
    """Apply the collected rename decisions, producing a new AST."""

    def __init__(self, decisions: dict[int, str]) -> None:
        super().__init__()
        self.decisions = decisions

    def enter_field(self, node: FieldNode, *_args):
        new_name = self.decisions.get(id(node))
        if new_name is None:
            return None
        return FieldNode(
            alias=node.alias or NameNode(value=node.name.value),
            name=NameNode(value=new_name),
            arguments=node.arguments,
            directives=node.directives,
            selection_set=node.selection_set,
            loc=node.loc,
        )
