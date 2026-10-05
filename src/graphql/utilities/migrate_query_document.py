"""Migrate executable documents between GraphQL schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

from ..error import GraphQLError, GraphQLSyntaxError
from ..language import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    MemberCoordinateNode,
    NameNode,
    OperationDefinitionNode,
    SelectionSetNode,
    Visitor,
    parse,
    parse_schema_coordinate,
    print_ast,
    visit,
)
from ..type import (
    GraphQLField,
    GraphQLObjectType,
    GraphQLSchema,
    get_named_type,
    is_abstract_type,
    is_composite_type,
    is_object_type,
)
from ..validation import validate
from .build_ast_schema import build_schema
from .resolve_schema_coordinate import ResolvedField, resolve_schema_coordinate

if TYPE_CHECKING:
    from ..type import GraphQLCompositeType

__all__ = ["FieldCoordinate", "QueryMigrationResult", "migrate_query_document"]


class FieldCoordinate(NamedTuple):
    """An output field identified by a GraphQL schema coordinate."""

    parent_type: str
    """The object or interface type that declares the field."""
    field_name: str
    """The output field name."""

    def __str__(self) -> str:
        return f"{self.parent_type}.{self.field_name}"


FieldMappings: TypeAlias = (
    Mapping[str, str]
    | Mapping[FieldCoordinate, FieldCoordinate]
    | Iterable[
        tuple[
            str | FieldCoordinate | tuple[str, str],
            str | FieldCoordinate | tuple[str, str],
        ]
    ]
)


class QueryMigrationResult(NamedTuple):
    """The result of migrating an executable GraphQL document."""

    document: DocumentNode | None
    """The migrated document, or ``None`` when migration fails."""
    errors: tuple[GraphQLError, ...]
    """The migration and validation errors."""

    @property
    def ok(self) -> bool:
        """Whether the migration produced a document."""
        return self.document is not None and not self.errors


@dataclass
class _FieldUsage:
    operation: OperationDefinitionNode
    path: tuple[str, ...]
    runtime_types: set[GraphQLObjectType]


@dataclass
class _FieldOccurrence:
    node: FieldNode
    parent_type: GraphQLCompositeType
    usages: list[_FieldUsage]

    @property
    def runtime_types(self) -> set[GraphQLObjectType]:
        result: set[GraphQLObjectType] = set()
        for usage in self.usages:
            result.update(usage.runtime_types)
        return result


def migrate_query_document(
    old_schema: GraphQLSchema,
    new_schema: GraphQLSchema,
    document: DocumentNode | str,
    field_mappings: FieldMappings,
) -> QueryMigrationResult:
    """Migrate output fields in an executable document.

    Field mappings are explicit schema coordinates. A field is only rewritten when
    its static parent type, including fragment and inline fragment type conditions,
    matches the source coordinate. When the new field has another name, the original
    name is retained as an alias unless the field already had an alias.

    :param old_schema: The schema that the source document is valid against.
    :param new_schema: The schema used to validate the migrated document.
    :param document: The source document AST or GraphQL document text.
    :param field_mappings: Explicit output field coordinate mappings.
    :returns: The migrated document and any diagnostics.

    >>> old_schema = build_schema('type Query { name: String }')
    >>> new_schema = build_schema('type Query { displayName: String }')
    >>> result = migrate_query_document(
    ...     old_schema, new_schema, '{ name }',
    ...     {'Query.name': 'Query.displayName'})
    >>> result.ok
    True
    >>> print(print_ast(result.document))
    {
      name: displayName
    }
    """
    errors: list[GraphQLError] = []
    mappings, mapping_errors = _parse_mappings(
        old_schema, new_schema, field_mappings
    )
    errors.extend(mapping_errors)

    if isinstance(document, str):
        try:
            document_ast = parse(document)
        except GraphQLSyntaxError as error:
            return QueryMigrationResult(None, (*errors, error))
    else:
        document_ast = document

    if errors:
        return QueryMigrationResult(None, tuple(errors))

    old_errors = validate(old_schema, document_ast)
    if old_errors:
        return QueryMigrationResult(None, tuple(old_errors))

    occurrences = _collect_occurrences(old_schema, document_ast)
    renames: dict[int, str] = {}
    used_sources: set[FieldCoordinate] = set()
    for occurrence in occurrences.values():
        target = _resolve_target(
            occurrence, mappings, new_schema, errors, used_sources
        )
        if target is not None and target != occurrence.node.name.value:
            renames[id(occurrence.node)] = target

    if errors:
        errors.sort(key=_error_sort_key)
        return QueryMigrationResult(None, tuple(errors))

    for source, target in mappings.items():
        if source in used_sources:
            continue
        if _get_mapping_target_field(new_schema, target) is None:
            errors.append(GraphQLError(f"Target field '{target}' does not exist."))

    if errors:
        errors.sort(key=_error_sort_key)
        return QueryMigrationResult(None, tuple(errors))

    _detect_field_conflicts(occurrences, renames, errors)
    if errors:
        errors.sort(key=_error_sort_key)
        return QueryMigrationResult(None, tuple(errors))

    migrated_document = visit(document_ast, _FieldRenameVisitor(renames))
    new_errors = validate(new_schema, migrated_document)
    if new_errors:
        return QueryMigrationResult(None, tuple(new_errors))

    return QueryMigrationResult(migrated_document, ())


def _parse_mappings(
    old_schema: GraphQLSchema,
    new_schema: GraphQLSchema,
    field_mappings: FieldMappings,
) -> tuple[dict[FieldCoordinate, FieldCoordinate], list[GraphQLError]]:
    if hasattr(field_mappings, "items"):
        items = field_mappings.items()
    else:
        items = field_mappings

    mappings: dict[FieldCoordinate, FieldCoordinate] = {}
    errors: list[GraphQLError] = []
    for source, target in items:
        source_coordinate = _coordinate_from_value(source)
        target_coordinate = _coordinate_from_value(target)
        if source_coordinate is None or target_coordinate is None:
            errors.append(
                GraphQLError(
                    "Field mappings must be object or interface field coordinates,"
                    f" received {source!r} -> {target!r}."
                )
            )
            continue

        source_error = _validate_mapping_side(old_schema, source_coordinate)
        target_error = _validate_mapping_side(
            new_schema, target_coordinate, allow_missing_field=True
        )
        if source_error:
            errors.append(source_error)
        if target_error:
            errors.append(target_error)
        if source_error or target_error:
            continue
        mappings[source_coordinate] = target_coordinate
    return mappings, errors


def _coordinate_from_value(
    value: str | FieldCoordinate | tuple[str, str],
) -> FieldCoordinate | None:
    if isinstance(value, FieldCoordinate):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        parent_type, field_name = value
        if isinstance(parent_type, str) and isinstance(field_name, str):
            return FieldCoordinate(parent_type, field_name)
        return None
    if not isinstance(value, str):
        return None
    try:
        coordinate_ast = parse_schema_coordinate(value)
    except GraphQLSyntaxError:
        return None
    if not isinstance(coordinate_ast, MemberCoordinateNode):
        return None
    return FieldCoordinate(
        coordinate_ast.name.value, coordinate_ast.member_name.value
    )


def _validate_mapping_side(
    schema: GraphQLSchema,
    coordinate: FieldCoordinate,
    allow_missing_field: bool = False,
) -> GraphQLError | None:
    try:
        resolved = resolve_schema_coordinate(schema, str(coordinate))
    except TypeError as error:
        return GraphQLError(str(error))
    if resolved is None and allow_missing_field:
        return None
    if not isinstance(resolved, ResolvedField):
        return GraphQLError(f"Schema coordinate '{coordinate}' is not an output field.")
    return None


def _collect_occurrences(
    schema: GraphQLSchema, document_ast: DocumentNode
) -> dict[int, _FieldOccurrence]:
    fragments = {
        definition.name.value: definition
        for definition in document_ast.definitions
        if isinstance(definition, FragmentDefinitionNode)
    }
    occurrences: dict[int, _FieldOccurrence] = {}

    for definition in document_ast.definitions:
        if not isinstance(definition, OperationDefinitionNode):
            continue
        root_type = schema.get_root_type(definition.operation)
        if is_object_type(root_type):
            _expand_selection_set(
                schema,
                definition.selection_set,
                root_type,
                {root_type},
                definition,
                (),
                fragments,
                occurrences,
                set(),
            )
    return occurrences


def _expand_selection_set(
    schema: GraphQLSchema,
    selection_set: SelectionSetNode,
    parent_type: GraphQLCompositeType,
    runtime_types: set[GraphQLObjectType],
    operation: OperationDefinitionNode,
    response_path: tuple[str, ...],
    fragments: dict[str, FragmentDefinitionNode],
    occurrences: dict[int, _FieldOccurrence],
    active_fragments: set[str],
) -> None:
    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            field_def = schema.get_field(parent_type, selection.name.value)
            response_key = (
                selection.alias.value if selection.alias else selection.name.value
            )
            occurrence = occurrences.get(id(selection))
            if occurrence is None:
                occurrences[id(selection)] = _FieldOccurrence(
                    selection,
                    parent_type,
                    [_FieldUsage(operation, response_path, set(runtime_types))],
                )
            else:
                occurrence.usages.append(
                    _FieldUsage(operation, response_path, set(runtime_types))
                )

            if field_def is not None:
                field_type = get_named_type(field_def.type)
                if is_composite_type(field_type):
                    child_runtime = _runtime_types(schema, field_type)
                    if selection.selection_set is not None:
                        _expand_selection_set(
                            schema,
                            selection.selection_set,
                            field_type,
                            child_runtime,
                            operation,
                            (*response_path, response_key),
                            fragments,
                            occurrences,
                            active_fragments,
                        )
        elif isinstance(selection, FragmentSpreadNode):
            fragment_name = selection.name.value
            fragment = fragments.get(fragment_name)
            if fragment is None or fragment_name in active_fragments:
                continue
            condition_type = schema.get_type(fragment.type_condition.name.value)
            if not is_composite_type(condition_type):
                continue
            fragment_runtime = _intersect_runtime(
                schema, runtime_types, condition_type
            )
            if not fragment_runtime:
                continue
            _expand_selection_set(
                schema,
                fragment.selection_set,
                condition_type,
                fragment_runtime,
                operation,
                response_path,
                fragments,
                occurrences,
                active_fragments | {fragment_name},
            )
        elif isinstance(selection, InlineFragmentNode):
            condition_type = parent_type
            fragment_runtime = runtime_types
            if selection.type_condition is not None:
                conditional_type = schema.get_type(
                    selection.type_condition.name.value
                )
                if not is_composite_type(conditional_type):
                    continue
                condition_type = conditional_type
                fragment_runtime = _intersect_runtime(
                    schema, runtime_types, condition_type
                )
                if not fragment_runtime:
                    continue
            _expand_selection_set(
                schema,
                selection.selection_set,
                condition_type,
                fragment_runtime,
                operation,
                response_path,
                fragments,
                occurrences,
                active_fragments,
            )


def _runtime_types(
    schema: GraphQLSchema, type_: GraphQLCompositeType
) -> set[GraphQLObjectType]:
    if is_object_type(type_):
        return {type_}
    if is_abstract_type(type_):
        return set(schema.get_possible_types(type_))
    return set()


def _intersect_runtime(
    schema: GraphQLSchema,
    runtime_types: set[GraphQLObjectType],
    condition_type: GraphQLCompositeType,
) -> set[GraphQLObjectType]:
    condition_runtime = _runtime_types(schema, condition_type)
    return runtime_types.intersection(condition_runtime)


def _resolve_target(
    occurrence: _FieldOccurrence,
    mappings: dict[FieldCoordinate, FieldCoordinate],
    new_schema: GraphQLSchema,
    errors: list[GraphQLError],
    used_sources: set[FieldCoordinate],
) -> str | None:
    parent_type = occurrence.parent_type
    field_name = occurrence.node.name.value
    source = FieldCoordinate(parent_type.name, field_name)

    if is_object_type(parent_type):
        target = mappings.get(source)
        if target is None:
            return None
        used_sources.add(source)
        if target.parent_type != parent_type.name:
            errors.append(_static_target_error(occurrence.node, source, target))
            return None
        if _get_mapping_target_field(new_schema, target) is None:
            errors.append(
                GraphQLError(
                    f"Target field '{target}' does not exist.",
                    occurrence.node,
                )
            )
            return None
        return target.field_name

    target = mappings.get(source)
    if target is not None:
        used_sources.add(source)
        if target.parent_type != parent_type.name:
            errors.append(_static_target_error(occurrence.node, source, target))
            return None
        if _get_mapping_target_field(new_schema, target) is None:
            errors.append(
                GraphQLError(
                    f"Target field '{target}' does not exist.",
                    occurrence.node,
                )
            )
            return None
        return target.field_name

    concrete_targets: dict[str, FieldCoordinate] = {}
    for runtime_type in sorted(occurrence.runtime_types, key=lambda type_: type_.name):
        concrete_source = FieldCoordinate(runtime_type.name, field_name)
        concrete_target = mappings.get(concrete_source)
        if concrete_target is not None:
            used_sources.add(concrete_source)
            concrete_targets[runtime_type.name] = concrete_target

    if not concrete_targets:
        return None

    for runtime_type in sorted(occurrence.runtime_types, key=lambda type_: type_.name):
        if runtime_type.name not in concrete_targets:
            errors.append(
                GraphQLError(
                    f"Cannot migrate field '{source}' on abstract type"
                    f" '{parent_type.name}': implementation"
                    f" '{runtime_type.name}.{field_name}' has no mapping.",
                    occurrence.node,
                )
            )

    if len(concrete_targets) != len(occurrence.runtime_types):
        return None

    expected_target = next(iter(concrete_targets.values()))
    for runtime_name, concrete_target in concrete_targets.items():
        if concrete_target != expected_target:
            errors.append(
                GraphQLError(
                    f"Cannot migrate field '{source}': implementations have"
                    " incompatible field mappings"
                    f" ({runtime_name}.{field_name} -> {concrete_target} differs"
                    f" from {expected_target}).",
                    occurrence.node,
                )
            )
            return None

    if expected_target.parent_type != parent_type.name:
        errors.append(
            GraphQLError(
                f"Cannot migrate field '{source}' on abstract type"
                f" '{parent_type.name}': target field '{expected_target}' is not"
                f" declared on '{parent_type.name}'.",
                occurrence.node,
            )
        )
        return None
    if _get_mapping_target_field(new_schema, expected_target) is None:
        errors.append(
            GraphQLError(
                f"Target field '{expected_target}' does not exist.",
                occurrence.node,
            )
        )
        return None
    return expected_target.field_name


def _get_mapping_target_field(
    schema: GraphQLSchema, target: FieldCoordinate
) -> GraphQLField | None:
    parent_type = schema.get_type(target.parent_type)
    if not is_composite_type(parent_type):
        return None
    return schema.get_field(parent_type, target.field_name)


def _static_target_error(
    node: FieldNode, source: FieldCoordinate, target: FieldCoordinate
) -> GraphQLError:
    return GraphQLError(
        f"Cannot migrate field '{source}' to '{target}': the target field's parent"
        f" type must be '{source.parent_type}'.",
        node,
    )


def _detect_field_conflicts(
    occurrences: dict[int, _FieldOccurrence],
    renames: dict[int, str],
    errors: list[GraphQLError],
) -> None:
    records_by_target: dict[
        tuple[int, tuple[str, ...], str],
        list[tuple[_FieldOccurrence, _FieldUsage, str]],
    ] = {}
    for occurrence in occurrences.values():
        target_name = renames.get(id(occurrence.node), occurrence.node.name.value)
        for usage in occurrence.usages:
            key = (
                id(usage.operation),
                usage.path,
                target_name,
            )
            records_by_target.setdefault(key, []).append(
                (occurrence, usage, target_name)
            )

    reported: set[tuple[int, int]] = set()
    for records in records_by_target.values():
        for index, (occurrence, usage, target_name) in enumerate(records):
            for other_occurrence, other_usage, _ in records[index + 1 :]:
                if other_occurrence.node is occurrence.node:
                    continue
                if not usage.runtime_types.intersection(other_usage.runtime_types):
                    continue
                moved = (
                    id(occurrence.node) in renames
                    or id(other_occurrence.node) in renames
                )
                if not moved:
                    continue
                response_key = (
                    occurrence.node.alias.value
                    if occurrence.node.alias
                    else occurrence.node.name.value
                )
                other_response_key = (
                    other_occurrence.node.alias.value
                    if other_occurrence.node.alias
                    else other_occurrence.node.name.value
                )
                if response_key == other_response_key:
                    continue
                node_ids = tuple(
                    sorted((id(occurrence.node), id(other_occurrence.node)))
                )
                if node_ids in reported:
                    continue
                reported.add(node_ids)
                errors.append(
                    GraphQLError(
                        f"Migration of field"
                        f" '{occurrence.parent_type.name}.{occurrence.node.name.value}'"
                        f" conflicts with an existing selection of target field"
                        f" '{occurrence.parent_type.name}.{target_name}'.",
                        [occurrence.node, other_occurrence.node],
                    )
                )


class _FieldRenameVisitor(Visitor):
    def __init__(self, renames: dict[int, str]) -> None:
        super().__init__()
        self.renames = renames
        self.current_targets: list[str | None] = []

    def enter_field(self, node: FieldNode, *_args) -> None:
        self.current_targets.append(self.renames.get(id(node)))

    def leave_field(self, node: FieldNode, *_args) -> FieldNode | None:
        target_name = self.current_targets.pop()
        if target_name is None or target_name == node.name.value:
            return None

        original_name = node.name.value
        alias = node.alias or replace(node.name)
        return replace(
            node,
            name=NameNode(value=target_name),
            alias=alias,
        )


def _error_sort_key(error: GraphQLError) -> tuple[int, int, str]:
    if error.locations:
        location = error.locations[0]
        return location.line, location.column, error.message
    return 10**9, 10**9, error.message
