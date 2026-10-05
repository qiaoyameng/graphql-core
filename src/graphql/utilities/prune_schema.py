"""Pruning a GraphQL schema to the parts used by a set of operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ..language import (
    ArgumentNode,
    DocumentNode,
    FieldNode,
    OperationType,
    Visitor,
    visit,
)
from ..pyutils import natural_comparison_key
from ..type import (
    GraphQLArgument,
    GraphQLBoolean,
    GraphQLDirective,
    GraphQLEnumType,
    GraphQLField,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNamedType,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLType,
    GraphQLUnionType,
    get_named_type,
    is_input_object_type,
    is_introspection_type,
    is_interface_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
    is_specified_directive,
    is_specified_scalar_type,
    is_union_type,
)
from .type_info import TypeInfo, TypeInfoVisitor

if TYPE_CHECKING:
    from collections.abc import Collection

    from ..type import GraphQLSchemaKwargs

__all__ = ["PrunedSchema", "PruneSchemaReport", "prune_schema"]


class PrunedSchema(NamedTuple):
    """Result of pruning a schema."""

    schema: GraphQLSchema
    """The new, independently usable schema that only contains used parts."""

    report: "PruneSchemaReport"
    """A report describing what was removed and what had to be retained."""


class PruneSchemaReport(NamedTuple):
    """A report of a schema pruning run.

    The report explains both the parts that were removed and the parts that were
    retained beyond the literal selections in the operations, together with the
    reason why they were necessary to keep the schema valid and the operations
    executable.
    """

    removed_types: tuple[str, ...]
    """User defined types that were removed because the operations do not use them."""

    removed_root_fields: tuple[str, ...]
    """Coordinates (``TypeName.fieldName``) of removed fields on root operation
    types."""

    removed_fields: tuple[str, ...]
    """Coordinates (``TypeName.fieldName``) of removed fields on non-root types."""

    removed_arguments: tuple[str, ...]
    """Coordinates (``TypeName.fieldName(argName:)`` or ``@directive(argName:)``)
    of removed arguments."""

    removed_directives: tuple[str, ...]
    """Directives removed because the operations do not use them."""

    notes: tuple[str, ...]
    """Human readable explanations of parts retained or synthesized for validity."""


def prune_schema(
    schema: GraphQLSchema,
    document_ast: DocumentNode | Collection[DocumentNode],
) -> PrunedSchema:
    """Prune a schema to the parts required by one or more validated operations.

    This creates a new, independently usable schema that contains exactly the root
    fields, fields, arguments, directives and types needed by the given operations,
    as well as everything required to keep the schema valid and the operations
    executable: complete input object field contracts, interface implementations,
    union members, recursive type cycles and types referenced by default values.

    The given documents are validated against the original schema with the
    specified validation rules; a :class:`TypeError` is raised when they are not
    valid. The executor is not modified and no resolvers are invoked.

    :param schema: The original schema to prune.
    :param document_ast: A validated operation document, or a collection of them.
    :returns: A :class:`PrunedSchema` with the new schema and a pruning report.

    >>> from graphql import build_schema, parse, print_schema, prune_schema, validate
    >>> schema = build_schema('''
    ...     type Query {
    ...       wanted: String
    ...       unwanted: Int
    ...     }
    ... ''')
    >>> document = parse('{ wanted }')
    >>> result = prune_schema(schema, document)
    >>> assert validate(result.schema, document) == []
    >>> print(print_schema(result.schema))
    type Query {
      wanted: String
    }
    >>> result.report.removed_root_fields
    ('Query.unwanted',)
    """
    documents = _as_documents(document_ast)

    # Imported lazily to avoid an import cycle (validation imports utilities).
    from ..validation import validate

    # The operations must be valid against the original schema.
    validation_errors = [
        error for document in documents for error in validate(schema, document)
    ]
    if validation_errors:
        messages = "\n\n".join(error.message for error in validation_errors)
        msg = (
            "Cannot prune schema with invalid operations:\n\n"
            f"{messages}\n\nFix the operations before pruning the schema."
        )
        raise TypeError(msg)

    usage = _UsageCollector()
    for document in documents:
        type_info = TypeInfo(schema)
        usage.type_info = type_info
        visit(document, TypeInfoVisitor(type_info, usage))

    context = _PruneContext(schema, usage)
    context.compute_closure()
    return context.build()


def _as_documents(
    document_ast: DocumentNode | Collection[DocumentNode],
) -> tuple[DocumentNode, ...]:
    """Normalize a document or a collection of documents to a tuple."""
    if isinstance(document_ast, DocumentNode):
        return (document_ast,)
    return tuple(document_ast)


class _UsageCollector(Visitor):
    """Collect the schema elements used by a set of validated operations."""

    def __init__(self) -> None:
        super().__init__()
        self.type_info: TypeInfo | None = None
        self.types: set[str] = set()
        """Named types referenced anywhere (root types, fields, variables, ...)."""

        self.fields: dict[str, set[str]] = {}
        """Field names selected on each object or interface type."""

        self.field_arguments: dict[tuple[str, str], set[str]] = {}
        """Argument names passed to each ``(TypeName, fieldName)`` selection."""

        self.directives: set[str] = set()
        """Custom directives used anywhere in the operations."""

        self.directive_arguments: dict[str, set[str]] = {}
        """Argument names passed to each used custom directive."""

        self._field_stack: list[str] = []
        self._skip_stack: list[bool] = []

    def enter_operation_definition(self, *_args: Any) -> None:
        root_type = self._type_info.get_parent_type()
        if root_type is not None:
            self.types.add(root_type.name)

    def enter_fragment_definition(self, *_args: Any) -> None:
        parent_type = self._type_info.get_parent_type()
        if parent_type is not None:
            self.types.add(parent_type.name)

    def enter_inline_fragment(self, *_args: Any) -> None:
        parent_type = self._type_info.get_parent_type()
        if parent_type is not None:
            self.types.add(parent_type.name)

    def enter_variable_definition(self, *_args: Any) -> None:
        input_type = self._type_info.get_input_type()
        if input_type is not None:
            self.types.add(get_named_type(input_type).name)

    def enter_field(self, node: FieldNode, *_args: Any) -> None:
        field_name = node.name.value
        parent_type = self._type_info.get_parent_type()
        field_def = self._type_info.get_field_def()
        if parent_type is not None:
            self.types.add(parent_type.name)
        if (
            parent_type is None
            or field_def is None
            or field_name.startswith("__")
        ):
            self._skip_stack.append(True)
            return
        self._skip_stack.append(False)
        parent_name = parent_type.name
        self.fields.setdefault(parent_name, set()).add(field_name)
        self.types.add(get_named_type(field_def.type).name)
        self._field_stack.append(field_name)

    def leave_field(self, *_args: Any) -> None:
        skipped = self._skip_stack.pop()
        if not skipped:
            self._field_stack.pop()

    def enter_argument(self, node: ArgumentNode, *_args: Any) -> None:
        argument_name = node.name.value
        argument = self._type_info.get_argument()
        if argument is not None:
            self.types.add(get_named_type(argument.type).name)
        directive = self._type_info.get_directive()
        if directive is not None:
            if not is_specified_directive(directive):
                self.directive_arguments.setdefault(
                    directive.name, set()
                ).add(argument_name)
            return
        field_name = self._field_stack[-1] if self._field_stack else ""
        parent_type = self._type_info.get_parent_type()
        if field_name and parent_type is not None:
            self.field_arguments.setdefault(
                (parent_type.name, field_name), set()
            ).add(argument_name)

    def enter_directive(self, *_args: Any) -> None:
        directive = self._type_info.get_directive()
        if directive is not None and not is_specified_directive(directive):
            self.directives.add(directive.name)

    @property
    def _type_info(self) -> TypeInfo:
        if self.type_info is None:  # pragma: no cover
            msg = "Usage collector used without a TypeInfo."
            raise RuntimeError(msg)
        return self.type_info


class _PruneContext:
    """Compute the closure of used schema elements and build the pruned schema."""

    def __init__(self, schema: GraphQLSchema, usage: _UsageCollector) -> None:
        self.schema = schema
        self.usage = usage

        self.retained: set[str] = set()
        self.kept_fields: dict[str, set[str]] = {}
        self.kept_arguments: dict[tuple[str, str], set[str]] = {}

        self.anchored: set[str] = set()
        self.synthetic_query = False
        self.synthetic_query_name = "Query"

        self.type_cache: dict[str, GraphQLNamedType] = {}
        self.notes: list[str] = []

    # -- Closure computation -------------------------------------------------

    def _structural_closure(self) -> None:
        """Close over types reachable through the currently retained elements.

        Only retained fields and their retained arguments are traversed, so
        types referenced solely by removed fields or arguments are pruned.
        """
        pending = [
            self.schema.type_map[name]
            for name in self.retained
            if name in self.schema.type_map
        ]

        def referent(named: GraphQLNamedType | None) -> None:
            if (
                named is not None
                and not is_introspection_type(named)
                and not is_specified_scalar_type(named)
                and named.name not in self.retained
            ):
                self.retained.add(named.name)
                pending.append(named)

        while pending:
            type_ = pending.pop()
            if is_object_type(type_) or is_interface_type(type_):
                for interface in type_.interfaces:
                    referent(interface)
                if is_interface_type(type_):
                    impls = self.schema.get_implementations(type_)
                    for object_type in impls.objects:
                        referent(object_type)
                    for interface_type in impls.interfaces:
                        referent(interface_type)
                for field_name in tuple(self.kept_fields.get(type_.name, ())):
                    field = type_.fields.get(field_name)
                    if field is None:
                        continue
                    referent(get_named_type(field.type))
                    kept_arguments = self.kept_arguments.get(
                        (type_.name, field_name)
                    )
                    if kept_arguments is not None:
                        for argument_name in kept_arguments:
                            argument = field.args.get(argument_name)
                            if argument is not None:
                                referent(get_named_type(argument.type))
            elif is_union_type(type_):
                for member in type_.types:
                    referent(member)
            elif is_input_object_type(type_):
                # Input objects keep their complete field contract.
                for field in type_.fields.values():
                    referent(get_named_type(field.type))

    def compute_closure(self) -> None:
        usage = self.usage

        # Everything the visitor referenced (root types, return types, variable
        # types, argument types, ...) starts the closure.
        for name in usage.types:
            type_ = self.schema.get_type(name)
            if (
                type_ is not None
                and not is_introspection_type(type_)
                and not is_specified_scalar_type(type_)
            ):
                self.retained.add(type_.name)

        used_roots: dict[OperationType, GraphQLObjectType | None] = {}
        for operation in OperationType:
            root_type = self.schema.get_root_type(operation)
            if root_type is not None and root_type.name in self.retained:
                used_roots[operation] = root_type

        # Seed the selected fields and the arguments passed to them.
        for name in self.retained:
            self.kept_fields[name] = set(usage.fields.get(name, ()))
        for (type_name, field_name), arguments in usage.field_arguments.items():
            if type_name in self.retained:
                self.kept_arguments[(type_name, field_name)] = set(arguments)

        # Fixpoint: new implementations may need interface fields, and newly
        # retained interface fields add return/argument types and propagate to
        # every implementation.
        while True:
            previous = set(self.retained)
            self._structural_closure()

            changed = self.retained != previous
            for name in tuple(self.retained):
                self.kept_fields.setdefault(name, set())

            # Close interface implementation relationships:
            #  * a retained implementation must provide every retained field of
            #    its interfaces (interface -> implementation), and
            #  * an interface field must declare every argument used by a
            #    retained implementation field (implementation -> interface),
            #    since implementation arguments must be a subset of the
            #    interface field arguments.
            for name in tuple(self.retained):
                type_ = self.schema.get_type(name)
                if not is_object_type(type_) and not is_interface_type(type_):
                    continue
                for interface in type_.interfaces:
                    if interface.name not in self.retained:
                        continue
                    # Interface fields retained on the interface itself must be
                    # provided by every implementation.
                    interface_field_names = tuple(
                        self.kept_fields.get(interface.name, ())
                    )
                    for field_name in interface_field_names:
                        if field_name not in type_.fields:
                            continue
                        fields = self.kept_fields.setdefault(type_.name, set())
                        if field_name not in fields:
                            fields.add(field_name)
                            changed = True
                        arguments = self.kept_arguments.setdefault(
                            (type_.name, field_name), set()
                        )
                        for argument_name in interface.fields[field_name].args:
                            if argument_name in self.kept_arguments.get(
                                (interface.name, field_name), ()
                            ) and argument_name not in arguments:
                                arguments.add(argument_name)
                                changed = True
                    # Arguments used on the implementation side must exist on
                    # the interface field for the schema to remain valid.
                    for field_name in tuple(self.kept_fields.get(type_.name, ())):
                        interface_field = interface.fields.get(field_name)
                        if interface_field is None:
                            continue
                        interface_fields = self.kept_fields.setdefault(
                            interface.name, set()
                        )
                        if field_name not in interface_fields:
                            interface_fields.add(field_name)
                            changed = True
                        interface_arguments = self.kept_arguments.setdefault(
                            (interface.name, field_name), set()
                        )
                        for argument_name in self.kept_arguments.get(
                            (type_.name, field_name), ()
                        ):
                            if argument_name not in interface_arguments:
                                interface_arguments.add(argument_name)
                                changed = True

            self._structural_closure()

            # Composite types must define at least one field; retain a single
            # deterministic anchor field and explain it in the report.
            for name in sorted(self.retained, key=natural_comparison_key):
                type_ = self.schema.get_type(name)
                if not is_object_type(type_) and not is_interface_type(type_):
                    continue
                if self.kept_fields[name] or not type_.fields:
                    continue
                anchor = min(type_.fields, key=natural_comparison_key)
                self.kept_fields[name].add(anchor)
                self.anchored.add(name)
                changed = True

            self._structural_closure()
            for name in tuple(self.retained):
                self.kept_fields.setdefault(name, set())

            if not changed:
                break

        self._ensure_query_root(used_roots)
        # The query root fallback may retain an anchor field whose return and
        # argument types still need to be closed.
        self._structural_closure()

    def _ensure_query_root(
        self, used_roots: dict[OperationType, GraphQLObjectType | None]
    ) -> None:
        query = self.schema.query_type
        if query is not None and query.name in self.retained:
            return
        if used_roots.get(OperationType.QUERY) is not None:
            return
        if not any(
            used_roots.get(operation) is not None
            for operation in (OperationType.MUTATION, OperationType.SUBSCRIPTION)
        ):
            return
        # A schema must define a Query root type; keep a minimal one when only
        # mutations or subscriptions are used.
        if query is not None:
            self.retained.add(query.name)
            self.kept_fields.setdefault(query.name, set())
            if query.fields and not self.kept_fields[query.name]:
                anchor = min(query.fields, key=natural_comparison_key)
                self.kept_fields[query.name].add(anchor)
                self.anchored.add(query.name)
        else:
            name = "Query"
            suffix = 1
            while name in self.schema.type_map:
                suffix += 1
                name = f"Query{suffix}"
            self.synthetic_query_name = name
            self.synthetic_query = True

    # -- Phase 3: build the new schema ----------------------------------------

    def build(self) -> PrunedSchema:
        """Construct the pruned schema and assemble the pruning report."""
        self._collect_notes()

        original_kwargs = self.schema.to_kwargs()

        query = self._build_query_root(original_kwargs)
        mutation = self._build_root(original_kwargs["mutation"])
        subscription = self._build_root(original_kwargs["subscription"])

        types = tuple(
            self._build_named_type(self.schema.type_map[name])
            for name in self.retained
            if name in self.schema.type_map
        )

        directives = self._build_directives(original_kwargs["directives"])

        new_schema = GraphQLSchema(
            query=query,
            mutation=mutation,
            subscription=subscription,
            types=types,
            directives=tuple(directives),
            description=original_kwargs.get("description"),
            extensions=original_kwargs.get("extensions"),
        )
        self._assert_valid(new_schema)
        report = self._build_report()
        return PrunedSchema(new_schema, report)

    def _build_query_root(
        self, original_kwargs: GraphQLSchemaKwargs
    ) -> GraphQLObjectType | None:
        query = original_kwargs["query"]
        if self.synthetic_query:
            return GraphQLObjectType(
                self.synthetic_query_name,
                {"_empty": GraphQLField(GraphQLBoolean)},
                description="Synthetic empty query root retained by prune_schema.",
            )
        if query is not None and query.name in self.retained:
            return self._build_named_type(query)
        return None

    def _build_root(
        self, root: GraphQLObjectType | None
    ) -> GraphQLObjectType | None:
        if root is not None and root.name in self.retained:
            return self._build_named_type(root)
        return None

    def _assert_valid(self, new_schema: GraphQLSchema) -> None:
        from ..type import validate_schema

        errors = validate_schema(new_schema)
        if errors:  # pragma: no cover - defensive guard for the pruner itself
            messages = "\n".join(error.message for error in errors)
            msg = f"prune_schema produced an invalid schema:\n{messages}"
            raise TypeError(msg)

    def _map_type(self, type_: GraphQLType) -> GraphQLType:
        if is_list_type(type_):
            return GraphQLList(self._map_type(type_.of_type))
        if is_non_null_type(type_):
            return GraphQLNonNull(self._map_type(type_.of_type))
        named_type = get_named_type(type_)
        return self._build_named_type(named_type)

    def _build_named_type(self, type_: GraphQLNamedType) -> GraphQLNamedType:
        # Memoized so recursive type cycles resolve to the same built instance.
        built = self.type_cache.get(type_.name)
        if built is not None:
            return built
        if is_specified_scalar_type(type_):
            return type_
        if is_object_type(type_):
            built = self._build_object_type(type_)
        elif is_interface_type(type_):
            built = self._build_interface_type(type_)
        elif is_union_type(type_):
            built = self._build_union_type(type_)
        elif is_input_object_type(type_):
            built = self._build_input_object_type(type_)
        elif type(type_) is GraphQLEnumType:
            built = self._build_enum_type(type_)
        elif type(type_) is GraphQLScalarType:
            built = self._build_scalar_type(type_)
        else:  # pragma: no cover
            built = type_
        self.type_cache[type_.name] = built
        return built

    def _build_scalar_type(self, type_: GraphQLScalarType) -> GraphQLScalarType:
        return GraphQLScalarType(**type_.to_kwargs())

    def _build_enum_type(self, type_: GraphQLEnumType) -> GraphQLEnumType:
        # All enum values are retained: variable values supplied at execution
        # time are not visible in the operation text.
        return GraphQLEnumType(**type_.to_kwargs())

    def _build_object_type(self, type_: GraphQLObjectType) -> GraphQLObjectType:
        kwargs = type_.to_kwargs()

        def fields_thunk():
            return {
                name: self._build_field(type_.name, name, field)
                for name, field in type_.fields.items()
                if name in self.kept_fields.get(type_.name, ())
            }

        def interfaces_thunk():
            return [
                self._build_named_type(interface)
                for interface in type_.interfaces
                if interface.name in self.retained
            ]

        return GraphQLObjectType(
            name=kwargs["name"],
            fields=fields_thunk,
            interfaces=interfaces_thunk,
            is_type_of=kwargs.get("is_type_of"),
            description=kwargs.get("description"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
            extension_ast_nodes=kwargs.get("extension_ast_nodes"),
        )

    def _build_interface_type(
        self, type_: GraphQLInterfaceType
    ) -> GraphQLInterfaceType:
        kwargs = type_.to_kwargs()

        def fields_thunk():
            return {
                name: self._build_field(type_.name, name, field)
                for name, field in type_.fields.items()
                if name in self.kept_fields.get(type_.name, ())
            }

        def interfaces_thunk():
            return [
                self._build_named_type(interface)
                for interface in type_.interfaces
                if interface.name in self.retained
            ]

        return GraphQLInterfaceType(
            name=kwargs["name"],
            fields=fields_thunk,
            interfaces=interfaces_thunk,
            resolve_type=kwargs.get("resolve_type"),
            description=kwargs.get("description"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
            extension_ast_nodes=kwargs.get("extension_ast_nodes"),
        )

    def _build_union_type(self, type_: GraphQLUnionType) -> GraphQLUnionType:
        kwargs = type_.to_kwargs()
        return GraphQLUnionType(
            name=kwargs["name"],
            types=[
                self._build_named_type(member)
                for member in type_.types
                if member.name in self.retained
            ],
            resolve_type=kwargs.get("resolve_type"),
            description=kwargs.get("description"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
            extension_ast_nodes=kwargs.get("extension_ast_nodes"),
        )

    def _build_input_object_type(
        self, type_: GraphQLInputObjectType
    ) -> GraphQLInputObjectType:
        kwargs = type_.to_kwargs()

        def fields_thunk():
            # The complete input field contract is retained, including required
            # fields that never appear in the example variable values.
            return {
                name: self._build_input_field(field)
                for name, field in type_.fields.items()
            }

        return GraphQLInputObjectType(
            name=kwargs["name"],
            fields=fields_thunk,
            description=kwargs.get("description"),
            out_type=kwargs.get("out_type"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
            extension_ast_nodes=kwargs.get("extension_ast_nodes"),
            is_one_of=kwargs.get("is_one_of", False),
        )

    def _build_field(
        self, parent_name: str, field_name: str, field: GraphQLField
    ) -> GraphQLField:
        kwargs = field.to_kwargs()
        kept = self.kept_arguments.get((parent_name, field_name))
        if kept is None:
            args = {}
        else:
            args = {
                name: self._build_argument(argument)
                for name, argument in field.args.items()
                if name in kept
            }

        return GraphQLField(
            type_=self._map_type(kwargs["type_"]),
            args=args,
            resolve=kwargs.get("resolve"),
            subscribe=kwargs.get("subscribe"),
            description=kwargs.get("description"),
            deprecation_reason=kwargs.get("deprecation_reason"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
        )

    def _build_argument(self, argument: GraphQLArgument) -> GraphQLArgument:
        kwargs = argument.to_kwargs()
        return GraphQLArgument(
            type_=self._map_type(kwargs["type_"]),
            default_value=kwargs.get("default_value"),
            default=kwargs.get("default"),
            description=kwargs.get("description"),
            deprecation_reason=kwargs.get("deprecation_reason"),
            out_name=kwargs.get("out_name"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
        )

    def _build_input_field(self, field: GraphQLInputField) -> GraphQLInputField:
        kwargs = field.to_kwargs()
        return GraphQLInputField(
            type_=self._map_type(kwargs["type_"]),
            default_value=kwargs.get("default_value"),
            default=kwargs.get("default"),
            description=kwargs.get("description"),
            deprecation_reason=kwargs.get("deprecation_reason"),
            out_name=kwargs.get("out_name"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
        )

    def _build_directives(
        self, directives: Collection[GraphQLDirective]
    ) -> list[GraphQLDirective]:
        result = []
        for directive in directives:
            if is_specified_directive(directive):
                result.append(directive)
                continue
            if directive.name not in self.usage.directives:
                continue
            result.append(self._build_directive(directive))
        return result

    def _build_directive(
        self, directive: GraphQLDirective
    ) -> GraphQLDirective:
        kwargs = directive.to_kwargs()
        kept = self.usage.directive_arguments.get(directive.name, set())
        return GraphQLDirective(
            name=kwargs["name"],
            locations=kwargs["locations"],
            args={
                name: self._build_argument(argument)
                for name, argument in directive.args.items()
                if name in kept
            },
            is_repeatable=kwargs.get("is_repeatable", False),
            deprecation_reason=kwargs.get("deprecation_reason"),
            description=kwargs.get("description"),
            extensions=kwargs.get("extensions"),
            ast_node=kwargs.get("ast_node"),
            extension_ast_nodes=kwargs.get("extension_ast_nodes"),
        )

    # -- Reporting ------------------------------------------------------------

    def _collect_notes(self) -> None:
        notes = self.notes

        if self.synthetic_query:
            notes.append(
                f"Added a synthetic Query root type '{self.synthetic_query_name}'"
                " with a Boolean field because a GraphQL schema must define a Query"
                " root type even when only mutations or subscriptions are used."
            )

        for name in sorted(self.retained, key=natural_comparison_key):
            type_ = self.schema.get_type(name)
            if type_ is None:
                continue
            if is_object_type(type_) or is_interface_type(type_):
                if name in self.anchored:
                    anchor = min(
                        self.kept_fields.get(name, ()), key=natural_comparison_key
                    )
                    notes.append(
                        f"Retained field '{type_.name}.{anchor}' although it is not"
                        " selected by the operations, because the retained type would"
                        " otherwise define no fields, which is not allowed."
                    )
                if is_interface_type(type_):
                    impls = self.schema.get_implementations(type_)
                    names = sorted(
                        (
                            *(t.name for t in impls.objects),
                            *(t.name for t in impls.interfaces),
                        ),
                        key=natural_comparison_key,
                    )
                    if names:
                        notes.append(
                            f"Retained all implementations of interface"
                            f" '{type_.name}' ({', '.join(names)}) and their interface"
                            " relationships so abstract type resolution and fragment"
                            " conditions stay complete."
                        )
                for interface in type_.interfaces:
                    if interface.name in self.retained and any(
                        argument_name
                        not in self.usage.field_arguments.get(
                            (type_.name, field_name), ()
                        )
                        for field_name in self.kept_fields.get(type_.name, ())
                        for argument_name in self.kept_arguments.get(
                            (type_.name, field_name), ()
                        )
                    ):
                        notes.append(
                            f"Retained interface-required arguments on"
                            f" '{type_.name}' so its implementations remain valid."
                        )
                        break
            elif is_union_type(type_):
                members = [
                    member.name
                    for member in type_.types
                    if member.name in self.retained
                ]
                if members:
                    notes.append(
                        f"Retained all members of union '{type_.name}'"
                        f" ({', '.join(sorted(members, key=natural_comparison_key))})"
                        " so union resolution stays complete."
                    )
            elif is_input_object_type(type_):
                notes.append(
                    f"Retained the complete field contract of input object"
                    f" '{type_.name}', including required fields absent from example"
                    " variables."
                )
            elif type(type_) is GraphQLEnumType:
                notes.append(
                    f"Retained all values of enum '{type_.name}' because variable"
                    " values are supplied at execution time and pruning a value could"
                    " invalidate an operation."
                )

        query = self.schema.query_type
        if (
            not self.synthetic_query
            and query is not None
            and query.name in self.retained
            and not self.usage.fields.get(query.name)
        ):
            notes.append(
                "Retained a minimal Query root type because a GraphQL schema must"
                " define a Query root type."
            )

    def _build_report(self) -> PruneSchemaReport:
        root_names = set()
        for operation in OperationType:
            root = self.schema.get_root_type(operation)
            if root is not None:
                root_names.add(root.name)

        removed_root_fields: list[str] = []
        removed_fields: list[str] = []
        removed_arguments: list[str] = []

        for name in sorted(self.retained, key=natural_comparison_key):
            type_ = self.schema.get_type(name)
            if not is_object_type(type_) and not is_interface_type(type_):
                continue
            kept = self.kept_fields.get(name, set())
            target = removed_root_fields if name in root_names else removed_fields
            for field_name in sorted(type_.fields, key=natural_comparison_key):
                if field_name not in kept:
                    target.append(f"{name}.{field_name}")
            for field_name in sorted(kept, key=natural_comparison_key):
                field = type_.fields.get(field_name)
                if field is None:
                    continue
                kept_arguments = self.kept_arguments.get((name, field_name))
                for argument_name in sorted(field.args, key=natural_comparison_key):
                    if kept_arguments is None or argument_name not in kept_arguments:
                        removed_arguments.append(
                            f"{name}.{field_name}({argument_name}:)"
                        )

        removed_directives = sorted(
            (
                directive.name
                for directive in self.schema.directives
                if not is_specified_directive(directive)
                and directive.name not in self.usage.directives
            ),
            key=natural_comparison_key,
        )

        removed_types = [
            name
            for name, type_ in self.schema.type_map.items()
            if name not in self.retained
            and not is_introspection_type(type_)
            and not is_specified_scalar_type(type_)
        ]

        return PruneSchemaReport(
            removed_types=tuple(removed_types),
            removed_root_fields=tuple(removed_root_fields),
            removed_fields=tuple(removed_fields),
            removed_arguments=tuple(removed_arguments),
            removed_directives=tuple(removed_directives),
            notes=tuple(dict.fromkeys(self.notes)),
        )
