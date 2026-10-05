"""Validation of recorded GraphQL responses without resolvers.

When replaying a recorded GraphQL response, the gateway has no access to the
resolvers of the upstream service. All it has is the schema, the query, the
variables and the response JSON. This package validates that such a response
conforms to the GraphQL contract of the operation:

- response keys follow the selected operation and the active ``@skip`` and
  ``@include`` directives (aliases are honored),
- missing keys, explicit nulls, wrong list nesting and non-null violations are
  reported separately,
- abstract (interface/union) positions are validated conservatively: when the
  response does not carry enough type information to determine the runtime
  type, the branches that could not be decided are reported instead of
  arbitrarily accepting one implementation,
- custom scalar output values are checked with caller provided functions,
- the ``path`` entries of response ``errors`` are cross-checked against the
  request, and the null positions that are allowed because of non-null error
  propagation are computed; a single error never excuses the whole ``data``.
"""

from .validate_response import (
    ResponseValidationIssue,
    ResponseValidationIssueKind,
    ResponseValidationReport,
    ScalarOutputChecker,
    validate_response,
)

__all__ = [
    "ResponseValidationIssue",
    "ResponseValidationIssueKind",
    "ResponseValidationReport",
    "ScalarOutputChecker",
    "validate_response",
]
