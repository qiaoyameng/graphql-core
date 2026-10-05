Response Validation
===================

.. currentmodule:: graphql.response_validation

.. automodule:: graphql.response_validation
   :no-members:
   :no-inherited-members:

Validate a recorded response against the schema, the selected operation and
the variables, without running any resolvers:

.. autofunction:: validate_response

.. autoclass:: ResponseValidationReport
   :no-inherited-members:

.. autoclass:: ResponseValidationIssue
   :no-inherited-members:

.. autoclass:: ResponseValidationIssueKind
   :no-inherited-members:

.. autoclass:: ScalarOutputChecker
