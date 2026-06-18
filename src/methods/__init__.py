"""The fuzzy reconciliation pipeline (plan §1, §5.1).

Pipeline order:
    fuzzify -> conflict -> capacity -> choquet -> mamdani -> reconcile

Governing invariant (plan §1.5): independent sources agreeing raises confidence;
redundant/correlated sources agreeing does NOT. The Mamdani confidence is driven by
interaction-corrected agreement, never raw agreement.

FIREWALL (plan §4.0): this package must NOT import from src.synth, nor share any
assumption module with it.
"""
