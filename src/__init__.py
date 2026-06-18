"""attribution-fuzzy-reconciliation — fuzzy information-fusion for attribution credit.

Package layout:
    synth/        synthetic data generator (the ONLY place ground truth lives)
    methods/      the fuzzy reconciliation pipeline
    baselines/    competing aggregation methods
    eval/         recovery metrics + uncertainty calibration
    experiments/  run scripts

See EXECUTION_PLAN.md for the design, guardrails, and phase gates.
"""

__version__ = "0.0.0"
