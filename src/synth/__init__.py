"""Synthetic data generator (Phase 2).

FIREWALL (plan §4.0): code in this package treats src/methods as a black box and
must NOT share any assumption module with it. Ground truth is generated from a
latent-incrementality process the method never sees. This separation is what makes
the synthetic validation non-circular.
"""
