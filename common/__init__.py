"""Shared measurement harness for the labs.

Built in M0, before any kernel (LEARNING_PATH.md, Stage 1): bench.py
(CUDA-event timing, warm-up discard, median + p5/p95), correctness.py
(dtype-aware tolerances), and results_schema.py (what every results file
records). Everything later trusts the numbers this package produces.
"""
