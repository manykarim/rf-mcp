"""Benchmark suite for token optimization and performance validation.

This module provides comprehensive benchmarks to validate the performance targets
from the ADR (Architecture Decision Record) for token optimization:

Performance Targets:
- Token reduction: 70-95% on typical pages
- ARIA snapshot generation: <200ms
- Ref lookup: <1ms per operation
- Incremental diff: <50ms
- Pre-validation: <100ms
- Memory per ref: <200 bytes

Run benchmarks with:
    pytest tests/benchmarks/ -v --benchmark-only
    python scripts/run_benchmarks.py
"""
