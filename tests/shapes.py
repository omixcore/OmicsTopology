"""Convenience re-export of the shape generators used by the test suite.

The canonical implementations live in :mod:`omictopo.datasets` so that users
get the same shapes the test suite validates against.
"""

from omictopo.datasets import blob, circle, torus, two_clusters

__all__ = ["circle", "two_clusters", "blob", "torus"]
