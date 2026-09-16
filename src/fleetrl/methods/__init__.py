"""Fourteen independently configurable methods with shared physical infrastructure."""

from .registry import METHODS, get_method

__all__ = ["METHODS", "get_method"]
