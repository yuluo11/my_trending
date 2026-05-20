"""Reflection-agent entrypoints.

This package owns post-decision reflection-task shaping, lesson extraction, and
the final reflection-output contract. Context assembly and retrieval stay under
``services.reflection`` so postmortem reasoning and infrastructure remain
separate.
"""

from .base_agent import BaseReflectionAgent, ReflectionRuntimeState, ReflectionTask
from .reflection_agent import ReflectionAgent

__all__ = [
    "BaseReflectionAgent",
    "ReflectionAgent",
    "ReflectionRuntimeState",
    "ReflectionTask",
]
