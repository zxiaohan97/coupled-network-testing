"""Common state and test labels used across the package."""

from enum import IntEnum


class TestType(IntEnum):
    """Type of observation collected from a node."""

    SOCIAL = 0
    PHYSICAL = 1


class BinaryState(IntEnum):
    """Binary state label used for physical and social states."""

    ZERO = 0
    ONE = 1
