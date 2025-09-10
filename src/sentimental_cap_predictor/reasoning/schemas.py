from __future__ import annotations

"""Schemas describing simplified physical constructs used in reasoning tasks."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ContainerSchema:
    """Represents a container with a fixed capacity.

    Parameters
    ----------
    name:
        Identifier for the container.
    capacity:
        Total volume the container can hold.
    volume:
        Current volume contained. Defaults to ``0.0``.
    """

    name: str
    capacity: float
    volume: float = 0.0

    def describe(self) -> str:
        """Return a human readable description of the container."""

        return (
            f"{self.name} holds {self.volume} of {self.capacity} total capacity."
        )

    def remaining(self) -> float:
        """Volume still available in the container."""

        return self.capacity - self.volume

    def fill(self, amount: float) -> None:
        """Add volume to the container without exceeding its capacity."""

        self.volume = min(self.capacity, self.volume + amount)


@dataclass
class PathSchema:
    """Represents a sequence of steps between locations."""

    steps: List[Tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        """Return a human readable description of the path."""

        if not self.steps:
            return "Empty path"
        segments = " -> ".join(f"{start}->{end}" for start, end in self.steps)
        return f"Path: {segments}"

    @classmethod
    def step(cls, start: str, end: str) -> "PathSchema":
        """Create a path containing a single step."""

        return cls(steps=[(start, end)])

    def add_step(self, start: str, end: str) -> None:
        """Append a step to the path."""

        self.steps.append((start, end))


@dataclass
class ForceSchema:
    """Represents a force vector."""

    magnitude: float
    direction: Tuple[float, float, float]
    point: Optional[str] = None

    def describe(self) -> str:
        """Return a human readable description of the force."""

        direction = f"({self.direction[0]}, {self.direction[1]}, {self.direction[2]})"
        location = f" at {self.point}" if self.point else ""
        return f"Force of {self.magnitude}N in direction {direction}{location}"

    @classmethod
    def from_components(
        cls, x: float, y: float, z: float, point: Optional[str] = None
    ) -> "ForceSchema":
        """Construct a force from its directional components."""

        magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        return cls(magnitude=magnitude, direction=(x, y, z), point=point)


@dataclass
class BalanceSchema:
    """Represents a simple two sided balance."""

    left: float = 0.0
    right: float = 0.0

    def describe(self) -> str:
        """Return a human readable description of the balance state."""

        if self.left == self.right:
            status = "balanced"
        elif self.left > self.right:
            status = "left heavy"
        else:
            status = "right heavy"
        return f"Balance: left={self.left}, right={self.right} ({status})"

    def is_balanced(self) -> bool:
        """Determine whether the balance is in equilibrium."""

        return self.left == self.right

    def add_left(self, weight: float) -> None:
        """Add weight to the left side."""

        self.left += weight

    def add_right(self, weight: float) -> None:
        """Add weight to the right side."""

        self.right += weight
