from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Roi:
	left: int
	top: int
	width: int
	height: int

	@property
	def bbox(self) -> Tuple[int, int, int, int]:
		return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class AppConfig:
	roi: Roi
	target_fps: int
	show_window: bool
	brightness_threshold: int
	demo_auto_click: bool


def getenv_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def getenv_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}

