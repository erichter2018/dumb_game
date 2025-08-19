from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import mss
import numpy as np


@dataclass
class Frame:
	image_bgra: np.ndarray  # HxWx4 uint8
	timestamp_ns: int

	def to_rgb(self) -> np.ndarray:
		# Convert BGRA -> RGB (drop alpha)
		return self.image_bgra[:, :, :3][:, :, ::-1].copy()


class RoiCapture:
	def __init__(self, left: int, top: int, width: int, height: int):
		self._monitor = {"left": left, "top": top, "width": width, "height": height}
		self._sct: Optional[mss.mss] = None

	def __enter__(self) -> "RoiCapture":
		self._sct = mss.mss()
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		if self._sct is not None:
			self._sct.close()
			self._sct = None

	def grab(self) -> Frame:
		if self._sct is None:
			raise RuntimeError("RoiCapture must be used as a context manager or explicitly opened")
		rs = self._sct.grab(self._monitor)
		img = np.array(rs, dtype=np.uint8)  # BGRA
		return Frame(image_bgra=img, timestamp_ns=time.time_ns())

	def stream(self, target_fps: int):
		frame_interval_s = 1.0 / max(1, target_fps)
		next_time = time.perf_counter()
		while True:
			yield self.grab()
			next_time += frame_interval_s
			sleep_s = next_time - time.perf_counter()
			if sleep_s > 0:
				time.sleep(sleep_s)

