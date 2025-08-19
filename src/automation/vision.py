from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


def compute_mean_brightness(rgb_image: np.ndarray) -> float:
	gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
	return float(np.mean(gray))


def match_template_rgb(rgb_image: np.ndarray, template_rgb: np.ndarray, method: int = cv2.TM_CCOEFF_NORMED) -> Tuple[Tuple[int, int], float]:
	result = cv2.matchTemplate(rgb_image, template_rgb, method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
		score = 1.0 - float(min_val)
		location = min_loc
	else:
		score = float(max_val)
		location = max_loc
	return location, score


def load_png_rgb(path: str) -> Optional[np.ndarray]:
	if not os.path.exists(path):
		return None
	bgr = cv2.imread(path, cv2.IMREAD_COLOR)
	if bgr is None:
		return None
	return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def auto_template_from_on_off(on_rgb: np.ndarray, off_rgb: np.ndarray, erosion: int = 1, pad: int = 2) -> Optional[np.ndarray]:
	if on_rgb.shape != off_rgb.shape:
		return None
	diff = cv2.absdiff(on_rgb, off_rgb)
	gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	_, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
	if erosion > 0:
		mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8))
	# Find bounding box of change
	coords = cv2.findNonZero(mask)
	if coords is None:
		return None
	x, y, w, h = cv2.boundingRect(coords)
	x0 = max(0, x - pad)
	y0 = max(0, y - pad)
	x1 = min(on_rgb.shape[1], x + w + pad)
	y1 = min(on_rgb.shape[0], y + h + pad)
	return on_rgb[y0:y1, x0:x1].copy()


def find_color_mask(rgb_image: np.ndarray, lower_rgb: Tuple[int, int, int], upper_rgb: Tuple[int, int, int]) -> np.ndarray:
	lower = np.array(lower_rgb, dtype=np.uint8)
	upper = np.array(upper_rgb, dtype=np.uint8)
	mask = cv2.inRange(rgb_image, lower, upper)
	return mask

