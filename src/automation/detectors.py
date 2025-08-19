from __future__ import annotations

from typing import Optional, Tuple, Iterable, List

import cv2
import numpy as np


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

# ---------------- Color heuristics -----------------

def find_white_on_blue_candidate(rgb_roi: np.ndarray) -> Optional[Tuple[int, int, float]]:
	hsv = _rgb_to_hsv(rgb_roi)
	blue_mask = cv2.inRange(hsv, (100, 80, 60), (140, 255, 255))
	white_mask = cv2.inRange(hsv, (0, 0, 220), (179, 40, 255))
	blue_float = (blue_mask > 0).astype(np.float32)
	kernel_size = max(7, int(min(rgb_roi.shape[0], rgb_roi.shape[1]) * 0.07) | 1)
	blue_density = cv2.blur(blue_float, (kernel_size, kernel_size))
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((white_mask > 0).astype(np.uint8), connectivity=8)
	best_score = 0.0
	best_xy: Optional[Tuple[int, int]] = None
	for label in range(1, num_labels):
		area = int(stats[label, cv2.CC_STAT_AREA])
		if area < 50:
			continue
		cx_f, cy_f = centroids[label]
		cx = int(round(cx_f))
		cy = int(round(cy_f))
		cy0 = max(0, cy - kernel_size // 2)
		cy1 = min(blue_density.shape[0], cy + kernel_size // 2 + 1)
		cx0 = max(0, cx - kernel_size // 2)
		cx1 = min(blue_density.shape[1], cx + kernel_size // 2 + 1)
		local_blue = blue_density[cy0:cy1, cx0:cx1]
		if local_blue.size == 0:
			continue
		blue_mean = float(np.mean(local_blue))
		score = blue_mean * min(1.0, np.log1p(area) / 6.0)
		if score > best_score:
			best_score = score
			best_xy = (cx, cy)
	if best_xy is None:
		return None
	return (best_xy[0], best_xy[1], best_score)


def find_white_in_red_circle_candidate(rgb_roi: np.ndarray) -> Optional[Tuple[int, int, float]]:
	hsv = _rgb_to_hsv(rgb_roi)
	mask1 = cv2.inRange(hsv, (0, 140, 80), (10, 255, 255))
	mask2 = cv2.inRange(hsv, (160, 140, 80), (179, 255, 255))
	red_mask = cv2.bitwise_or(mask1, mask2)
	red_mask = cv2.medianBlur(red_mask, 5)
	red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	white_mask = cv2.inRange(hsv, (0, 0, 230), (179, 30, 255))
	contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	best_score = 0.0
	best_xy: Optional[Tuple[int, int]] = None
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < 40:
			continue
		perim = cv2.arcLength(cnt, True)
		if perim <= 0:
			continue
		circularity = float(4 * np.pi * area / (perim * perim))
		if circularity < 0.6:
			continue
		(x, y), radius = cv2.minEnclosingCircle(cnt)
		cx = int(round(x))
		cy = int(round(y))
		r = int(round(radius))
		if r <= 3:
			continue
		mask = np.zeros_like(red_mask)
		cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
		region_area = int(np.count_nonzero(mask))
		if region_area <= 0:
			continue
		white_in_region = cv2.bitwise_and(white_mask, white_mask, mask=mask)
		white_area = int(np.count_nonzero(white_in_region))
		white_ratio = white_area / float(max(1, region_area))
		if white_ratio < 0.03:
			continue
		score = circularity * min(1.0, np.log1p(area) / 6.0) * min(1.0, white_ratio / 0.2)
		if score > best_score:
			best_score = score
			best_xy = (cx, cy)
	if best_xy is None:
		return None
	return (best_xy[0], best_xy[1], best_score)

# ---------------- Template matching -----------------

def match_template_rgb(rgb_image: np.ndarray, template_rgb: np.ndarray) -> Tuple[Tuple[int, int], float]:
	result = cv2.matchTemplate(rgb_image, template_rgb, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	return max_loc, float(max_val)


def _build_red_white_mask_from_template(template_rgb: np.ndarray) -> np.ndarray:
	hsv = _rgb_to_hsv(template_rgb)
	mask_red1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
	mask_red2 = cv2.inRange(hsv, (160, 120, 80), (179, 255, 255))
	mask_red = cv2.bitwise_or(mask_red1, mask_red2)
	mask_white = cv2.inRange(hsv, (0, 0, 220), (179, 40, 255))
	mask = cv2.bitwise_or(mask_red, mask_white)
	mask = cv2.medianBlur(mask, 3)
	return mask


def match_template_multiscale_masked(rgb_image: np.ndarray, template_rgb: np.ndarray, scales: Iterable[float]) -> Tuple[Tuple[int, int], float, float, Tuple[int, int]]:
	best_score = -1.0
	best_loc = (0, 0)
	best_scale = 1.0
	best_size = (template_rgb.shape[1], template_rgb.shape[0])
	base_mask = _build_red_white_mask_from_template(template_rgb)
	for s in scales:
		if s <= 0:
			continue
		new_w = max(1, int(round(template_rgb.shape[1] * s)))
		new_h = max(1, int(round(template_rgb.shape[0] * s)))
		resized = cv2.resize(template_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
		mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
		if new_w > rgb_image.shape[1] or new_h > rgb_image.shape[0]:
			continue
		res = cv2.matchTemplate(rgb_image, resized, cv2.TM_CCORR_NORMED, mask=mask)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		score = float(max_val)
		if score > best_score:
			best_score = score
			best_loc = max_loc
			best_scale = s
			best_size = (new_w, new_h)
	return best_loc, float(best_score), float(best_scale), best_size


def match_template_multiscale(rgb_image: np.ndarray, template_rgb: np.ndarray, scales: Iterable[float]) -> Tuple[Tuple[int, int], float, float, Tuple[int, int]]:
	best_score = -1.0
	best_loc = (0, 0)
	best_scale = 1.0
	best_size = (template_rgb.shape[1], template_rgb.shape[0])
	for s in scales:
		if s <= 0:
			continue
		new_w = max(1, int(round(template_rgb.shape[1] * s)))
		new_h = max(1, int(round(template_rgb.shape[0] * s)))
		resized = cv2.resize(template_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
		if new_w > rgb_image.shape[1] or new_h > rgb_image.shape[0]:
			continue
		loc, score = match_template_rgb(rgb_image, resized)
		if score > best_score:
			best_score = score
			best_loc = loc
			best_scale = s
			best_size = (new_w, new_h)
	return best_loc, float(best_score), float(best_scale), best_size


def find_template_instances_multiscale_masked(rgb_image: np.ndarray, template_rgb: np.ndarray, scales: Iterable[float], thresh: float, nms_radius: int = 16) -> List[Tuple[int, int, float, float]]:
	"""Return list of (cx, cy, score, scale) in image coordinates using masked template matching across scales.
	Simple NMS used to suppress duplicates within nms_radius.
	"""
	candidates: List[Tuple[int, int, float, float]] = []
	base_mask = _build_red_white_mask_from_template(template_rgb)
	for s in scales:
		if s <= 0:
			continue
		new_w = max(1, int(round(template_rgb.shape[1] * s)))
		new_h = max(1, int(round(template_rgb.shape[0] * s)))
		resized = cv2.resize(template_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
		mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
		if new_w > rgb_image.shape[1] or new_h > rgb_image.shape[0]:
			continue
		res = cv2.matchTemplate(rgb_image, resized, cv2.TM_CCORR_NORMED, mask=mask)
		ys, xs = np.where(res >= thresh)
		for (px, py) in zip(xs.tolist(), ys.tolist()):
			cx = px + new_w // 2
			cy = py + new_h // 2
			score = float(res[py, px])
			candidates.append((cx, cy, score, float(s)))
	# NMS
	candidates.sort(key=lambda t: t[2], reverse=True)
	picked: List[Tuple[int, int, float, float]] = []
	for cx, cy, score, scale in candidates:
		good = True
		for pcx, pcy, _, _ in picked:
			if (cx - pcx) * (cx - pcx) + (cy - pcy) * (cy - pcy) <= nms_radius * nms_radius:
				good = False
				break
		if good:
			picked.append((cx, cy, score, scale))
	return picked
