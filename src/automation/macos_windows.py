from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from AppKit import NSWorkspace
from Quartz import (
	CGWindowListCopyWindowInfo,
	kCGWindowListOptionOnScreenOnly,
	kCGNullWindowID,
	CGMainDisplayID,
	CGDisplayBounds,
	CGDisplayPixelsWide,
	CGDisplayPixelsHigh,
)


def list_on_screen_windows() -> List[Dict]:
	options = kCGWindowListOptionOnScreenOnly
	window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
	return list(window_list or [])


def find_windows_by_title_substring(substr: str) -> List[Dict]:
	s = substr.lower()
	results = []
	for w in list_on_screen_windows():
		title = (w.get('kCGWindowName') or '')
		if isinstance(title, str) and s in title.lower():
			results.append(w)
	return results


def get_window_bounds(window: Dict) -> Tuple[int, int, int, int]:
	# Returns (x, y, width, height) as reported by CGWindow (may be points on Retina)
	bounds = window.get('kCGWindowBounds') or {}
	x = int(bounds.get('X', 0))
	y = int(bounds.get('Y', 0))
	w = int(bounds.get('Width', 0))
	h = int(bounds.get('Height', 0))
	return (x, y, w, h)


def get_main_display_scale() -> float:
	did = CGMainDisplayID()
	bounds = CGDisplayBounds(did)
	w_points = float(bounds.size.width)
	w_pixels = float(CGDisplayPixelsWide(did))
	if w_points <= 0:
		return 1.0
	return max(1.0, w_pixels / w_points)


def scale_rect_to_pixels(rect: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
	x, y, w, h = rect
	return (int(round(x * scale)), int(round(y * scale)), int(round(w * scale)), int(round(h * scale)))


def find_first_window_bounds_by_title(substr: str) -> Optional[Tuple[int, int, int, int]]:
	wins = find_windows_by_title_substring(substr)
	if not wins:
		return None
	return get_window_bounds(wins[0])


def find_first_window_bounds_by_title_pixels(substr: str) -> Optional[Tuple[int, int, int, int]]:
	rect = find_first_window_bounds_by_title(substr)
	if rect is None:
		return None
	scale = get_main_display_scale()
	return scale_rect_to_pixels(rect, scale)
