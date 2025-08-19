from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2

from .capture import RoiCapture
from .vision import compute_mean_brightness
from .input_control import click


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Desktop game automation")
	p.add_argument("--roi", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"), required=True, help="Screen ROI in pixels")
	p.add_argument("--fps", type=int, default=60, help="Target capture FPS")
	p.add_argument("--show", action="store_true", help="Show preview window")
	p.add_argument("--demo-auto-click", action="store_true", help="Demo: click ROI center if brightness exceeds threshold")
	p.add_argument("--brightness-threshold", type=int, default=180, help="Brightness threshold (0-255)")
	return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	left, top, width, height = args.roi
	cx = left + width // 2
	cy = top + height // 2

	print(f"[automation] ROI=({left},{top},{width},{height}) FPS={args.fps} show={args.show}")

	if args.show:
		cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("preview", max(320, width), max(240, height))

	with RoiCapture(left, top, width, height) as cap:
		for frame in cap.stream(args.fps):
			rgb = frame.to_rgb()
			brightness = compute_mean_brightness(rgb)
			print(f"ts={frame.timestamp_ns} brightness={brightness:.1f}")

			if args.demo_auto_click and brightness >= args.brightness_threshold:
				click(cx, cy)
				time.sleep(0.05)

			if args.show:
				disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
				cv2.imshow("preview", disp)
				if cv2.waitKey(1) & 0xFF == 27:  # ESC
					break

	if args.show:
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass
	return 0


if __name__ == "__main__":
	sys.exit(main())

