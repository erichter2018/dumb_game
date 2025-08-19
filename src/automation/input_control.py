from __future__ import annotations

import subprocess
import time
from typing import Tuple

from pynput.mouse import Button, Controller


_mouse = Controller()


def move_mouse_to(x: int, y: int) -> None:
	_mouse.position = (x, y)


def click(x: int, y: int, button: str = "left", hold_ms: int = 20) -> None:
	"""Click using both pynput and osascript as fallback"""
	try:
		# Try pynput first
		btn = Button.left if button == "left" else Button.right
		prev = _mouse.position
		_mouse.position = (x, y)
		_mouse.press(btn)
		time.sleep(max(0, hold_ms) / 1000.0)
		_mouse.release(btn)
		_mouse.position = prev
		print(f"[input_control] pynput click at ({x}, {y})", flush=True)
	except Exception as e:
		print(f"[input_control] pynput failed: {e}, trying osascript", flush=True)
		
	# Also try osascript as backup (macOS native)
	try:
		cmd = f'osascript -e "tell application \\"System Events\\" to click at {{{x}, {y}}}"'
		result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			print(f"[input_control] osascript click successful at ({x}, {y})", flush=True)
		else:
			print(f"[input_control] osascript failed: {result.stderr}", flush=True)
	except Exception as e:
		print(f"[input_control] osascript exception: {e}", flush=True)

