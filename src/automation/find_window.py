from __future__ import annotations

import argparse
import sys
from typing import Optional

from .macos_windows import find_first_window_bounds_by_title


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description="Find macOS window bounds by title substring")
	p.add_argument("--title", default="iPhone Mirroring", help="Substring of window title to search for")
	args = p.parse_args(argv)
	bounds = find_first_window_bounds_by_title(args.title)
	if bounds is None:
		print(f"No on-screen window found matching title substring: {args.title}")
		return 2
	x, y, w, h = bounds
	print(f"{x} {y} {w} {h}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
