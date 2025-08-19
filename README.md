### Desktop game automation (macOS)

This Python tool captures a small screen region at high FPS, analyzes frames with OpenCV, and controls the mouse for automation.

### Prerequisites
- macOS (Screen Recording + Accessibility permissions required)
- Python 3.10+

### Grant permissions (macOS)
1. System Settings → Privacy & Security → Screen Recording → enable for your terminal (iTerm/Terminal) and/or the Python binary inside `.venv`.
2. System Settings → Privacy & Security → Accessibility → enable for your terminal and/or the Python binary inside `.venv`.

After enabling, restart the terminal if needed.

### Setup
```bash
cd /Users/erichter/Desktop/dumb.game.nosync
python3 -m venv .venv
./.venv/bin/pip install -U pip setuptools wheel
./.venv/bin/pip install -r requirements.txt
```

### Run (help)
```bash
cd /Users/erichter/Desktop/dumb.game.nosync
PYTHONPATH=src ./.venv/bin/python -m automation.app --help
```

### Example: watch a 300x300 ROI at 60 FPS and show a preview window
```bash
PYTHONPATH=src ./.venv/bin/python -m automation.app --roi 800 400 300 300 --fps 60 --show
```

### Example: demo auto-click when ROI is bright
```bash
PYTHONPATH=src ./.venv/bin/python -m automation.app --roi 800 400 300 300 --fps 60 --demo-auto-click --brightness-threshold 180
```

### Project layout
- `src/automation/capture.py`: fast ROI capture with `mss`
- `src/automation/vision.py`: OpenCV helpers (template/color search)
- `src/automation/input_control.py`: mouse control via `pynput`
- `src/automation/app.py`: CLI main loop
- `src/automation/config.py`: small configuration helpers

### Notes
- On Retina displays, coordinates are treated in screen pixels. If you notice offset, tell me your display setup and we can add scaling calibration.
- If you prefer hotkeys to start/stop, we can add a global keyboard hook.

