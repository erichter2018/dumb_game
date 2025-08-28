#!/usr/bin/env python3

import os
import sys
import threading
import time
import base64
from io import BytesIO
from typing import Optional, Tuple

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np
from PIL import Image

from .capture import RoiCapture
from .macos_windows import find_first_window_bounds_by_title_pixels
from .input_control import click, move_mouse_to
from .detection import detect_red_blobs, detect_blue_rectangles
import cv2

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'automation_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
TARGET_APP_TITLE = "iPhone Mirroring"
automation_worker = None

def encode_rgb_to_base64(rgb):
    """Helper function to encode RGB image to base64 - matches show_build pattern"""
    import cv2
    import base64
    
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64



# Multiple ROI presets
roi_presets = {
    'research_indicator': {'x': 360, 'y': 856, 'width': 50, 'height': 50},
    'actual_research': {'x': 300, 'y': 396, 'width': 50, 'height': 50}
}

# Current ROI settings (starts with research_indicator)
current_roi_name = 'research_indicator'
roi_settings = {
    'x': 360,
    'y': 856,
    'width': 50,
    'height': 50,
    'enabled': True
}

# Red blob detection parameters
RED_MIN_AREA = 30
RED_S_MIN = 100
RED_V_MIN = 60

class AutomationWorker:
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False
# Mouse control handled by imported click function

    def start(self):
        if self._running:
            socketio.emit('status', {'message': 'Already running'})
            return
        print("[automation] Starting web-based automation worker", flush=True)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        socketio.emit('status', {'message': 'STARTED - Looking for iPhone Mirroring window'})

    def stop(self):
        if not self._running:
            socketio.emit('status', {'message': 'Already stopped'})
            return
        print("[automation] Stopping worker...", flush=True)
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        socketio.emit('status', {'message': 'STOPPED'})

    def is_running(self):
        return self._running


    
    def _detect_blue_blob(self, rgb: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """More permissive blue blob detection"""
        import cv2
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # More permissive blue ranges in HSV - wider hue range and lower saturation/value thresholds
        mask1 = cv2.inRange(hsv, (90, 30, 30), (140, 255, 255))   # Main blue range
        mask2 = cv2.inRange(hsv, (180, 30, 30), (200, 255, 255))  # Cyan-ish blues
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Less aggressive cleanup to preserve more blue pixels
        mask = cv2.medianBlur(mask, 3)  # Reduced from 5 to 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))  # Close gaps
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best = None
        best_area = 0
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < 10:  # Reduced minimum area from 20 to 10
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if area > best_area:
                best_area = area
                best = (cx, cy, area)
        return best

    def _run(self):
        """Main detection loop"""
        cap = None
        roi = None
        last_bounds_check = 0.0
        initial_roi_set = False
        
        while not self._stop_event.is_set():
            try:
                now = time.perf_counter()
                
                # Refresh window bounds periodically
                if now - last_bounds_check >= 2.0 or roi is None:
                    window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
                    last_bounds_check = now
                    
                    if window_bounds is None:
                        socketio.emit('status', {'message': f'{TARGET_APP_TITLE} window not found'})
                        time.sleep(1.0)
                        continue
                    
                    # ROI coordinates are set from frontend initialization
                    if not initial_roi_set:
                        initial_roi_set = True
                    
                    # Calculate ROI from current settings
                    x, y, w, h = window_bounds
                    roi_x = x + roi_settings['x']
                    roi_y = y + roi_settings['y']
                    roi_w = roi_settings['width']
                    roi_h = roi_settings['height']
                    
                    # Ensure ROI stays within window bounds
                    roi_x = max(x, min(roi_x, x + w - roi_w))
                    roi_y = max(y, min(roi_y, y + h - roi_h))
                    roi_w = min(roi_w, x + w - roi_x)
                    roi_h = min(roi_h, y + h - roi_y)
                    new_roi = (roi_x, roi_y, roi_w, roi_h)
                    
                    if roi != new_roi:
                        # Close old capture
                        if cap is not None:
                            try:
                                cap.__exit__(None, None, None)
                            except:
                                pass
                        
                        # Create new capture
                        cap = RoiCapture(*new_roi)
                        cap.__enter__()
                        roi = new_roi
                        socketio.emit('status', {'message': f'ROI: {roi_w}x{roi_h} at ({roi_x},{roi_y})'})
                
                if cap is None:
                    time.sleep(0.1)
                    continue
                
                # Capture frame
                frame = cap.grab()
                rgb = frame.to_rgb()
                
                # Convert to base64 for web display
                pil_img = Image.fromarray(rgb)
                # Scale up for better web viewing
                pil_img = pil_img.resize((rgb.shape[1] * 4, rgb.shape[0] * 4), Image.NEAREST)
                
                buffer = BytesIO()
                pil_img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                # Simple red blob detection (yes/no only, no clicking)
                red_blobs = detect_red_blobs(rgb)
                has_red_blob = len(red_blobs) > 0
                
                # Send live preview and detection result
                socketio.emit('frame', {'image': img_str})
                
                if has_red_blob:
                    blob = red_blobs[0]  # Get the first detected blob
                    socketio.emit('detection', {
                        'found': True,
                        'x': blob['center'][0],
                        'y': blob['center'][1], 
                        'message': f'Red blob detected at ({blob["center"][0]}, {blob["center"][1]})'
                    })
                else:
                    socketio.emit('detection', {
                        'found': False,
                        'message': 'No red blob'
                    })
                
            except Exception as e:
                socketio.emit('status', {'message': f'Error: {e}'})
                time.sleep(0.5)
            
            time.sleep(0.1)
        
        # Cleanup
        if cap is not None:
            try:
                cap.__exit__(None, None, None)
            except:
                pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_automation():
    global automation_worker
    if automation_worker is None:
        automation_worker = AutomationWorker()
    automation_worker.start()
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_automation():
    global automation_worker
    if automation_worker:
        automation_worker.stop()
    return jsonify({'status': 'stopped'})

@app.route('/update_roi', methods=['POST'])
def update_roi():
    global roi_settings
    data = request.get_json()
    if data:
        if 'x' in data:
            roi_settings['x'] = max(0, int(data['x']))
        if 'y' in data:
            roi_settings['y'] = max(0, int(data['y']))
        if 'width' in data:
            roi_settings['width'] = max(50, min(1000, int(data['width'])))
        if 'height' in data:
            roi_settings['height'] = max(50, min(1000, int(data['height'])))
        if 'enabled' in data:
            roi_settings['enabled'] = bool(data['enabled'])
        print(f"[automation] ROI updated: {roi_settings}", flush=True)
    return jsonify({'status': 'updated', 'settings': roi_settings})

@app.route('/status')
def get_status():
    return jsonify({
        'running': automation_worker.is_running() if automation_worker else False,
        'settings': roi_settings,
        'current_roi_name': current_roi_name,
        'roi_presets': roi_presets
    })




@app.route('/switch_roi', methods=['POST'])
def switch_roi():
    """Switch to a different ROI preset"""
    global current_roi_name, roi_settings
    
    data = request.get_json()
    if not data or 'roi_name' not in data:
        return jsonify({'error': 'roi_name required'})
    
    roi_name = data['roi_name']
    if roi_name not in roi_presets:
        return jsonify({'error': f'Unknown ROI preset: {roi_name}'})
    
    # Switch to the new ROI preset
    current_roi_name = roi_name
    preset = roi_presets[roi_name]
    roi_settings.update({
        'x': preset['x'],
        'y': preset['y'],
        'width': preset['width'],
        'height': preset['height']
    })
    
    message = f'Switched to ROI preset: {roi_name} at ({preset["x"]}, {preset["y"]}) {preset["width"]}x{preset["height"]}'
    print(f"[automation] {message}", flush=True)
    
    return jsonify({
        'message': message,
        'current_roi_name': current_roi_name,
        'settings': roi_settings
    })

# Global variable to store detected builds and red blobs
detected_builds = []
detected_red_blobs = []

# Global variable to control the everything routine
everything_running = False
everything_thread = None

# Global variables for everything with cleanup
everything_cleanup_running = False
everything_cleanup_thread = None

# Global variables for mouse movement detection
last_mouse_position = None
last_mouse_check_time = None

def check_mouse_movement():
    """Check if mouse has moved and return True if we should pause"""
    global last_mouse_position, last_mouse_check_time
    
    try:
        from pynput.mouse import Controller
        mouse = Controller()
        current_pos = mouse.position
        current_time = time.time()
        
        # Initialize on first check
        if last_mouse_position is None:
            last_mouse_position = current_pos
            last_mouse_check_time = current_time
            return False
        
        # Check if mouse moved significantly (more than 5 pixels)
        distance = ((current_pos[0] - last_mouse_position[0]) ** 2 + 
                   (current_pos[1] - last_mouse_position[1]) ** 2) ** 0.5
        
        if distance > 5:
            print(f"[automation] Mouse movement detected: {distance:.1f} pixels - pausing for 5 seconds", flush=True)
            last_mouse_position = current_pos
            last_mouse_check_time = current_time
            return True
        
        return False
    except Exception as e:
        print(f"[automation] Error checking mouse movement: {e}", flush=True)
        return False

def wait_with_mouse_detection(seconds, check_stop_flag=None):
    """Wait for specified seconds, but keep extending if mouse keeps moving"""
    total_waited = 0
    while total_waited < seconds:
        # Check if we should stop (for everything routine) - check more frequently
        if check_stop_flag and not check_stop_flag():
            print(f"[automation] Stop detected during mouse wait - aborting", flush=True)
            return False
            
        # Wait 0.1 seconds (shorter intervals for more responsiveness)
        time.sleep(0.1)
        total_waited += 0.1
        
        # Check for mouse movement every 0.5 seconds
        if total_waited % 0.5 < 0.1:  # Every 0.5 seconds
            if check_mouse_movement():
                print(f"[automation] Mouse still moving - extending pause (waited {total_waited:.1f}s so far)", flush=True)
                total_waited = 0  # Reset the timer
    
    return True

@app.route('/check_build', methods=['POST'])
def check_build():
    """Scan the entire mirroring window for blue rectangles"""
    global detected_builds
    try:
        # Get current window bounds
        window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
        if window_bounds is None:
            return jsonify({'message': f'{TARGET_APP_TITLE} window not found', 'builds': []})
        
        window_x, window_y, window_w, window_h = window_bounds
        
        # Exclude top 15% and bottom 15%, use middle 70%
        top_offset = int(window_h * 0.15)  # Skip top 15%
        effective_height = int(window_h * 0.70)  # Use middle 70%
        effective_y = window_y + top_offset
        
        print(f"[automation] Scanning window for blue rectangles: ({window_x}, {window_y}) {window_w}x{window_h}, using middle 70% ({window_w}x{effective_height}) starting at y={effective_y}", flush=True)
        
        # Capture the window, excluding top 15% and bottom 15%
        roi_tuple = (window_x, effective_y, window_w, effective_height)
        cap = RoiCapture(*roi_tuple)
        cap.__enter__()
        frame = cap.grab()
        rgb = frame.to_rgb()
        cap.__exit__(None, None, None)
        
        if rgb is not None:
            # Detect blue rectangles
            builds = detect_blue_rectangles(rgb)
            detected_builds = builds
            
            print(f"[automation] Found {len(builds)} blue rectangles", flush=True)
            
            build_images = []
            for i, build in enumerate(builds):
                print(f"[automation] Build {i+1}: ({build['x']}, {build['y']}) {build['width']}x{build['height']}", flush=True)
                
                # Capture individual build image
                print(f"[automation] Capturing build {i+1}: region ({window_x + build['x']}, {effective_y + build['y']}) size {build['width']}x{build['height']}", flush=True)
                
                build_cap = RoiCapture(window_x + build['x'], effective_y + build['y'], build['width'], build['height'])
                build_cap.__enter__()
                build_frame = build_cap.grab()
                build_cap.__exit__(None, None, None)
                
                if build_frame:
                    build_rgb = build_frame.to_rgb()
                    build_key = chr(ord('A') + i)
                    img_base64 = encode_rgb_to_base64(build_rgb)
                    
                    print(f"[automation] Successfully captured build {build_key}: {build_rgb.shape}", flush=True)
                    
                    build_images.append({
                        'key': build_key,
                        'image': f"data:image/jpeg;base64,{img_base64}",
                        'info': build
                    })
                else:
                    print(f"[automation] Failed to capture build {i+1} frame", flush=True)
            
            # Send all build images via SocketIO
            print(f"[automation] Emitting {len(build_images)} build images via SocketIO", flush=True)
            socketio.emit('build_images', {'images': build_images})
            
            return jsonify({'builds': builds, 'message': f'Found {len(builds)} blue rectangles'})
        else:
            print(f"[automation] Failed to capture window", flush=True)
            return jsonify({'message': 'Failed to capture window', 'builds': []})
            
    except Exception as e:
        error_msg = f'Error scanning builds: {e}'
        print(f"[automation] {error_msg}", flush=True)
        return jsonify({'message': error_msg, 'builds': []})

@app.route('/show_build', methods=['POST'])
def show_build():
    """Show a specific detected build in the ROI preview"""
    try:
        data = request.get_json()
        build_index = data.get('build_index', 0)
        
        if build_index < 0 or build_index >= len(detected_builds):
            return jsonify({'message': 'Invalid build index'})
        
        build = detected_builds[build_index]
        
        # Get current window bounds
        window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
        if window_bounds is None:
            return jsonify({'message': f'{TARGET_APP_TITLE} window not found'})
        
        window_x, window_y, window_w, window_h = window_bounds
        
        # Calculate absolute coordinates of the build
        # Note: build coordinates are relative to the scanning area (middle 70% of window)
        # so we need to add the top offset
        top_offset = int(window_h * 0.15)  # Same offset used in scanning
        build_x = window_x + build['x']
        build_y = window_y + top_offset + build['y']
        
        print(f"[automation] Capturing build area: absolute coords ({build_x}, {build_y}) size {build['width']}x{build['height']}", flush=True)
        
        # Capture the build area
        roi_tuple = (build_x, build_y, build['width'], build['height'])
        cap = RoiCapture(*roi_tuple)
        cap.__enter__()
        frame = cap.grab()
        rgb = frame.to_rgb()
        cap.__exit__(None, None, None)
        
        if rgb is not None:
            # Convert to base64 and emit via socketio
            import cv2
            import base64
            
            print(f"[automation] Captured RGB image shape: {rgb.shape}", flush=True)
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            print(f"[automation] Encoded image size: {len(img_base64)} bytes", flush=True)
            
            # Send to frontend
            socketio.emit('frame', {'image': img_base64})
            
            print(f"[automation] Showing build {build_index + 1}: ({build['x']}, {build['y']}) {build['width']}x{build['height']}", flush=True)
        else:
            print(f"[automation] Failed to capture RGB image for build {build_index + 1}", flush=True)
            
        return jsonify({'message': f'Showing build {build_index + 1}'})
        
    except Exception as e:
        error_msg = f'Error showing build: {e}'
        print(f"[automation] {error_msg}", flush=True)
        return jsonify({'message': error_msg})

@app.route('/check_red_blob', methods=['POST'])
def check_red_blob():
    """Check for red blobs and return ALL images with labels"""
    global detected_red_blobs
    try:
        # Find the iPhone Mirroring window
        window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
        if not window_bounds:
            return jsonify({'message': f'{TARGET_APP_TITLE} window not found'})
        
        window_x, window_y, window_w, window_h = window_bounds
        
        # Exclude top 15% and bottom 15%, use middle 70% (same as build detection)
        top_offset = int(window_h * 0.15)  # Skip top 15%
        effective_height = int(window_h * 0.70)  # Use middle 70%
        effective_y = window_y + top_offset
        
        print(f"[automation] Scanning window for red blobs: ({window_x}, {window_y}) {window_w}x{window_h}, using middle 70% ({window_w}x{effective_height}) starting at y={effective_y}", flush=True)
        
        # Capture the window, excluding top 15% and bottom 15%
        cap = RoiCapture(window_x, effective_y, window_w, effective_height)
        with cap:
            frame = cap.grab()
            if frame:
                rgb = frame.to_rgb()
                red_blobs = detect_red_blobs(rgb)
                
                detected_red_blobs = []
                blob_images = []
                
                for i, blob in enumerate(red_blobs):
                    # Convert to absolute coordinates and add to global storage
                    # Note: blob coordinates are relative to the scanning area (middle 70% of window)
                    abs_x = window_x + blob['bounds'][0]
                    abs_y = effective_y + blob['bounds'][1]  # Use effective_y to account for top offset
                    blob_info = {
                        'x': blob['bounds'][0],
                        'y': blob['bounds'][1],
                        'width': blob['bounds'][2],
                        'height': blob['bounds'][3],
                        'center_x': blob['center'][0],
                        'center_y': blob['center'][1],
                        'area': blob['area'],
                        'circularity': blob['circularity'],
                        'abs_x': abs_x,
                        'abs_y': abs_y
                    }
                    detected_red_blobs.append(blob_info)
                    print(f"[automation] Red Blob: ({blob['bounds'][0]}, {blob['bounds'][1]}) {blob['bounds'][2]}x{blob['bounds'][3]}, circularity: {blob['circularity']:.2f}", flush=True)
                    
                    # Capture individual blob image
                    padding = 20
                    capture_x = max(0, blob_info['x'] - padding)
                    capture_y = max(0, blob_info['y'] - padding)
                    capture_w = min(window_w - capture_x, blob_info['width'] + 2 * padding)
                    capture_h = min(window_h - capture_y, blob_info['height'] + 2 * padding)
                    
                    print(f"[automation] Capturing blob {i+1}: region ({window_x + capture_x}, {effective_y + capture_y}) size {capture_w}x{capture_h}", flush=True)
                    
                    blob_cap = RoiCapture(window_x + capture_x, effective_y + capture_y, capture_w, capture_h)
                    with blob_cap:
                        blob_frame = blob_cap.grab()
                        if blob_frame:
                            blob_rgb = blob_frame.to_rgb()
                            blob_key = chr(ord('A') + i)
                            img_base64 = encode_rgb_to_base64(blob_rgb)
                            
                            print(f"[automation] Successfully captured blob {blob_key}: {blob_rgb.shape}", flush=True)
                            
                            blob_images.append({
                                'key': blob_key,
                                'image': f"data:image/jpeg;base64,{img_base64}",
                                'info': blob_info
                            })
                        else:
                            print(f"[automation] Failed to capture blob {i+1} frame", flush=True)
                
                print(f"[automation] Found {len(detected_red_blobs)} red blobs", flush=True)
                
                # Send all blob images via SocketIO
                print(f"[automation] Emitting {len(blob_images)} red blob images via SocketIO", flush=True)
                socketio.emit('red_blob_images', {'images': blob_images})
                
                return jsonify({
                    'message': f'Found {len(detected_red_blobs)} red blobs',
                    'total_red_blobs': len(detected_red_blobs),
                    'red_blobs': detected_red_blobs
                })
            else:
                return jsonify({'message': 'Failed to capture window'})
    except Exception as e:
        print(f"[automation] Error checking red blobs: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/show_red_blob', methods=['POST'])
def show_red_blob():
    """Show a specific red blob in the ROI preview with key label"""
    global detected_red_blobs
    try:
        data = request.get_json()
        red_blob_index = data.get('red_blob_index', 0)
        
        if not detected_red_blobs or red_blob_index >= len(detected_red_blobs):
            return jsonify({'message': 'No red blobs available'})
        
        red_blob = detected_red_blobs[red_blob_index]
        
        # Create a key label (A, B, C, D, etc.)
        blob_key = chr(ord('A') + red_blob_index)
        
        # Find the window again and capture the specific red blob area
        window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
        if not window_bounds:
            return jsonify({'message': f'{TARGET_APP_TITLE} window not found'})
        
        window_x, window_y, window_w, window_h = window_bounds
        
        # Capture the red blob area with some padding
        padding = 20
        capture_x = max(0, red_blob['x'] - padding)
        capture_y = max(0, red_blob['y'] - padding)
        capture_w = min(window_w - capture_x, red_blob['width'] + 2 * padding)
        capture_h = min(window_h - capture_y, red_blob['height'] + 2 * padding)
        
        cap = RoiCapture(window_x + capture_x, window_y + capture_y, capture_w, capture_h)
        with cap:
            frame = cap.grab()
            if frame:
                rgb = frame.to_rgb()
                
                # No image overlay - keep the original image clean
                encoded_image = encode_rgb_to_base64(rgb)
                socketio.emit('frame', {'image': encoded_image})
                print(f"[automation] Showing red blob {red_blob_index + 1} [{blob_key}]: ({red_blob['x']}, {red_blob['y']}) {red_blob['width']}x{red_blob['height']}", flush=True)
        
        return jsonify({
            'message': f'Showing red blob [{blob_key}] ({red_blob_index + 1} of {len(detected_red_blobs)}): ({red_blob["x"]}, {red_blob["y"]}) {red_blob["width"]}x{red_blob["height"]}',
            'key': blob_key,
            'index': red_blob_index + 1,
            'total': len(detected_red_blobs)
        })
    except Exception as e:
        print(f"[automation] Error showing red blob: {e}", flush=True)
        return jsonify({'error': str(e)}), 500



@app.route('/everything', methods=['POST'])
def everything():
    """everything routine: continuous buildbuildbuild with periodic research checks"""
    global everything_running, everything_thread
    try:
        # Set the flag to start the routine
        everything_running = True
        print(f"[automation] === EVERYTHING ROUTINE STARTED ===", flush=True)
        
        # Start the continuous routine in a separate thread
        import threading
        def everything_loop():
            global everything_running
            cycle_count = 0
            need_focus = True  # Only focus on first cycle, not after mouse movement restarts
            research_timer = 0  # Initialize timer at function level to persist across cycles
            
            while everything_running:
                cycle_count += 1
                research_timer = 0  # Reset timer at start of each cycle for consistent 3s intervals
                print(f"[automation] === EVERYTHING CYCLE {cycle_count} ===", flush=True)
                
                try:
                    # Step 1: Find a blue rectangle (buildbuildbuild part 1)
                    window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
                    if window_bounds is None:
                        print(f"[automation] {TARGET_APP_TITLE} window not found - stopping everything routine", flush=True)
                        break
                    
                    window_x, window_y, window_w, window_h = window_bounds
                    
                    # Scan for blue rectangles (same as buildbuildbuild)
                    top_offset = int(window_h * 0.15)  # Skip top 15%
                    effective_height = int(window_h * 0.70)  # Use middle 70%
                    effective_y = window_y + top_offset
                    
                    roi_tuple = (window_x, effective_y, window_w, effective_height)
                    cap = RoiCapture(*roi_tuple)
                    cap.__enter__()
                    frame = cap.grab()
                    rgb = frame.to_rgb()
                    cap.__exit__(None, None, None)
                    
                    if rgb is None:
                        print(f"[automation] Failed to capture window area - continuing", flush=True)
                        time.sleep(1)
                        continue
                    
                    # Detect blue rectangles
                    builds = detect_blue_rectangles(rgb)
                    
                    if not builds:
                        print(f"[automation] No blue rectangles found - waiting 5 seconds", flush=True)
                        # Wait 5 seconds, but check for stop every 0.5 seconds
                        for i in range(10):  # 10 * 0.5 = 5 seconds
                            if not everything_running:
                                print(f"[automation] Stop requested during build wait - exiting", flush=True)
                                return
                            time.sleep(0.5)
                        continue
                    
                    # Use the first detected build
                    build = builds[0]
                    print(f"[automation] Found blue rectangle: ({build['x']}, {build['y']}) {build['width']}x{build['height']}", flush=True)
                    
                    # Check for stop before focusing
                    if not everything_running:
                        print(f"[automation] Stop requested before focusing - exiting", flush=True)
                        return
                    
                    # Step 2: Focus the app (only on first cycle, not after mouse movement restarts)
                    if need_focus:
                        app_center_x = window_x + (window_w // 2)
                        app_center_y = window_y + (window_h // 2)
                        print(f"[automation] Focusing app at ({app_center_x}, {app_center_y})", flush=True)
                        move_mouse_to(app_center_x, app_center_y)
                        time.sleep(0.1)
                        click(app_center_x, app_center_y, hold_ms=50)
                        time.sleep(0.3)
                        need_focus = False  # Don't focus again on subsequent cycles
                    else:
                        print(f"[automation] Skipping app focus (already focused from previous cycle)", flush=True)
                    
                    # Step 3: Start pressing down on the blue rectangle
                    build_center_x = window_x + build['x'] + (build['width'] // 2)
                    build_center_y = effective_y + build['y'] + (build['height'] // 2)
                    
                    # Check for stop before starting press
                    if not everything_running:
                        print(f"[automation] Stop requested before starting press - exiting", flush=True)
                        return
                    
                    print(f"[automation] Starting continuous press at ({build_center_x}, {build_center_y})", flush=True)
                    move_mouse_to(build_center_x, build_center_y)
                    time.sleep(0.1)
                    
                    # Begin the press (this will be a long hold with interruptions)
                    from pynput.mouse import Button, Controller
                    _mouse = Controller()
                    _mouse.position = (build_center_x, build_center_y)
                    _mouse.press(Button.left)
                    
                    # Step 4: While holding, run research every 3 seconds
                    press_start_time = time.time()
                    
                    try:
                        while everything_running:
                            time.sleep(0.1)  # Small sleep to avoid busy waiting
                            research_timer += 0.1
                            
                            # Check for stop every loop iteration
                            if not everything_running:
                                print(f"[automation] Stop detected in main loop - breaking", flush=True)
                                break
                            
                            # Check for mouse movement every 0.5 seconds
                            if research_timer % 0.5 < 0.1:  # Approximately every 0.5 seconds
                                if check_mouse_movement():
                                    # Mouse moved, pause with continuous detection
                                    _mouse.release(Button.left)
                                    print(f"[automation] Pausing due to mouse movement", flush=True)
                                    
                                    # Wait with continuous mouse detection, checking for stop
                                    if wait_with_mouse_detection(5, lambda: everything_running):
                                        if everything_running:
                                            print(f"[automation] Mouse movement stopped - restarting with fresh build detection", flush=True)
                                            # Break out of the inner loop to restart build detection
                                            break
                                    else:
                                        # Stop was requested during wait
                                        return
                                    continue
                            
                            # Every 3 seconds, run research
                            if research_timer >= 3.0:
                                research_timer = 0
                                print(f"[automation] Running research check (press held for {time.time() - press_start_time:.1f}s)", flush=True)
                                
                                # Release mouse temporarily
                                _mouse.release(Button.left)
                                time.sleep(0.05)
                                
                                # Run check_and_finish_research logic inline (without focusing clicks)
                                try:
                                    # Check research_indicator for red blob
                                    research_indicator = roi_presets['research_indicator']
                                    ri_roi = (window_x + research_indicator['x'], window_y + research_indicator['y'], 
                                             research_indicator['width'], research_indicator['height'])
                                    
                                    cap = RoiCapture(*ri_roi)
                                    cap.__enter__()
                                    frame = cap.grab()
                                    rgb = frame.to_rgb()
                                    cap.__exit__(None, None, None)
                                    
                                    if rgb is not None:
                                        red_blobs = detect_red_blobs(rgb)
                                        
                                        if red_blobs:
                                            # Click research_indicator
                                            ri_center_x = window_x + research_indicator['x'] + (research_indicator['width'] // 2) + 10
                                            ri_center_y = window_y + research_indicator['y'] + (research_indicator['height'] // 2) + 10
                                            print(f"[automation] Research: Red blob found, clicking at ({ri_center_x}, {ri_center_y})", flush=True)
                                            move_mouse_to(ri_center_x, ri_center_y)
                                            click(ri_center_x, ri_center_y, hold_ms=100)
                                            
                                            # Process actual_research for blue blobs
                                            actual_research = roi_presets['actual_research']
                                            ar_roi = (window_x + actual_research['x'], window_y + actual_research['y'],
                                                     actual_research['width'], actual_research['height'])
                                            ar_center_x = window_x + actual_research['x'] + (actual_research['width'] // 2)
                                            ar_center_y = window_y + actual_research['y'] + (actual_research['height'] // 2)
                                            
                                            blue_clicks = 0
                                            max_blue_clicks = 10
                                            
                                            while blue_clicks < max_blue_clicks and everything_running:
                                                cap = RoiCapture(*ar_roi)
                                                cap.__enter__()
                                                frame = cap.grab()
                                                rgb = frame.to_rgb()
                                                cap.__exit__(None, None, None)
                                                
                                                if rgb is not None:
                                                    # Create a temporary worker instance to access the method
                                                    temp_worker = AutomationWorker()
                                                    blue_blob = temp_worker._detect_blue_blob(rgb)
                                                    
                                                    if blue_blob is not None:
                                                        blue_clicks += 1
                                                        print(f"[automation] Research: Blue blob click #{blue_clicks}", flush=True)
                                                        move_mouse_to(ar_center_x, ar_center_y)
                                                        click(ar_center_x, ar_center_y, hold_ms=100)
                                                    else:
                                                        print(f"[automation] Research: No more blue blobs after {blue_clicks} clicks", flush=True)
                                                        break
                                                else:
                                                    break
                                            
                                            # Click research_indicator again to finish
                                            print(f"[automation] Research: Finishing research routine", flush=True)
                                            move_mouse_to(ri_center_x, ri_center_y)
                                            click(ri_center_x, ri_center_y, hold_ms=100)
                                        else:
                                            print(f"[automation] Research: No red blob found", flush=True)
                                    
                                except Exception as research_error:
                                    print(f"[automation] Research error: {research_error}", flush=True)
                                
                                # Resume pressing on the build
                                if everything_running:
                                    print(f"[automation] Resuming press at build center", flush=True)
                                    move_mouse_to(build_center_x, build_center_y)
                                    time.sleep(0.1)
                                    _mouse.position = (build_center_x, build_center_y)
                                    _mouse.press(Button.left)
                                    press_start_time = time.time()  # Reset timer
                    
                    finally:
                        # Always release the mouse button
                        try:
                            _mouse.release(Button.left)
                            print(f"[automation] Released mouse button in finally block", flush=True)
                        except Exception as e:
                            print(f"[automation] Error releasing mouse in finally: {e}", flush=True)
                    
                    # If we get here and everything_running is still True, 
                    # it means we need to find a new build (the current one might be done)
                    if everything_running:
                        print(f"[automation] Build cycle complete, looking for next build...", flush=True)
                        time.sleep(1)  # Brief pause before next cycle
                
                except Exception as cycle_error:
                    print(f"[automation] Cycle error: {cycle_error}", flush=True)
                    time.sleep(1)  # Brief pause before retry
            
            print(f"[automation] === EVERYTHING ROUTINE STOPPED ===", flush=True)
        
        # Start the thread
        everything_thread = threading.Thread(target=everything_loop, daemon=True)
        everything_thread.start()
        print(f"[automation] Everything thread started with ID: {everything_thread.ident}", flush=True)
        
        return jsonify({'message': 'everything routine started - will run continuously until stopped'})
        
    except Exception as e:
        everything_running = False
        error_msg = f'Error starting everything routine: {e}'
        print(f"[automation] {error_msg}", flush=True)
        return jsonify({'message': error_msg})

@app.route('/stop_everything', methods=['POST'])
def stop_everything():
    """Stop the everything routine"""
    global everything_running, everything_thread
    
    print(f"[automation] === EVERYTHING ROUTINE STOP REQUESTED ===", flush=True)
    
    if everything_running:
        everything_running = False
        print(f"[automation] Set everything_running to False", flush=True)
        
        # Try to release any held mouse buttons
        try:
            from pynput.mouse import Button, Controller
            _mouse = Controller()
            _mouse.release(Button.left)
            print(f"[automation] Released mouse button", flush=True)
        except Exception as e:
            print(f"[automation] Error releasing mouse: {e}", flush=True)
        
        # Wait a moment for the thread to stop gracefully
        if everything_thread and everything_thread.is_alive():
            print(f"[automation] Waiting for thread {everything_thread.ident} to stop...", flush=True)
            everything_thread.join(timeout=2)  # Wait up to 2 seconds
            if everything_thread.is_alive():
                print(f"[automation] Thread did not stop gracefully within 2 seconds", flush=True)
            else:
                print(f"[automation] Thread stopped successfully", flush=True)
        
        # Send status update to frontend
        socketio.emit('status', {'message': 'EVERYTHING ROUTINE STOPPED'})
        
        return jsonify({'message': 'everything routine stopped'})
    else:
        print(f"[automation] Everything routine was not running", flush=True)
        return jsonify({'message': 'everything routine is not running'})

@app.route('/everything_with_cleanup', methods=['POST'])
def everything_with_cleanup():
    """Enhanced everything routine with MAX detection and cleanup logic"""
    global everything_cleanup_running, everything_cleanup_thread
    try:
        # Set the flag to start the routine
        everything_cleanup_running = True
        print(f"[automation] === EVERYTHING WITH CLEANUP ROUTINE STARTED ===", flush=True)
        
        # Start the continuous routine in a separate thread
        import threading
        def everything_cleanup_loop():
            global everything_cleanup_running
            
            # Import input control functions at the top
            
            cycle_count = 0
            need_focus = True  # Only focus on first cycle, not after mouse movement restarts
            research_timer = 0  # Initialize timer at function level to persist across cycles
            max_check_timer = 0  # Timer for MAX detection checks
            last_build_center = None  # Track last build location for cleanup search
            tried_red_blobs = []  # Track recently tried red blob positions to avoid repeating
            last_red_blob_click = None  # Track the last red blob click location
            builds_before_click = []  # Track builds that existed before red blob click
            
            while everything_cleanup_running:
                cycle_count += 1
                research_timer = 0  # Reset timer at start of each cycle for consistent 3s intervals
                max_check_timer = 0  # Reset MAX check timer
                print(f"[automation] === EVERYTHING WITH CLEANUP CYCLE {cycle_count} ===", flush=True)
                
                try:
                    # Step 1: Find a blue rectangle (buildbuildbuild part 1)
                    window_bounds = find_first_window_bounds_by_title_pixels(TARGET_APP_TITLE)
                    if window_bounds is None:
                        print(f"[automation] {TARGET_APP_TITLE} window not found - stopping everything with cleanup routine", flush=True)
                        break
                    
                    window_x, window_y, window_w, window_h = window_bounds
                    
                    # Scan for blue rectangles (same as buildbuildbuild)
                    top_offset = int(window_h * 0.15)  # Skip top 15%
                    effective_height = int(window_h * 0.70)  # Use middle 70%
                    
                    cap = RoiCapture(window_x, window_y + top_offset, window_w, effective_height)
                    cap.__enter__()
                    frame = cap.grab()
                    rgb = frame.to_rgb()
                    cap.__exit__(None, None, None)
                    
                    if rgb is not None:
                        builds = detect_blue_rectangles(rgb)
                        
                        if not builds:
                            print(f"[automation] No blue rectangles found - searching for red blobs to click", flush=True)
                            
                            # Search for red blobs when no builds are found
                            red_blobs = detect_red_blobs(rgb)
                            if red_blobs:
                                # Filter out recently tried red blobs to ensure progression
                                available_blobs = []
                                for blob in red_blobs:
                                    blob_key = (blob['center'][0], blob['center'][1])  # Use center as unique identifier
                                    if blob_key not in tried_red_blobs:
                                        available_blobs.append(blob)
                                
                                # If we've tried all blobs, reset the tried list and use all blobs
                                if not available_blobs:
                                    print(f"[automation] All {len(red_blobs)} red blobs tried, resetting list", flush=True)
                                    tried_red_blobs.clear()
                                    available_blobs = red_blobs
                                else:
                                    print(f"[automation] Found {len(red_blobs)} red blobs, {len(available_blobs)} untried (skipping {len(red_blobs) - len(available_blobs)} recently tried)", flush=True)
                                
                                # Try multiple red blobs to avoid infinite loops on non-functional ones
                                success = False
                                max_attempts = min(3, len(available_blobs))  # Try up to 3 different red blobs
                                
                                for attempt in range(max_attempts):
                                    red_blob = available_blobs[attempt]
                                    blob_key = (red_blob['center'][0], red_blob['center'][1])
                                    
                                    # Add this blob to tried list
                                    if blob_key not in tried_red_blobs:
                                        tried_red_blobs.append(blob_key)
                                        # Keep only last 10 tried blobs to avoid memory buildup
                                        if len(tried_red_blobs) > 10:
                                            tried_red_blobs.pop(0)
                                    
                                    # Try multiple click positions for each red blob
                                    click_positions = [
                                        (red_blob['center'][0], red_blob['center'][1] + 20),      # Original: 20 pixels below center
                                        (red_blob['center'][0] + 5, red_blob['center'][1] + 25),  # Slightly right and lower
                                        (red_blob['center'][0] - 5, red_blob['center'][1] + 15),  # Slightly left and higher
                                        (red_blob['center'][0], red_blob['center'][1] + 30),      # Further below
                                    ]
                                    
                                    blob_success = False
                                    for pos_attempt, (rel_x, rel_y) in enumerate(click_positions):
                                        red_center_x = window_x + rel_x
                                        red_center_y = window_y + top_offset + rel_y  # Add top_offset since red blob coords are relative to cropped region
                                        print(f"[automation] Red blob {attempt + 1}/{max_attempts}, position {pos_attempt + 1}/4", flush=True)
                                        print(f"[automation]   Blob center in scan region: ({red_blob['center'][0]}, {red_blob['center'][1]})", flush=True)
                                        print(f"[automation]   Click position (rel_x, rel_y): ({rel_x}, {rel_y})", flush=True)
                                        print(f"[automation]   Window bounds: ({window_x}, {window_y}) + top_offset: {top_offset}", flush=True)
                                        print(f"[automation]   Final click coordinates: ({red_center_x}, {red_center_y})", flush=True)
                                        
                                        # Capture builds before clicking to identify new ones later
                                        cap_before = RoiCapture(window_x, window_y + top_offset, window_w, effective_height)
                                        cap_before.__enter__()
                                        frame_before = cap_before.grab()
                                        if frame_before:
                                            rgb_before = frame_before.to_rgb()
                                            builds_before_click = detect_blue_rectangles(rgb_before)
                                            print(f"[automation] Builds before click: {len(builds_before_click)}", flush=True)
                                        cap_before.__exit__(None, None, None)
                                        
                                        move_mouse_to(red_center_x, red_center_y)
                                        click(red_center_x, red_center_y, hold_ms=100)
                                        
                                        # Store this click location for build selection
                                        last_red_blob_click = (red_center_x, red_center_y)
                                        
                                        # Wait a moment and check if builds appeared
                                        time.sleep(2)  # Wait for builds to appear
                                        cap_check = RoiCapture(*window_bounds)
                                        cap_check.__enter__()
                                        frame_check = cap_check.grab()
                                        if frame_check:
                                            rgb_check = frame_check.to_rgb()
                                            builds_after_click = detect_blue_rectangles(rgb_check)
                                            if builds_after_click:
                                                print(f"[automation] Red blob {attempt + 1}, position {pos_attempt + 1} produced {len(builds_after_click)} builds - success!", flush=True)
                                                success = True
                                                blob_success = True
                                                break  # Exit position loop
                                            else:
                                                print(f"[automation] Red blob {attempt + 1}, position {pos_attempt + 1} didn't produce builds", flush=True)
                                        cap_check.__exit__(None, None, None)
                                        
                                        # If this is the last position for this blob, wait a bit before trying next blob
                                        if pos_attempt < len(click_positions) - 1:
                                            time.sleep(1)  # Short wait between position attempts
                                    
                                    if blob_success:
                                        break  # Exit blob loop if we found a working position
                                
                                if not success:
                                    print(f"[automation] All {max_attempts} red blob attempts failed - refreshing red blob list", flush=True)
                                    
                                    # Rescan for fresh red blobs since current ones may be stale/invalid
                                    cap_refresh = RoiCapture(window_x, window_y + top_offset, window_w, effective_height)
                                    cap_refresh.__enter__()
                                    frame_refresh = cap_refresh.grab()
                                    rgb_refresh = frame_refresh.to_rgb()
                                    cap_refresh.__exit__(None, None, None)
                                    
                                    if rgb_refresh is not None:
                                        fresh_red_blobs = detect_red_blobs(rgb_refresh)
                                        
                                        if fresh_red_blobs:
                                            print(f"[automation] Found {len(fresh_red_blobs)} fresh red blobs after rescan (was {len(red_blobs)})", flush=True)
                                            
                                            # Reset tried_red_blobs to allow retry of refreshed blobs
                                            tried_red_blobs.clear()
                                            
                                            # Continue with fresh blobs (they will be detected in next cycle)
                                            print(f"[automation] Red blob list refreshed, continuing search...", flush=True)
                                        else:
                                            print(f"[automation] No fresh red blobs found after rescan - waiting 5 seconds", flush=True)
                                            for i in range(10):  # 10 * 0.5 = 5 seconds
                                                if not everything_cleanup_running:
                                                    return
                                                time.sleep(0.5)
                                    else:
                                        print(f"[automation] Failed to capture for red blob refresh - waiting 5 seconds", flush=True)
                                        for i in range(10):  # 10 * 0.5 = 5 seconds
                                            if not everything_cleanup_running:
                                                return
                                            time.sleep(0.5)
                                continue
                            else:
                                print(f"[automation] No red blobs found either - waiting 5 seconds", flush=True)
                                for i in range(10):  # 10 * 0.5 = 5 seconds
                                    if not everything_cleanup_running:
                                        print(f"[automation] Stop requested during build wait - exiting", flush=True)
                                        return
                                    time.sleep(0.5)
                                continue
                        
                        # If we just clicked a red blob, try to find NEW builds that appeared after the click
                        if 'last_red_blob_click' in locals() and last_red_blob_click and 'builds_before_click' in locals():
                            # Find builds that are NEW (not in the before_click list)
                            new_builds = []
                            for candidate_build in builds:
                                is_new = True
                                for old_build in builds_before_click:
                                    # Check if this build existed before (same position and size)
                                    if (abs(candidate_build['x'] - old_build['x']) < 10 and
                                        abs(candidate_build['y'] - old_build['y']) < 10 and
                                        abs(candidate_build['width'] - old_build['width']) < 10 and
                                        abs(candidate_build['height'] - old_build['height']) < 10):
                                        is_new = False
                                        break
                                if is_new:
                                    new_builds.append(candidate_build)
                            
                            print(f"[automation] Found {len(new_builds)} NEW builds out of {len(builds)} total", flush=True)
                            
                            if new_builds:
                                # Find the closest NEW build to the red blob click
                                click_x, click_y = last_red_blob_click
                                closest_build = None
                                min_distance = float('inf')
                                
                                for candidate_build in new_builds:
                                    build_center_x = candidate_build['x'] + candidate_build['width'] // 2
                                    build_center_y = candidate_build['y'] + candidate_build['height'] // 2
                                    distance = ((build_center_x - (click_x - window_x - top_offset)) ** 2 + 
                                              (build_center_y - (click_y - window_y - top_offset)) ** 2) ** 0.5
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_build = candidate_build
                                
                                build = closest_build
                                print(f"[automation] Selected NEW build closest to red blob click: ({build['x']}, {build['y']}) {build['width']}x{build['height']}, distance: {min_distance:.1f}", flush=True)
                            else:
                                # No new builds, fall back to closest existing build
                                click_x, click_y = last_red_blob_click
                                closest_build = None
                                min_distance = float('inf')
                                
                                for candidate_build in builds:
                                    build_center_x = candidate_build['x'] + candidate_build['width'] // 2
                                    build_center_y = candidate_build['y'] + candidate_build['height'] // 2
                                    distance = ((build_center_x - (click_x - window_x - top_offset)) ** 2 + 
                                              (build_center_y - (click_y - window_y - top_offset)) ** 2) ** 0.5
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_build = candidate_build
                                
                                build = closest_build
                                print(f"[automation] No new builds found, selected closest existing build: ({build['x']}, {build['y']}) {build['width']}x{build['height']}, distance: {min_distance:.1f}", flush=True)
                        else:
                            # Use the first valid build (largest by area)
                            build = builds[0]
                            print(f"[automation] Selected largest build from {len(builds)} available: ({build['x']}, {build['y']}) {build['width']}x{build['height']}", flush=True)
                        
                        build_x = build['x']
                        build_y = build['y']
                        build_w = build['width']
                        build_h = build['height']
                        build_center_x = window_x + build_x + build_w // 2
                        build_center_y = window_y + top_offset + build_y + build_h // 2  # Add top_offset since build coords are relative to cropped region
                        
                        # Store last build center for cleanup
                        last_build_center = (build_center_x, build_center_y)
                        
                        print(f"[automation] Found blue rectangle: ({build_x}, {build_y}) {build_w}x{build_h}", flush=True)
                        print(f"[automation] Build center calculation: window_y({window_y}) + top_offset({top_offset}) + build_y({build_y}) + build_h/2({build_h//2}) = {build_center_y}", flush=True)
                        print(f"[automation] Final build center: ({build_center_x}, {build_center_y})", flush=True)
                        
                        # Step 2: Focus the app (only on first cycle or if needed)
                        if need_focus:
                            # Check for stop before focusing
                            if not everything_cleanup_running:
                                print(f"[automation] Stop requested before focusing - exiting", flush=True)
                                return
                            
                            # Step 2: Focus the app
                            app_center_x = window_x + (window_w // 2)
                            app_center_y = window_y + (window_h // 2)
                            print(f"[automation] Focusing app at ({app_center_x}, {app_center_y})", flush=True)
                            move_mouse_to(app_center_x, app_center_y)
                            time.sleep(0.1)
                            click(app_center_x, app_center_y, hold_ms=50)
                            time.sleep(0.3)
                            need_focus = False  # Don't focus again unless restarted
                        else:
                            print(f"[automation] Skipping app focus (already focused from previous cycle)", flush=True)
                        
                        # Check for stop before starting press
                        if not everything_cleanup_running:
                            print(f"[automation] Stop requested before starting press - exiting", flush=True)
                            return
                        
                        print(f"[automation] Starting continuous press at ({build_center_x}, {build_center_y})", flush=True)
                        move_mouse_to(build_center_x, build_center_y)
                        time.sleep(0.1)
                        
                        # Begin the press (this will be a long hold with interruptions)
                        from pynput.mouse import Button, Controller
                        _mouse = Controller()
                        _mouse.position = (build_center_x, build_center_y)
                        _mouse.press(Button.left)
                        
                        # Step 4: While holding, run research every 3 seconds and check for MAX every 10 seconds
                        press_start_time = time.time()
                        
                        try:
                            while everything_cleanup_running:
                                time.sleep(0.1)  # Small sleep to avoid busy waiting
                                research_timer += 0.1
                                max_check_timer += 0.1
                                
                                # Check for stop every loop iteration
                                if not everything_cleanup_running:
                                    print(f"[automation] Stop detected in main loop - breaking", flush=True)
                                    break
                                
                                # Check for mouse movement every 0.5 seconds
                                if research_timer % 0.5 < 0.1:  # Approximately every 0.5 seconds
                                    if check_mouse_movement():
                                        # Mouse moved, pause with continuous detection
                                        _mouse.release(Button.left)
                                        print(f"[automation] Pausing due to mouse movement", flush=True)
                                        
                                        # Wait with continuous mouse detection, checking for stop
                                        if wait_with_mouse_detection(5, lambda: everything_cleanup_running):
                                            if everything_cleanup_running:
                                                print(f"[automation] Mouse movement stopped - restarting with fresh build detection", flush=True)
                                                # Break out of the inner loop to restart build detection
                                                break
                                        else:
                                            # Stop was requested during wait
                                            return
                                        continue
                                
                                # Check for MAX every 10 seconds
                                if max_check_timer >= 10.0:
                                    max_check_timer = 0
                                    print(f"[automation] Checking for MAX state (press held for {time.time() - press_start_time:.1f}s)", flush=True)
                                    
                                    # Capture current build area to check for MAX
                                    build_roi = (build_center_x - build_w//2, build_center_y - build_h//2, build_w, build_h)
                                    cap = RoiCapture(*build_roi)
                                    cap.__enter__()
                                    frame = cap.grab()
                                    max_rgb = frame.to_rgb()
                                    cap.__exit__(None, None, None)
                                    
                                    if max_rgb is not None and _detect_max_text(max_rgb):
                                        print(f"[automation] MAX detected! Starting cleanup routine", flush=True)
                                        _mouse.release(Button.left)
                                        time.sleep(0.1)
                                        
                                        # Run cleanup routine
                                        cleanup_success = run_cleanup_routine(window_bounds, last_build_center)
                                        if cleanup_success:
                                            print(f"[automation] Cleanup completed successfully - restarting cycle", flush=True)
                                        else:
                                            print(f"[automation] Cleanup failed - restarting cycle anyway", flush=True)
                                        break  # Break out to restart cycle
                                
                                # Every 3 seconds, run research
                                if research_timer >= 3.0:
                                    research_timer = 0
                                    print(f"[automation] Running research check (press held for {time.time() - press_start_time:.1f}s)", flush=True)
                                    
                                    # Release mouse temporarily
                                    _mouse.release(Button.left)
                                    time.sleep(0.05)
                                    
                                    # Run check_and_finish_research logic inline (without focusing clicks)
                                    try:
                                        # Check research_indicator for red blob
                                        research_indicator = roi_presets['research_indicator']
                                        ri_roi = (window_x + research_indicator['x'], window_y + research_indicator['y'], 
                                                 research_indicator['width'], research_indicator['height'])
                                        
                                        cap = RoiCapture(*ri_roi)
                                        cap.__enter__()
                                        frame = cap.grab()
                                        rgb = frame.to_rgb()
                                        cap.__exit__(None, None, None)
                                        
                                        if rgb is not None:
                                            red_blobs_detected = detect_red_blobs(rgb)
                                            if red_blobs_detected:
                                                ri_center_x = ri_roi[0] + research_indicator['width'] // 2 + 10
                                                ri_center_y = ri_roi[1] + research_indicator['height'] // 2 + 10
                                                
                                                print(f"[automation] Research: Red blob found, clicking at ({ri_center_x}, {ri_center_y})", flush=True)
                                                move_mouse_to(ri_center_x, ri_center_y)
                                                click(ri_center_x, ri_center_y, hold_ms=100)
                                                
                                                # Process actual_research for blue blobs
                                                actual_research = roi_presets['actual_research']
                                                ar_roi = (window_x + actual_research['x'], window_y + actual_research['y'], 
                                                          actual_research['width'], actual_research['height'])
                                                ar_center_x = ar_roi[0] + actual_research['width'] // 2
                                                ar_center_y = ar_roi[1] + actual_research['height'] // 2
                                                
                                                blue_clicks = 0
                                                while blue_clicks < 10 and everything_cleanup_running:
                                                    cap = RoiCapture(*ar_roi)
                                                    cap.__enter__()
                                                    frame = cap.grab()
                                                    rgb = frame.to_rgb()
                                                    cap.__exit__(None, None, None)
                                                    
                                                    if rgb is not None:
                                                        # Create a temporary worker instance to access the method
                                                        temp_worker = AutomationWorker()
                                                        blue_blob = temp_worker._detect_blue_blob(rgb)
                                                        if blue_blob is not None:
                                                            blue_clicks += 1
                                                            print(f"[automation] Research: Blue blob click #{blue_clicks}", flush=True)
                                                            move_mouse_to(ar_center_x, ar_center_y)
                                                            click(ar_center_x, ar_center_y, hold_ms=100)
                                                        else:
                                                            break
                                                    else:
                                                        break
                                                
                                                print(f"[automation] Research: No more blue blobs after {blue_clicks} clicks", flush=True)
                                                
                                                # Click research_indicator again to finish
                                                print(f"[automation] Research: Finishing research routine", flush=True)
                                                move_mouse_to(ri_center_x, ri_center_y)
                                                click(ri_center_x, ri_center_y, hold_ms=100)
                                            else:
                                                print(f"[automation] Research: No red blob found", flush=True)
                                        
                                    except Exception as research_error:
                                        print(f"[automation] Research error: {research_error}", flush=True)
                                    
                                    if everything_cleanup_running:
                                        print(f"[automation] Resuming press at build center", flush=True)
                                        move_mouse_to(build_center_x, build_center_y)
                                        time.sleep(0.1)
                                        _mouse.position = (build_center_x, build_center_y)
                                        _mouse.press(Button.left)
                                        press_start_time = time.time()  # Reset timer
                        
                        finally:
                            # Always release the mouse button
                            try:
                                _mouse.release(Button.left)
                                print(f"[automation] Released mouse button in finally block", flush=True)
                            except Exception as e:
                                print(f"[automation] Error releasing mouse in finally: {e}", flush=True)
                    
                    else:
                        print(f"[automation] Failed to capture screen region", flush=True)
                        time.sleep(1)
                        continue
                
                except Exception as e:
                    print(f"[automation] Error in everything with cleanup cycle: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    time.sleep(2)
                    continue
                
                print(f"[automation] Build cycle complete, looking for next build...", flush=True)
        
        everything_cleanup_thread = threading.Thread(target=everything_cleanup_loop)
        everything_cleanup_thread.start()
        
        return jsonify({'message': 'everything with cleanup routine started'})
    
    except Exception as e:
        print(f"[automation] Error starting everything with cleanup routine: {e}", flush=True)
        everything_cleanup_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/stop_everything_with_cleanup', methods=['POST'])
def stop_everything_with_cleanup():
    """Stop the everything with cleanup routine"""
    global everything_cleanup_running, everything_cleanup_thread
    if everything_cleanup_running:
        everything_cleanup_running = False
        print(f"[automation] === EVERYTHING WITH CLEANUP ROUTINE STOP REQUESTED ===", flush=True)
        print(f"[automation] Set everything_cleanup_running to False", flush=True)
        
        # Try to release any held mouse buttons
        try:
            from pynput.mouse import Button, Controller
            _mouse = Controller()
            _mouse.release(Button.left)
            print(f"[automation] Released mouse button", flush=True)
        except Exception as e:
            print(f"[automation] Error releasing mouse: {e}", flush=True)
        
        # Wait for thread to stop
        if everything_cleanup_thread and everything_cleanup_thread.is_alive():
            print(f"[automation] Waiting for thread {everything_cleanup_thread.ident} to stop...", flush=True)
            everything_cleanup_thread.join(timeout=2.0)
            if everything_cleanup_thread.is_alive():
                print(f"[automation] Thread did not stop within timeout", flush=True)
            else:
                print(f"[automation] Thread stopped successfully", flush=True)
        
        # Send status update to frontend
        socketio.emit('status', {'message': 'EVERYTHING WITH CLEANUP ROUTINE STOPPED'})
        
        return jsonify({'message': 'everything with cleanup routine stopped'})
    else:
        print(f"[automation] Everything with cleanup routine was not running", flush=True)
        return jsonify({'message': 'everything with cleanup routine is not running'})

def run_cleanup_routine(window_bounds, last_build_center):
    """
    Run cleanup routine when MAX is detected:
    1. Search around last build area for red blobs (expanding outward)
    2. Click 20 pixels below first red blob found
    3. Look for build and click it once
    4. Repeat: find red blob, click 20 below, build should stay this time
    """
    try:
        window_x, window_y, window_w, window_h = window_bounds
        
        if not last_build_center:
            print(f"[automation] Cleanup: No last build center available", flush=True)
            return False
        
        build_center_x, build_center_y = last_build_center
        
        # Convert to relative coordinates within window
        search_center_x = build_center_x - window_x
        search_center_y = build_center_y - window_y
        
        # Search in expanding squares around the last build location
        # Skip top and bottom 15% of screen
        top_offset = int(window_h * 0.15)
        effective_height = int(window_h * 0.70)
        search_area_y = top_offset
        search_area_h = effective_height
        
        # Define search radii to try (expanding outward)
        search_radii = [100, 200, 300, 400]
        
        for attempt in range(2):  # Two attempts: first opens build, second should build and stay
            print(f"[automation] Cleanup attempt #{attempt + 1}", flush=True)
            
            red_blob_found = False
            
            for radius in search_radii:
                if not everything_cleanup_running:
                    return False
                
                # Define search area around last build
                search_x = max(0, search_center_x - radius)
                search_y = max(search_area_y, search_center_y - radius)
                search_w = min(window_w - search_x, radius * 2)
                search_h = min(search_area_h - (search_y - search_area_y), radius * 2)
                
                print(f"[automation] Cleanup: Searching radius {radius} at ({search_x}, {search_y}) {search_w}x{search_h}", flush=True)
                
                # Capture search area
                cap = RoiCapture(window_x + search_x, window_y + search_y, search_w, search_h)
                cap.__enter__()
                frame = cap.grab()
                rgb = frame.to_rgb()
                cap.__exit__(None, None, None)
                
                if rgb is not None:
                    red_blobs = detect_red_blobs(rgb)
                    
                    if red_blobs:
                        # Use the first (largest) red blob
                        blob = red_blobs[0]
                        blob_center_x, blob_center_y = blob['center']
                        
                        # Convert back to absolute screen coordinates
                        abs_blob_x = window_x + search_x + blob_center_x
                        abs_blob_y = window_y + search_y + blob_center_y
                        
                        # Click 20 pixels below the red blob
                        click_x = abs_blob_x
                        click_y = abs_blob_y + 20
                        
                        print(f"[automation] Cleanup: Found red blob at ({abs_blob_x}, {abs_blob_y}), clicking at ({click_x}, {click_y})", flush=True)
                        

                        move_mouse_to(click_x, click_y)
                        time.sleep(0.1)
                        click(click_x, click_y, hold_ms=100)
                        time.sleep(0.5)  # Wait for build interface to appear
                        
                        red_blob_found = True
                        break
                
                time.sleep(0.1)  # Small delay between radius searches
            
            if not red_blob_found:
                print(f"[automation] Cleanup: No red blobs found in search area", flush=True)
                return False
            
            # After clicking red blob, look for a build to click
            print(f"[automation] Cleanup: Looking for build after red blob click", flush=True)
            time.sleep(0.3)  # Wait for build interface
            
            # Search for builds in the same area we just clicked
            cap = RoiCapture(window_x, window_y + top_offset, window_w, effective_height)
            cap.__enter__()
            frame = cap.grab()
            rgb = frame.to_rgb()
            cap.__exit__(None, None, None)
            
            if rgb is not None:
                builds = detect_blue_rectangles(rgb)
                
                if builds:
                    # Click the first build found
                    build = builds[0]
                    build_x = build['x']
                    build_y = build['y'] + top_offset  # Adjust for offset
                    build_w = build['width']
                    build_h = build['height']
                    build_center_x = window_x + build_x + build_w // 2
                    build_center_y = window_y + build_y + build_h // 2
                    
                    print(f"[automation] Cleanup: Found build at ({build_center_x}, {build_center_y}), clicking once", flush=True)
                    
                    move_mouse_to(build_center_x, build_center_y)
                    time.sleep(0.1)
                    click(build_center_x, build_center_y, hold_ms=100)
                    time.sleep(0.5)  # Wait for build action
                    
                    if attempt == 0:
                        print(f"[automation] Cleanup: First click done (should have opened build), searching for red blob again", flush=True)
                    else:
                        print(f"[automation] Cleanup: Second click done (build should stay), cleanup complete", flush=True)
                        return True
                else:
                    print(f"[automation] Cleanup: No builds found after red blob click - continuing to search for more red blobs", flush=True)
                    # Continue searching for more red blobs instead of giving up
                    attempt += 1
                    if attempt > 5:  # Prevent infinite loop
                        print(f"[automation] Cleanup: Too many attempts, giving up", flush=True)
                        return False
                    continue
            else:
                print(f"[automation] Cleanup: Failed to capture screen for build search", flush=True)
                return False
        
        return True
        
    except Exception as e:
        print(f"[automation] Cleanup error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def _detect_max_text(rgb: np.ndarray):
    """Detect if ROI has turned gray and contains 'MAX' text"""
    import cv2
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Check if image is predominantly gray (not blue)
    # Gray pixels should have low color variation
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Low saturation indicates gray/white colors
    gray_pixels = np.sum(saturation < 50)
    total_pixels = saturation.size
    gray_ratio = gray_pixels / total_pixels
    
    if gray_ratio < 0.6:  # Not predominantly gray
        return False
    
    # Use OCR to detect "MAX" text
    try:
        import pytesseract
        
        # Enhance contrast for better OCR
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        
        # Extract text
        text = pytesseract.image_to_string(enhanced, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        text = text.strip().upper()
        
        return 'MAX' in text
        
    except ImportError:
        # Fallback without OCR - just check if it's gray
        print(f"[automation] OCR not available, using gray detection only (gray_ratio: {gray_ratio:.2f})")
        return gray_ratio > 0.8
    except Exception as e:
        print(f"[automation] OCR error: {e}")
        return gray_ratio > 0.8

def _detect_blue_rectangles(rgb: np.ndarray):
    """Detect blue horizontal rectangles with specific characteristics:
    - Specific aspect ratio (roughly 2.5-4:1)
    - Contains white text/numbers
    - Contains yellow/orange circular element
    """
    import cv2
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Define blue color range in HSV
    # More permissive blue range to catch various shades
    lower_blue1 = np.array([90, 50, 50])   # Main blue range
    upper_blue1 = np.array([130, 255, 255])
    lower_blue2 = np.array([100, 30, 30])  # Lighter blues
    upper_blue2 = np.array([140, 255, 255])
    
    # Create masks for blue colors
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    builds = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter for rectangular shapes that are predominantly horizontal
        # Minimum size requirements
        if w < 100 or h < 40:  # Stricter minimum size
            continue
            
        # Aspect ratio check - more specific for the build rectangles
        aspect_ratio = w / h
        if aspect_ratio < 2.5 or aspect_ratio > 4.5:  # More specific ratio range
            continue
            
        # Area check
        area = w * h
        if area < 3000:  # Higher minimum area
            continue
            
        # Check if contour area is reasonably close to bounding rectangle area
        contour_area = cv2.contourArea(contour)
        rect_area = w * h
        fill_ratio = contour_area / rect_area
        if fill_ratio < 0.6:  # Should be at least 60% filled
            continue
        
        # Extract the region of interest for additional checks
        roi = rgb[y:y+h, x:x+w]
        
        # Check for white text (high brightness pixels)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        white_pixels = np.sum(gray_roi > 200)  # Count bright white pixels
        total_pixels = roi.shape[0] * roi.shape[1]
        white_ratio = white_pixels / total_pixels
        
        # Check for yellow/orange circular element
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Yellow/orange color range in HSV (more permissive)
        lower_yellow1 = np.array([10, 100, 100])   # Orange
        upper_yellow1 = np.array([30, 255, 255])   # Yellow
        lower_yellow2 = np.array([15, 50, 150])    # Lighter yellow/orange
        upper_yellow2 = np.array([35, 255, 255])   
        
        yellow_mask1 = cv2.inRange(hsv_roi, lower_yellow1, upper_yellow1)
        yellow_mask2 = cv2.inRange(hsv_roi, lower_yellow2, upper_yellow2)
        yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
        
        yellow_pixels = np.sum(yellow_mask > 0)
        yellow_ratio = yellow_pixels / total_pixels
        
        # Apply stricter criteria based on the specific build rectangle pattern
        if (white_ratio > 0.03 and    # At least 3% white pixels (text/numbers)
            white_ratio < 0.4 and     # Not too much white (avoid false positives)
            yellow_ratio > 0.01):     # At least 1% yellow/orange pixels (circle)
            
            builds.append({
                'x': int(x),
                'y': int(y), 
                'width': int(w),
                'height': int(h),
                'area': int(area),
                'aspect_ratio': round(aspect_ratio, 2),
                'white_ratio': round(white_ratio, 3),
                'yellow_ratio': round(yellow_ratio, 3)
            })
            
            print(f"[automation] Valid build found: ({x}, {y}) {w}x{h}, aspect={aspect_ratio:.2f}, white={white_ratio:.3f}, yellow={yellow_ratio:.3f}", flush=True)
        else:
            print(f"[automation] Rejected candidate: ({x}, {y}) {w}x{h}, aspect={aspect_ratio:.2f}, white={white_ratio:.3f}, yellow={yellow_ratio:.3f}", flush=True)
    
    # Sort by area (largest first)
    builds.sort(key=lambda b: b['area'], reverse=True)
    
    print(f"[automation] Found {len(builds)} valid build rectangles after robust filtering", flush=True)
    return builds

def main():
    print("Starting web-based automation controller...")
    print("Open http://localhost:3000 in your browser")
    socketio.run(app, host='localhost', port=3000, debug=False)

if __name__ == '__main__':
    main()
