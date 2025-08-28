"""
Simple, clean image detection functions.
"""
import numpy as np


def detect_red_blobs(rgb: np.ndarray):
    """
    Simple, unified red blob detection function.
    Takes a screen ROI and returns locations of UP ARROW red blobs (NOT exclamation points).
    
    Args:
        rgb: numpy array of RGB image data
        
    Returns:
        list: List of dicts with 'center': (x, y), 'bounds': (x, y, w, h), 'circularity': float
        
    Note:
        Red blobs in the top 30% and left 20% region are ignored to avoid UI elements.
        This restriction can be modified by changing the TOP_EXCLUDE_RATIO and LEFT_EXCLUDE_RATIO constants.
    """
    import cv2
    
    # Configuration for region exclusion - easily modifiable
    TOP_EXCLUDE_RATIO = 0.30    # Ignore red blobs in top 30% of window
    LEFT_EXCLUDE_RATIO = 0.20   # Ignore red blobs in left 20% of window
    
    # Calculate exclusion boundaries
    img_height, img_width = rgb.shape[:2]
    top_exclude_boundary = int(img_height * TOP_EXCLUDE_RATIO)
    left_exclude_boundary = int(img_width * LEFT_EXCLUDE_RATIO)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Tuned red color ranges for the research indicator arrows
    # These are 30x30 to 40x40 red circles with white up arrows
    lower_red1 = np.array([0, 100, 100])    # Lower red range
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])  # Upper red range  
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask - less aggressive for small objects
    red_mask = cv2.medianBlur(red_mask, 3)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_blobs = []
    for i, contour in enumerate(contours):
        # Filter by area - 30x30=900 to 40x40=1600 pixels
        area = cv2.contourArea(contour)
        if area < 400 or area > 2500:  # More generous range: 400-2500 pixels
            continue
        
        # Get bounding rectangle for position checking
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Filter out red blobs in the excluded region (top 30% and left 20%)
        if center_y < top_exclude_boundary and center_x < left_exclude_boundary:
            print(f"[automation] Red contour {i+1}: REJECTED - in excluded region (top {TOP_EXCLUDE_RATIO*100}% and left {LEFT_EXCLUDE_RATIO*100}%) at ({center_x}, {center_y})", flush=True)
            continue
        
        # Size check - should be roughly 30-40 pixels in each dimension
        if w < 20 or h < 20 or w > 50 or h > 50:
            continue
        
        # Must be roughly square/circular (research indicators are round)
        aspect_ratio = w / h
        if not (0.7 <= aspect_ratio <= 1.4):  # More lenient for pixelated circles
            continue
        
        # Calculate circularity - very lenient for small pixelated arrows
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.5:  # Slightly more lenient for pixelated arrows
            print(f"[automation] Red contour {i+1}: REJECTED - not circular enough ({circularity:.2f})", flush=True)
            continue
        
        # Check for white content inside (the arrow)
        roi = rgb[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Convert ROI to grayscale and check for white pixels
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        white_pixels = np.sum(roi_gray > 180)  # Slightly lower threshold for white
        total_pixels = roi_gray.size

        if total_pixels == 0:
            continue

        white_ratio = white_pixels / total_pixels

        # Slightly more lenient: should have white content for arrow - 18-60% white
        if white_ratio < 0.18 or white_ratio > 0.60:
            print(f"[automation] Red contour {i+1}: REJECTED - white ratio {white_ratio:.2f} not in arrow range (0.18-0.60)", flush=True)
            continue

        center_x = x + w // 2
        center_y = y + h // 2
        
        # Additional check: look for the specific red color intensity
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        red_pixels = np.sum((roi_hsv[:,:,1] > 80) & (roi_hsv[:,:,2] > 80))  # Lower thresholds
        red_ratio = red_pixels / total_pixels
        
        if red_ratio < 0.2:  # Should have some red content
            print(f"[automation] Red contour {i+1}: REJECTED - not enough red content ({red_ratio:.2f})", flush=True)
            continue
            
        # Additional check: white pixels should be concentrated in center (arrow pattern)
        center_region_y1 = max(0, h//4)
        center_region_y2 = min(h, 3*h//4)  
        center_region_x1 = max(0, w//4)
        center_region_x2 = min(w, 3*w//4)
        
        center_region = roi_gray[center_region_y1:center_region_y2, center_region_x1:center_region_x2]
        if center_region.size > 0:
            center_white_pixels = np.sum(center_region > 180)
            center_white_ratio = center_white_pixels / center_region.size
            
            # Arrow should have some concentrated white in center (more lenient)
            if center_white_ratio < 0.25:
                print(f"[automation] Red contour {i+1}: REJECTED - white not concentrated in center ({center_white_ratio:.2f})", flush=True)
                continue
        
        print(f"[automation] FOUND: Red arrow at ({center_x}, {center_y}) size {w}x{h}, area={area}, circularity={circularity:.2f}, white_ratio={white_ratio:.2f}, red_ratio={red_ratio:.2f}", flush=True)
        
        red_blobs.append({
            'center': (int(center_x), int(center_y)),
            'bounds': (int(x), int(y), int(w), int(h)),
            'area': int(area),
            'circularity': float(circularity),
            'white_ratio': float(white_ratio),
        })
    
    # Sort by area and circularity (best matches first)
    red_blobs.sort(key=lambda b: (b['area'], b['circularity']), reverse=True)
    
    return red_blobs


def detect_blue_rectangles(rgb: np.ndarray):
    """
    Detect blue horizontal rectangles: ~160x58 pixels with white text and yellow/orange circle.
    """
    import cv2
    try:
        import pytesseract
        HAS_TESSERACT = True
    except ImportError:
        HAS_TESSERACT = False
    
    print(f"[automation] Blue detection: Input image shape: {rgb.shape}", flush=True)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Blue color ranges tuned for build buttons (more permissive)
    lower_blue = np.array([90, 40, 40])    # Broader range for blue
    upper_blue = np.array([140, 255, 255])
    
    # Create mask for blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    print(f"[automation] Blue detection: Blue pixels found: {np.sum(blue_mask > 0)}", flush=True)
    
    # Minimal cleanup to preserve button shapes
    blue_mask = cv2.medianBlur(blue_mask, 3)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[automation] Blue detection: Found {len(contours)} contours", flush=True)
    
    blue_rectangles = []
    for i, contour in enumerate(contours):
        # Get bounding rectangle first
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        print(f"[automation] Blue contour {i+1}: bounds=({x}, {y}) {w}x{h}, area={area}", flush=True)
        
        # Target size: ~160x58 pixels, tight tolerance since they're consistent
        # Width should be 140-180, height should be 50-66 (much tighter ranges)
        if not (140 <= w <= 180 and 50 <= h <= 66):
            print(f"[automation] Blue contour {i+1}: REJECTED - size not ~160x58 ({w}x{h})", flush=True)
            continue
        
        # Aspect ratio should be roughly 160/58 = 2.76, tight tolerance 2.4-3.1
        aspect_ratio = w / h
        if not (2.4 <= aspect_ratio <= 3.1):
            print(f"[automation] Blue contour {i+1}: REJECTED - aspect ratio ({aspect_ratio:.2f}) not 2.4-3.1", flush=True)
            continue
        
        # Area should be reasonable for a ~160x58 rectangle (9280 pixels)
        expected_area = w * h
        if area < expected_area * 0.5:  # At least 50% filled
            print(f"[automation] Blue contour {i+1}: REJECTED - area too small ({area} < {expected_area * 0.5})", flush=True)
            continue
        
        # Extract ROI for content analysis
        roi = rgb[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Check for white text (should always have numbers/letters)
        white_pixels = np.sum(roi_gray > 180)
        white_ratio = white_pixels / (w * h)
        print(f"[automation] Blue contour {i+1}: white_pixels={white_pixels}, white_ratio={white_ratio:.3f}", flush=True)
        
        # Must have significant white content (text)
        if white_ratio < 0.08:  # At least 8% white pixels
            print(f"[automation] Blue contour {i+1}: REJECTED - insufficient white text ({white_ratio:.3f} < 0.08)", flush=True)
            continue
        
        # Check for yellow/orange circle (should always be present)
        yellow_lower = np.array([15, 100, 100])  # Orange-yellow range
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(roi_hsv, yellow_lower, yellow_upper)
        yellow_pixels = np.sum(yellow_mask > 0)
        print(f"[automation] Blue contour {i+1}: yellow_pixels={yellow_pixels}", flush=True)
        
        # Must have yellow/orange circle
        if yellow_pixels < 50:  # Should have at least 50 yellow pixels for the circle
            print(f"[automation] Blue contour {i+1}: REJECTED - no yellow circle ({yellow_pixels} < 50)", flush=True)
            continue
        
        # Try OCR for additional validation
        text = ''
        has_ocr_text = False
        if HAS_TESSERACT:
            try:
                text = pytesseract.image_to_string(roi_gray, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ').strip()
                has_ocr_text = len(text) > 0
                print(f"[automation] Blue contour {i+1}: OCR text='{text}'", flush=True)
            except Exception as e:
                print(f"[automation] Blue contour {i+1}: OCR failed: {e}", flush=True)
        
        # This looks like a build button!
        center_x = x + w // 2
        center_y = y + h // 2
        
        print(f"[automation] Blue contour {i+1}: ACCEPTED - build button at ({center_x}, {center_y})", flush=True)
        
        blue_rectangles.append({
            'center': (int(center_x), int(center_y)),
            'bounds': (int(x), int(y), int(w), int(h)),
            'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),  # Add these for compatibility
            'area': int(area),
            'aspect_ratio': float(aspect_ratio),
            'fill_ratio': float(area / (w * h)),
            'has_text': True,  # We verified white pixels above
            'has_yellow': True,  # We verified yellow pixels above
            'text': str(text),
            'white_ratio': float(white_ratio),
            'yellow_pixels': int(yellow_pixels)
        })
    
    # Sort by area (largest first)
    blue_rectangles.sort(key=lambda r: r['area'], reverse=True)
    
    print(f"[automation] Blue detection: Final result: {len(blue_rectangles)} blue rectangles found", flush=True)
    return blue_rectangles