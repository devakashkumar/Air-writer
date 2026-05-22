"""
Gesture-Controlled Digital Whiteboard
======================================
A hand gesture recognition drawing application using OpenCV and MediaPipe.

Features:
- Draw with your index finger
- Pan canvas with two fingers
- Erase with adjustable eraser size
- Color selection from 9 colors
- Brush size adjustment
- Save drawings as PNG files
- Smooth gesture tracking with real-time FPS monitoring

Controls:
- Index finger up: Draw/Erase
- Two fingers up: Pan canvas
- Fist/no fingers: Idle
- Q: Quit
- C: Clear canvas
- S: Save drawing
- E: Toggle eraser
- +/-: Adjust brush/eraser size
- 1-9: Select color (1=White, 2=Red, etc.)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math


# ============================================================
# CONFIGURATION SETTINGS
# ============================================================
# Adjust these settings to optimize performance on your system

# Camera capture settings
CAMERA_INDEX   = 0          # Webcam device index (usually 0 for default)
CAMERA_WIDTH   = 640        # Raw camera capture width
CAMERA_HEIGHT  = 480        # Raw camera capture height
REQUESTED_FPS  = 30         # Target frames per second

# Hand detection processing resolution
# Lower resolution = faster processing but less accuracy
PROCESS_WIDTH  = 424
PROCESS_HEIGHT = 240

# Display window settings
WINDOW_NAME    = "Gesture Whiteboard"
USE_FULLSCREEN = False      # Set True for fullscreen mode

# Maximum display resolution (used when not in fullscreen)
MAX_DISPLAY_W  = 1280
MAX_DISPLAY_H  = 720


# ============================================================
# SCREEN SIZE DETECTION
# ============================================================
# Automatically detect screen resolution and calculate layout dimensions

def get_screen_size():
    """
    Detect the user's screen resolution using tkinter.
    Falls back to default 1280x720 if detection fails.

    Returns:
        tuple: (screen_width, screen_height) in pixels
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        # Fallback if tkinter unavailable
        return 1280, 720


# Calculate display dimensions
_screen_width, _screen_height = get_screen_size()
SCREEN_W = _screen_width if USE_FULLSCREEN else min(_screen_width, MAX_DISPLAY_W)
SCREEN_H = _screen_height if USE_FULLSCREEN else min(_screen_height, MAX_DISPLAY_H)

# Layout dimensions: sidebar + camera area
SIDEBAR_W     = max(280, int(SCREEN_W * 0.195))  # Left control panel width
CAMERA_AREA_W = SCREEN_W - SIDEBAR_W              # Right drawing area width

# Optimize OpenCV for faster processing
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(4)  # Use 4 threads for parallel processing
except Exception:
    pass


# ============================================================
# WEBCAM INITIALIZATION
# ============================================================
# Set up camera capture with optimized settings

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# Verify camera is accessible
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Configure camera for optimal performance
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPEG compression
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          REQUESTED_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # Single frame buffer for low latency

# Create and configure display window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

if USE_FULLSCREEN:
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )
else:
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)


# ============================================================
# HAND DETECTION MODEL (MediaPipe)
# ============================================================
# Initialize hand tracking using Google's MediaPipe framework

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,        # Optimized for video (not static images)
    max_num_hands=1,                # Detect only one hand
    model_complexity=0,             # Lightweight model (0=light, 1=full)
    min_detection_confidence=0.60,  # Threshold for initial detection
    min_tracking_confidence=0.60,   # Threshold for frame-to-frame tracking
)


# ============================================================
# COLOR SCHEME (Dark Theme - BGR format)
# ============================================================
# All colors use OpenCV's BGR (Blue, Green, Red) format, not RGB

# Background colors
BG_BASE   = (10, 11, 15)    # Darkest background
BG_PANEL  = (15, 17, 22)    # Sidebar panel
BG_CARD   = (22, 25, 33)    # Card backgrounds
BG_HOVER  = (30, 34, 45)    # Hover state
BG_SEL    = (38, 43, 58)    # Selected state

# Border/outline colors
BORDER_DIM    = (38, 42, 55)     # Subtle borders
BORDER_MED    = (58, 63, 80)     # Medium emphasis
BORDER_BRIGHT = (88, 95, 118)    # Bright/highlighted

# Text colors
TXT_PRIMARY = (235, 238, 245)    # Main text
TXT_SECOND  = (160, 165, 180)    # Secondary text
TXT_MUTED   = (100, 106, 125)    # Muted/disabled text

# Status colors
ACCENT     = (255, 138, 80)      # Orange highlight
ACCENT_DIM = (100, 55, 32)       # Muted orange
DANGER     = (80, 60, 220)       # Red warning
DANGER_DIM = (40, 28, 100)       # Muted red
SUCCESS = (110, 210, 140)        # Green success
INFO    = (210, 170, 80)         # Yellow info

# Drawing color palette
PALETTE = {
    "WHITE":  (240, 240, 240),
    "RED":    (60, 60, 235),
    "ORANGE": (40, 140, 255),
    "YELLOW": (40, 220, 255),
    "GREEN":  (60, 210, 80),
    "CYAN":   (220, 200, 60),
    "BLUE":   (240, 100, 40),
    "PURPLE": (230, 70, 200),
    "PINK":   (170, 80, 240),
}

PALETTE_KEYS = list(PALETTE.keys())
ERASER_COLOR = (0, 0, 0)  # Black erases by blending


# ============================================================
# APPLICATION STATE
# ============================================================
# Track drawing state, settings, and user input

# Canvas and drawing state
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

current_color_key = "WHITE"
draw_color        = PALETTE["WHITE"]
is_eraser         = False

# Brush and eraser sizes
brush_size  = 8        # Default drawing brush size (pixels)
eraser_size = 48       # Default eraser radius (pixels)

# Smoothing variables for gesture tracking
# These reduce jitter in hand position detection
prev_x, prev_y           = 0, 0  # Previous draw position
move_prev_x, move_prev_y = 0, 0  # Previous pan position
smooth_x, smooth_y       = 0, 0  # Smoothed draw position
smooth_mx, smooth_my     = 0, 0  # Smoothed pan position

# Smoothing factors (higher = more smoothing but more lag)
SMOOTH_DRAW = 0.28  # For drawing gestures
SMOOTH_MOVE = 0.24  # For panning gestures

# Clear action cooldown (prevent accidental double-clears)
last_clear_time = 0.0
CLEAR_COOLDOWN  = 0.8

# Mouse tracking for UI hover states
mouse_x, mouse_y = -1, -1
menu_items = []  # List of clickable UI elements

# Performance monitoring
fps = 0.0
fps_last_time = time.time()

# Save button feedback
save_flash_until = 0.0  # Time until save button stops glowing


# ============================================================
# GEOMETRY & UTILITY HELPERS
# ============================================================

def cover_resize(img, target_width, target_height):
    """
    Resize image to fit target dimensions while maintaining aspect ratio.
    Crops excess from center (cover-fit, like CSS background-size: cover).

    Args:
        img: Input image
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Cropped and scaled image of exact dimensions
    """
    height, width = img.shape[:2]

    # Calculate scale to fit at least one dimension
    scale = max(target_width / width, target_height / height)

    # Resize to scaled dimensions
    new_width, new_height = int(width * scale), int(height * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Crop from center to target size
    x_offset = (new_width - target_width) // 2
    y_offset = (new_height - target_height) // 2

    return resized[y_offset:y_offset + target_height, x_offset:x_offset + target_width]


def rect_hit(rect, x, y):
    """
    Check if point (x, y) is inside rectangle.

    Args:
        rect: (x1, y1, x2, y2) rectangle bounds
        x, y: Point coordinates

    Returns:
        True if point is inside rectangle
    """
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def smooth_pt(raw_x, raw_y, prev_x, prev_y, factor=SMOOTH_DRAW):
    """
    Smooth point coordinates using exponential moving average.
    Reduces jitter from hand tracking.

    Args:
        raw_x, raw_y: Current unsmoothed coordinates
        prev_x, prev_y: Previous smoothed coordinates
        factor: Smoothing factor (0-1, higher = more smoothing)

    Returns:
        (smoothed_x, smoothed_y)
    """
    # Initialize on first call
    if prev_x == 0 and prev_y == 0:
        return raw_x, raw_y

    # Exponential moving average: new = prev + factor * (raw - prev)
    return int(prev_x + factor * (raw_x - prev_x)), int(prev_y + factor * (raw_y - prev_y))


# ============================================================
# DRAWING PRIMITIVES
# ============================================================
# Core drawing functions for UI rendering

def rrect(img, x1, y1, x2, y2, radius, color, thickness=-1, alpha=1.0):
    """
    Draw a rounded rectangle with optional alpha transparency.
    Uses ROI (Region of Interest) blending to avoid copying entire frame.

    Args:
        img: Target image
        x1, y1, x2, y2: Rectangle bounds
        radius: Corner radius in pixels
        color: BGR color tuple
        thickness: -1 for filled, >0 for outline
        alpha: Transparency (1.0 = opaque, 0.0 = transparent)
    """
    x1, y1, x2, y2, radius = map(int, (x1, y1, x2, y2, radius))

    height, width = img.shape[:2]

    # Clamp to image bounds
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return

    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    # Work on Region of Interest for efficiency
    roi = img[y1:y2 + 1, x1:x2 + 1]
    dst = roi.copy() if alpha < 0.999 else roi

    ox, oy = -x1, -y1

    if thickness < 0:  # Filled shape
        # Draw horizontal and vertical rectangles
        cv2.rectangle(dst, (x1 + radius + ox, y1 + oy), (x2 - radius + ox, y2 + oy), color, -1, cv2.LINE_AA)
        cv2.rectangle(dst, (x1 + ox, y1 + radius + oy), (x2 + ox, y2 - radius + oy), color, -1, cv2.LINE_AA)

        # Draw corner circles
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(dst, (cx + ox, cy + oy), radius, color, -1, cv2.LINE_AA)

    else:  # Outline only
        # Draw four edges
        cv2.line(dst, (x1 + radius + ox, y1 + oy), (x2 - radius + ox, y1 + oy), color, thickness, cv2.LINE_AA)
        cv2.line(dst, (x1 + radius + ox, y2 + oy), (x2 - radius + ox, y2 + oy), color, thickness, cv2.LINE_AA)
        cv2.line(dst, (x1 + ox, y1 + radius + oy), (x1 + ox, y2 - radius + oy), color, thickness, cv2.LINE_AA)
        cv2.line(dst, (x2 + ox, y1 + radius + oy), (x2 + ox, y2 - radius + oy), color, thickness, cv2.LINE_AA)

        # Draw corner arcs
        arcs = ((x1 + radius, y1 + radius, 180), (x2 - radius, y1 + radius, 270),
                (x1 + radius, y2 - radius, 90), (x2 - radius, y2 - radius, 0))

        for cx, cy, start in arcs:
            cv2.ellipse(dst, (cx + ox, cy + oy), (radius, radius), start, 0, 90, color, thickness, cv2.LINE_AA)

    # Apply alpha blending if needed
    if alpha < 0.999:
        cv2.addWeighted(dst, alpha, roi, 1 - alpha, 0, roi)


def txt(img, text, x, y, scale, color, thickness=1):
    """Draw text on image using Hershey Duplex font."""
    cv2.putText(img, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def divider(img, y, x1=None, x2=None, alpha=0.45):
    """Draw a subtle horizontal divider line in the sidebar."""
    if x1 is None:
        x1 = 24
    if x2 is None:
        x2 = SIDEBAR_W - 24

    y = int(y)
    base = img[y:y + 1, x1:x2].copy()

    cv2.line(img, (x1, y), (x2, y), BORDER_DIM, 1, cv2.LINE_AA)

    if alpha < 0.999:
        cv2.addWeighted(img[y:y + 1, x1:x2], alpha, base, 1 - alpha, 0, img[y:y + 1, x1:x2])


# ============================================================
# MENU SYSTEM & CLICK HANDLING
# ============================================================
# Handle UI interactions and button clicks

def add_item(action, rect, value=None):
    """
    Register a clickable UI element.

    Args:
        action: Action type (e.g., "color", "brush_inc", "save")
        rect: (x1, y1, x2, y2) hit-test bounds
        value: Optional parameter (e.g., color name)
    """
    menu_items.append({
        "action": action,
        "rect": rect,
        "value": value
    })


def handle_click(x, y):
    """
    Process mouse click on UI elements.

    Args:
        x, y: Click coordinates
    """
    global brush_size
    global eraser_size
    global is_eraser
    global current_color_key
    global draw_color
    global last_clear_time
    global save_flash_until

    for item in menu_items:
        if not rect_hit(item["rect"], x, y):
            continue

        action = item["action"]

        if action == "color":
            current_color_key = item["value"]
            draw_color = PALETTE[item["value"]]
            is_eraser = False

        elif action == "eraser":
            is_eraser = not is_eraser

        elif action == "brush_dec":
            brush_size = max(2, brush_size - 2)

        elif action == "brush_inc":
            brush_size = min(48, brush_size + 2)

        elif action == "erase_dec":
            eraser_size = max(12, eraser_size - 6)

        elif action == "erase_inc":
            eraser_size = min(120, eraser_size + 6)

        elif action == "clear":
            now = time.time()
            if now - last_clear_time >= CLEAR_COOLDOWN:
                canvas[:] = 0
                last_clear_time = now

        elif action == "save":
            filename = f"whiteboard_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            save_flash_until = time.time() + 1.2
            print(f"Saved: {filename}")

        return


def mouse_cb(event, x, y, flags, param):
    """Mouse callback for window events."""
    global mouse_x
    global mouse_y

    mouse_x, mouse_y = x, y

    # Handle clicks in sidebar (left side)
    if event == cv2.EVENT_LBUTTONDOWN and x < SIDEBAR_W:
        handle_click(x, y)


cv2.setMouseCallback(WINDOW_NAME, mouse_cb)


# ============================================================
# UI - PROFESSIONAL SIDEBAR
# ============================================================
# Build and render the left-side control panel

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SIMPLE = cv2.FONT_HERSHEY_SIMPLEX

UI_PAD = 22  # Sidebar padding
UI_GAP = 10  # Gap between elements


def text_size(text, scale, thickness=1, font=FONT):
    """Get text dimensions for layout calculations."""
    return cv2.getTextSize(str(text), font, scale, thickness)[0]


def fit_txt(img, text, x, y, max_width, scale, color, thickness=1, font=FONT):
    """
    Draw text and automatically shrink if it exceeds max width.
    Prevents text overflow in constrained UI areas.
    """
    text = str(text)
    current_scale = scale

    # Progressively reduce scale until text fits
    while current_scale > 0.22 and text_size(text, current_scale, thickness, font)[0] > max_width:
        current_scale -= 0.03

    cv2.putText(img, text, (int(x), int(y)), font, current_scale, color, thickness, cv2.LINE_AA)


def section_header(img, label, y):
    """Draw a section header in the sidebar."""
    fit_txt(img, label.upper(), UI_PAD, y, SIDEBAR_W - UI_PAD * 2, 0.31, TXT_MUTED, 1, FONT_SIMPLE)
    return y + 16


def pill(img, rect, label, selected=False, danger=False, accent=None):
    """
    Draw a pill-shaped button with hover and selected states.

    Args:
        img: Target image
        rect: (x1, y1, x2, y2) button bounds
        label: Button text
        selected: If True, show selected styling
        danger: If True, show warning/danger styling
        accent: Custom accent color for selected state
    """
    x1, y1, x2, y2 = map(int, rect)
    hover = rect_hit(rect, mouse_x, mouse_y)

    if selected:
        bg = BG_SEL
        border = accent or ACCENT
        text_color = TXT_PRIMARY
    elif danger:
        bg = (44, 30, 58) if not hover else (58, 38, 74)
        border = DANGER
        text_color = TXT_PRIMARY if hover else TXT_SECOND
    else:
        bg = BG_HOVER if hover else BG_CARD
        border = BORDER_BRIGHT if hover else BORDER_DIM
        text_color = TXT_PRIMARY if hover else TXT_SECOND

    # Draw shadow, background, and border
    rrect(img, x1 + 1, y1 + 2, x2 + 1, y2 + 2, 12, (0, 0, 0), -1, 0.13)
    rrect(img, x1, y1, x2, y2, 12, bg)
    rrect(img, x1, y1, x2, y2, 12, border, 1, 0.70 if (selected or hover or danger) else 0.45)

    # Center text
    tw, th = text_size(label, 0.36, 1, FONT_SIMPLE)
    fit_txt(img, label, (x1 + x2) // 2 - tw // 2, (y1 + y2) // 2 + th // 2, x2 - x1 - 14, 0.36, text_color, 1, FONT_SIMPLE)


def draw_palette(img, y_start):
    """
    Render the color palette grid in the sidebar.

    Args:
        img: Target image
        y_start: Top-left Y coordinate

    Returns:
        Bottom Y coordinate of the palette
    """
    cols = 3
    pad = UI_PAD
    gap = 9
    cell_width = (SIDEBAR_W - pad * 2 - gap * (cols - 1)) // cols
    cell_height = 50
    radius = 14

    for i, key in enumerate(PALETTE_KEYS):
        col_index = i % cols
        row_index = i // cols

        x1 = pad + col_index * (cell_width + gap)
        y1 = y_start + row_index * (cell_height + gap)
        x2 = x1 + cell_width
        y2 = y1 + cell_height

        rect = (x1, y1, x2, y2)
        color = PALETTE[key]
        selected = (key == current_color_key) and not is_eraser
        hover = rect_hit(rect, mouse_x, mouse_y)

        bg = BG_SEL if selected else (BG_HOVER if hover else BG_CARD)
        border = ACCENT if selected else (BORDER_BRIGHT if hover else BORDER_DIM)

        # Draw button background
        rrect(img, x1 + 1, y1 + 2, x2 + 1, y2 + 2, radius, (0, 0, 0), -1, 0.12)
        rrect(img, x1, y1, x2, y2, radius, bg)
        rrect(img, x1, y1, x2, y2, radius, border, 1, 0.75 if selected or hover else 0.45)

        # Draw color swatch circle
        cx = (x1 + x2) // 2
        cy = y1 + 18
        cv2.circle(img, (cx, cy), 10, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 10, (5, 6, 8), 1, cv2.LINE_AA)

        # Draw color name label
        label = key[:1] + key[1:].lower()
        tw, th = text_size(label, 0.28, 1, FONT_SIMPLE)
        fit_txt(img, label, cx - tw // 2, y2 - 12, cell_width - 8, 0.28,
                TXT_PRIMARY if selected else TXT_SECOND, 1, FONT_SIMPLE)

        # Selection indicator
        if selected:
            cv2.circle(img, (x2 - 12, y1 + 12), 4, ACCENT, -1, cv2.LINE_AA)

        add_item("color", rect, key)

    rows = math.ceil(len(PALETTE_KEYS) / cols)
    return y_start + rows * (cell_height + gap) - gap


def draw_eraser_btn(img, y):
    """Render the eraser toggle button."""
    x1, x2 = UI_PAD, SIDEBAR_W - UI_PAD
    y2 = y + 46

    rect = (x1, y, x2, y2)
    selected = is_eraser

    pill(img, rect, "ERASER ON" if selected else "ERASER", selected=selected, accent=ACCENT)

    # Draw eraser icon
    icx = x1 + 24
    icy = (y + y2) // 2

    rrect(img, icx - 9, icy - 6, icx + 9, icy + 6, 4,
          (210, 215, 225) if selected else TXT_MUTED)

    cv2.line(img, (icx - 5, icy + 6), (icx + 9, icy - 6),
             BG_CARD if not selected else BG_SEL, 1, cv2.LINE_AA)

    add_item("eraser", rect)
    return y2


def draw_slider(img, label, value, vmin, vmax, accent, y, act_dec, act_inc, show_preview=True):
    """
    Render an interactive slider control with +/- buttons.

    Args:
        img: Target image
        label: Slider label text
        value: Current value
        vmin, vmax: Min/max values
        accent: Highlight color
        y: Top Y coordinate
        act_dec: Decrease action name
        act_inc: Increase action name
    """
    x1 = UI_PAD
    x2 = SIDEBAR_W - UI_PAD
    slider_height = 80
    y2 = y + slider_height

    # Draw background card
    rrect(img, x1 + 1, y + 2, x2 + 1, y2 + 2, 16, (0, 0, 0), -1, 0.12)
    rrect(img, x1, y, x2, y2, 16, BG_CARD)
    rrect(img, x1, y, x2, y2, 16, BORDER_DIM, 1, 0.45)

    # Draw label and current value
    fit_txt(img, label, x1 + 14, y + 24, x2 - x1 - 80, 0.36, TXT_SECOND, 1, FONT_SIMPLE)

    val_str = f"{value}px"
    tw, th = text_size(val_str, 0.34, 1, FONT_SIMPLE)
    fit_txt(img, val_str, x2 - tw - 14, y + 24, 70, 0.34, accent, 1, FONT_SIMPLE)

    # Draw slider track
    track_x1 = x1 + 48
    track_x2 = x2 - 48
    track_y = y + 50

    rrect(img, track_x1, track_y - 3, track_x2, track_y + 3, 3, (42, 46, 58))

    # Calculate knob position
    normalized = (value - vmin) / max(1, vmax - vmin)
    knob_x = int(track_x1 + normalized * (track_x2 - track_x1))

    # Draw filled track up to knob
    rrect(img, track_x1, track_y - 3, knob_x, track_y + 3, 3, accent)

    # Draw knob with shadow
    cv2.circle(img, (knob_x, track_y), 8, accent, -1, cv2.LINE_AA)
    cv2.circle(img, (knob_x, track_y), 10, BG_CARD, 2, cv2.LINE_AA)

    # Draw +/- buttons
    button_width = 30
    dec_rect = (x1 + 12, track_y - 15, x1 + 12 + button_width, track_y + 15)
    inc_rect = (x2 - 12 - button_width, track_y - 15, x2 - 12, track_y + 15)

    for button_rect, symbol, action in [
        (dec_rect, "-", act_dec),
        (inc_rect, "+", act_inc)
    ]:
        hover = rect_hit(button_rect, mouse_x, mouse_y)
        bg = BG_HOVER if hover else (31, 34, 44)

        rrect(img, *button_rect, 9, bg)
        rrect(img, *button_rect, 9, BORDER_DIM, 1, 0.55)

        bx1, by1, bx2, by2 = button_rect
        tw, th = text_size(symbol, 0.44, 1, FONT_SIMPLE)

        fit_txt(img, symbol, (bx1 + bx2) // 2 - tw // 2, (by1 + by2) // 2 + th // 2,
                button_width, 0.44, TXT_PRIMARY, 1, FONT_SIMPLE)

        add_item(action, button_rect)

    return y2


def draw_action_btn(img, label, y, rect, danger=False, flash=False):
    """Draw a save/clear action button with optional flash effect."""
    if flash:
        x1, y1, x2, y2 = rect
        rrect(img, x1, y1, x2, y2, 12, (50, 150, 78))
        rrect(img, x1, y1, x2, y2, 12, SUCCESS, 1, 0.9)

        tw, th = text_size("SAVED", 0.36, 1, FONT_SIMPLE)
        fit_txt(img, "SAVED", (x1 + x2) // 2 - tw // 2, (y1 + y2) // 2 + th // 2,
                x2 - x1 - 12, 0.36, TXT_PRIMARY, 1, FONT_SIMPLE)
    else:
        pill(img, rect, label, danger=danger)


def draw_sidebar(img):
    global menu_items

    menu_items = []

    height = img.shape[0]

    cv2.rectangle(img, (0, 0), (SIDEBAR_W, height), BG_PANEL, -1)
    cv2.line(img, (SIDEBAR_W - 1, 0), (SIDEBAR_W - 1, height), BORDER_DIM, 1, cv2.LINE_AA)

    # Header card
    hx1, hy1, hx2, hy2 = 16, 16, SIDEBAR_W - 16, 82

    rrect(img, hx1, hy1, hx2, hy2, 18, (19, 22, 30))
    rrect(img, hx1, hy1, hx2, hy2, 18, BORDER_DIM, 1, 0.35)

    fit_txt(
        img,
        "WHITEBOARD",
        hx1 + 16,
        hy1 + 31,
        hx2 - hx1 - 32,
        0.52,
        TXT_PRIMARY,
        2,
        FONT_SIMPLE
    )

    fit_txt(
        img,
        "Gesture drawing studio",
        hx1 + 16,
        hy1 + 53,
        hx2 - hx1 - 32,
        0.32,
        TXT_MUTED,
        1,
        FONT_SIMPLE
    )

    y = 104

    y = section_header(img, "Color", y)
    y = draw_palette(img, y) + 18

    divider(img, y)
    y += 14

    y = section_header(img, "Tool", y)
    y = draw_eraser_btn(img, y) + 18

    divider(img, y)
    y += 14

    y = section_header(img, "Brush", y)
    y = draw_slider(
        img,
        "Stroke size",
        brush_size,
        2,
        48,
        INFO,
        y,
        "brush_dec",
        "brush_inc"
    ) + 12

    y = section_header(img, "Eraser", y)
    y = draw_slider(
        img,
        "Eraser radius",
        eraser_size,
        12,
        120,
        TXT_SECOND,
        y,
        "erase_dec",
        "erase_inc"
    ) + 18

    divider(img, y)
    y += 14

    pad = UI_PAD
    gap = 10
    half = (SIDEBAR_W - pad * 2 - gap) // 2

    save_rect = (pad, y, pad + half, y + 44)
    clear_rect = (pad + half + gap, y, SIDEBAR_W - pad, y + 44)

    flashing = time.time() < save_flash_until

    draw_action_btn(img, "SAVE", y, save_rect, flash=flashing)
    draw_action_btn(img, "CLEAR", y, clear_rect, danger=True)

    add_item("save", save_rect)
    add_item("clear", clear_rect)

    # Bottom shortcuts - ASCII only so OpenCV renders cleanly
    hint_y = height - 56

    if hint_y > y + 60:
        divider(img, hint_y - 12)

        fit_txt(
            img,
            "Q Quit   C Clear   S Save   E Eraser   +/- Size",
            UI_PAD,
            hint_y + 13,
            SIDEBAR_W - UI_PAD * 2,
            0.30,
            TXT_MUTED,
            1,
            FONT_SIMPLE
        )


# ============================================================
# UI - TOP HUD (Heads-Up Display)
# ============================================================
# Display current tool, size, and mode information at the top

def draw_hud(img, mode, fps_val):
    """
    Render the top status bar showing current tool, size, FPS, and gesture hints.

    Args:
        img: Target image
        mode: Current mode (DRAW, ERASE, MOVE, LOCKED, READY)
        fps_val: Frames per second value
    """
    height, width = img.shape[:2]

    hud_x1 = SIDEBAR_W + 18
    hud_y1 = 16
    hud_x2 = width - 18
    hud_y2 = 62

    # Draw HUD background card
    rrect(img, hud_x1 + 1, hud_y1 + 2, hud_x2 + 1, hud_y2 + 2, 16, (0, 0, 0), -1, 0.14)
    rrect(img, hud_x1, hud_y1, hud_x2, hud_y2, 16, (18, 20, 28), -1, 0.86)
    rrect(img, hud_x1, hud_y1, hud_x2, hud_y2, 16, BORDER_DIM, 1, 0.50)

    # Display current tool and size
    if is_eraser:
        tool_label = f"Eraser / {eraser_size}px"
        swatch_color = (220, 220, 220)
    else:
        tool_label = f"{current_color_key.capitalize()} / {brush_size}px"
        swatch_color = draw_color

    mid_y = (hud_y1 + hud_y2) // 2

    # Draw color swatch circle
    cv2.circle(img, (hud_x1 + 22, mid_y), 9, swatch_color, -1, cv2.LINE_AA)
    cv2.circle(img, (hud_x1 + 22, mid_y), 9, (0, 0, 0), 1, cv2.LINE_AA)

    fit_txt(img, tool_label, hud_x1 + 40, mid_y + 5, 230, 0.42, TXT_PRIMARY, 1, FONT_SIMPLE)

    # Display gesture hints
    hint = "Index: draw    Two fingers: pan    Fist: idle"
    available_width = max(180, hud_x2 - hud_x1 - 430)

    hw, _ = text_size(hint, 0.31, 1, FONT_SIMPLE)
    hint_x = hud_x1 + 280

    if hint_x + hw < hud_x2 - 160:
        fit_txt(img, hint, hint_x, mid_y + 5, available_width, 0.31, TXT_MUTED, 1, FONT_SIMPLE)

    # Display current mode and FPS
    pill_w = 138
    pill_x1 = hud_x2 - pill_w - 10
    pill_x2 = hud_x2 - 10
    pill_y1 = hud_y1 + 10
    pill_y2 = hud_y2 - 10

    mode_colors = {
        "DRAW":   (60, 200, 100),    # Green for drawing
        "ERASE":  (200, 180, 80),    # Yellow for erasing
        "MOVE":   (80, 160, 240),    # Blue for panning
        "LOCKED": TXT_MUTED,         # Gray when locked (cursor in sidebar)
        "READY":  BORDER_MED,        # Medium when ready
    }

    mode_color = mode_colors.get(mode, BORDER_MED)

    rrect(img, pill_x1, pill_y1, pill_x2, pill_y2, 10, (30, 33, 44))
    rrect(img, pill_x1, pill_y1, pill_x2, pill_y2, 10, mode_color, 1, 0.75)

    mode_str = f"{mode}  {fps_val:.0f} FPS"
    tw, th = text_size(mode_str, 0.32, 1, FONT_SIMPLE)

    fit_txt(img, mode_str, (pill_x1 + pill_x2) // 2 - tw // 2,
            (pill_y1 + pill_y2) // 2 + th // 2, pill_w - 10, 0.32, mode_color, 1, FONT_SIMPLE)


# ============================================================
# GESTURE CURSOR
# ============================================================
# Visual feedback for hand gesture state

def draw_cursor(img, x, y, mode):
    """
    Draw a cursor indicator showing current gesture mode.

    Args:
        img: Target image
        x, y: Cursor position
        mode: Gesture mode (DRAW, MOVE, or IDLE)
    """
    if mode == "DRAW":
        # Drawing mode: show brush/eraser preview circle
        color = draw_color if not is_eraser else (230, 230, 230)
        radius = eraser_size // 2 if is_eraser else max(4, brush_size // 2)

        if is_eraser:
            # Eraser cursor: concentric circles
            cv2.circle(img, (x, y), radius, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius - 4, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            # Brush cursor: filled circle with highlight
            cv2.circle(img, (x, y), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    elif mode == "MOVE":
        # Pan mode: crosshair cursor
        for dx, dy in ((18, 0), (-18, 0), (0, 18), (0, -18)):
            cv2.line(img, (x, y), (x + dx, y + dy), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(img, (x, y), (x + dx, y + dy), TXT_PRIMARY, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, TXT_PRIMARY, -1, cv2.LINE_AA)

    else:
        # Idle/waiting mode: simple circle
        cv2.circle(img, (x, y), 10, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (x, y), 10, TXT_MUTED, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 2, TXT_MUTED, -1, cv2.LINE_AA)


# ============================================================
# CANVAS COMPOSITING
# ============================================================
# Blend drawn content onto camera feed

def composite(frame, canvas):
    """
    Composite canvas drawing onto camera feed.
    Only blends pixels that have been drawn on (non-black).

    Args:
        frame: Camera frame
        canvas: Drawing canvas

    Returns:
        Composited frame
    """
    # Isolate the drawing area (right side, excluding sidebar)
    cam_area = frame[:, SIDEBAR_W:]
    canvas_area = canvas[:, SIDEBAR_W:]

    # Create mask of non-empty pixels (drawings)
    gray = cv2.cvtColor(canvas_area, cv2.COLOR_BGR2GRAY)
    mask = gray > 18

    # Blend: replace camera pixels where drawing exists
    cam_area[mask] = canvas_area[mask]

    return frame


# ============================================================
# HAND GESTURE RECOGNITION
# ============================================================
# Analyze hand landmarks to determine finger states

def fingers_up(hand_landmarks):
    """
    Detect which fingers are extended based on hand landmarks.

    MediaPipe hand landmarks:
    - 4: Thumb tip, 3: Thumb pip
    - 8: Index tip, 6: Index pip
    - 12: Middle tip, 10: Middle pip
    - 16: Ring tip, 14: Ring pip
    - 20: Pinky tip, 18: Pinky pip

    Args:
        hand_landmarks: MediaPipe hand landmarks object

    Returns:
        List of 5 values (0=down, 1=up) for [thumb, index, middle, ring, pinky]
    """
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: special case - check horizontal position (not vertical)
    fingers.append(
        1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x else 0
    )

    # Other fingers: check if tip is above pip (extended)
    for tip in tips[1:]:
        fingers.append(
            1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0
        )

    return fingers


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================
# Real-time gesture detection, drawing, and UI rendering

last_result = None
process_every = 1  # Process hand detection every N frames
frame_id = 0       # Frame counter

while True:
    # Capture frame from webcam
    ok, raw = cap.read()

    if not ok:
        print("Webcam read failed.")
        break

    # Flip horizontally for mirror effect (more intuitive for users)
    raw = cv2.flip(raw, 1)

    # Resize camera frame to fit in drawing area (excluding sidebar)
    cam_frame = cover_resize(raw, CAMERA_AREA_W, SCREEN_H)

    # Adaptive hand detection: skip frames if FPS drops to maintain smooth UI
    frame_id += 1
    process_every = 2 if fps and fps < 18 else 1

    # Run hand detection on lower resolution for speed
    if frame_id % process_every == 0 or last_result is None:
        small = cv2.resize(cam_frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True

        last_result = result
    else:
        # Reuse previous result when skipping detection
        result = last_result

    # Prepare frame: sidebar + camera view
    frame = np.empty((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    frame[:, :SIDEBAR_W] = BG_PANEL
    frame[:, SIDEBAR_W:SCREEN_W] = cam_frame

    mode = "READY"
    cursor_pos = None
    cursor_mode = "READY"

    # Process hand detection results
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Convert normalized landmarks to screen coordinates
            points = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                screen_x = SIDEBAR_W + int(lm.x * CAMERA_AREA_W)
                screen_y = int(lm.y * SCREEN_H)
                points.append((idx, screen_x, screen_y))

            # Get index and middle finger tips for gesture detection
            index_x, index_y = points[8][1], points[8][2]  # Index finger tip
            middle_x, middle_y = points[12][1], points[12][2]  # Middle finger tip

            # Smooth the index finger position for drawing
            smooth_x, smooth_y = smooth_pt(index_x, index_y, smooth_x, smooth_y, SMOOTH_DRAW)
            draw_x, draw_y = smooth_x, smooth_y

            # Get finger states
            fingers = fingers_up(hand_landmarks)

            # Check if hand is over sidebar (lock interaction)
            if draw_x < SIDEBAR_W:
                prev_x = prev_y = move_prev_x = move_prev_y = 0
                smooth_mx = smooth_my = 0
                mode = "LOCKED"

            # Two fingers up: pan/move canvas
            elif fingers[1] == 1 and fingers[2] == 1:
                prev_x = prev_y = 0
                mode = "MOVE"

                # Calculate midpoint between index and middle fingers
                raw_mid_x = (index_x + middle_x) // 2
                raw_mid_y = (index_y + middle_y) // 2

                # Smooth pan movement
                smooth_mx, smooth_my = smooth_pt(raw_mid_x, raw_mid_y, smooth_mx, smooth_my, SMOOTH_MOVE)

                # Apply canvas translation
                if move_prev_x and move_prev_y:
                    dx = smooth_mx - move_prev_x
                    dy = smooth_my - move_prev_y

                    if dx or dy:
                        transform = np.float32([
                            [1, 0, dx],
                            [0, 1, dy]
                        ])

                        canvas = cv2.warpAffine(
                            canvas,
                            transform,
                            (SCREEN_W, SCREEN_H),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )

                move_prev_x, move_prev_y = smooth_mx, smooth_my
                cursor_pos = (smooth_mx, smooth_my)
                cursor_mode = "MOVE"

            # One finger up (index only): draw or erase
            elif fingers[1] == 1 and fingers[2] == 0:
                move_prev_x = move_prev_y = smooth_mx = smooth_my = 0
                mode = "ERASE" if is_eraser else "DRAW"

                # Initialize position on first frame
                if not prev_x and not prev_y:
                    prev_x, prev_y = draw_x, draw_y

                # Draw line from previous to current position
                thickness = eraser_size if is_eraser else brush_size
                color = ERASER_COLOR if is_eraser else draw_color

                cv2.line(canvas, (prev_x, prev_y), (draw_x, draw_y), color, thickness, cv2.LINE_AA)

                # Draw circle at tip for smooth appearance
                cv2.circle(canvas, (draw_x, draw_y), max(2, thickness // 2), color, -1, cv2.LINE_AA)

                prev_x, prev_y = draw_x, draw_y
                cursor_pos = (draw_x, draw_y)
                cursor_mode = "DRAW"

            # No fingers up: idle state
            else:
                prev_x = prev_y = move_prev_x = move_prev_y = 0
                smooth_mx = smooth_my = 0
                cursor_pos = (draw_x, draw_y)
                cursor_mode = "IDLE"

    else:
        # No hand detected: reset positions
        prev_x = prev_y = move_prev_x = move_prev_y = 0
        smooth_x = smooth_y = smooth_mx = smooth_my = 0

    # Composite canvas drawing onto camera feed
    frame = composite(frame, canvas)

    # Render UI elements
    draw_sidebar(frame)
    draw_hud(frame, mode, fps)

    # Draw gesture cursor
    if cursor_pos:
        draw_cursor(frame, cursor_pos[0], cursor_pos[1], cursor_mode)

    # Calculate and smooth FPS
    now = time.time()
    instant_fps = 1.0 / max(now - fps_last_time, 1e-6)
    fps = fps * 0.88 + instant_fps * 0.12 if fps else instant_fps
    fps_last_time = now

    # Display frame
    cv2.imshow(WINDOW_NAME, frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Quit
        break

    elif key == ord("c"):  # Clear canvas
        now = time.time()
        if now - last_clear_time >= CLEAR_COOLDOWN:
            canvas[:] = 0
            last_clear_time = now

    elif key == ord("s"):  # Save canvas
        filename = f"whiteboard_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        save_flash_until = time.time() + 1.2
        print(f"Saved: {filename}")

    elif key in (ord("+"), ord("=")):  # Increase size
        if is_eraser:
            eraser_size = min(120, eraser_size + 6)
        else:
            brush_size = min(48, brush_size + 2)

    elif key in (ord("-"), ord("_")):  # Decrease size
        if is_eraser:
            eraser_size = max(12, eraser_size - 6)
        else:
            brush_size = max(2, brush_size - 2)

    elif ord("1") <= key <= ord("9"):  # Number keys: select color
        color_index = key - ord("1")
        if color_index < len(PALETTE_KEYS):
            current_color_key = PALETTE_KEYS[color_index]
            draw_color = PALETTE[current_color_key]
            is_eraser = False

    elif key == ord("e"):  # Toggle eraser
        is_eraser = not is_eraser


# Cleanup
cap.release()
cv2.destroyAllWindows()