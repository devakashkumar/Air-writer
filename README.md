# 🎨 Gesture-Controlled Digital Whiteboard

A real-time hand gesture recognition drawing application powered by OpenCV and Google's MediaPipe. Draw, erase, and pan a digital canvas using just your hand!

## ✨ Features

- **Gesture-Based Drawing**: Use your index finger to draw on the canvas
- **Multi-Finger Gestures**:
  - ☝️ **Index finger up** → Draw/Erase mode
  - 🖐 **palm up** → Pan/move canvas
  - ✊ **Fist (no fingers)** → Idle/pause mode
- **9 Color Palette**: White, Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink
- **Adjustable Brush & Eraser**: Real-time size control
- **Professional UI**: Dark theme sidebar with live FPS monitoring
- **Save Drawings**: Export canvas as PNG with timestamps
- **Smooth Gesture Tracking**: Exponential moving average filtering for jitter reduction
- **Responsive Performance**: Adaptive hand detection that skips frames if FPS drops

## 📋 Requirements

- Python 3.7+
- Webcam
- 4GB+ RAM (for smooth performance)

## 🛠️ Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## 🚀 Usage

**Start the application**:
```bash
python file.py
```

The app will open a window with:
- **Left sidebar**: Control panel with colors, sliders, and action buttons
- **Right side**: Camera feed with your canvas overlay
- **Top bar**: Current tool, size, FPS, and gesture hints

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `C` | Clear canvas (with 0.8s cooldown) |
| `S` | Save drawing as PNG |
| `E` | Toggle eraser on/off |
| `+` / `=` | Increase brush/eraser size |
| `-` / `_` | Decrease brush/eraser size |
| `1-9` | Select color (1=White, 2=Red, ... 9=Pink) |

## 🖱️ UI Controls

Click on the sidebar to interact:
- **Color Palette**: Click any color to select it
- **Eraser Button**: Toggle eraser mode
- **Brush Size Slider**: Adjust stroke thickness (+/- buttons or drag)
- **Eraser Size Slider**: Adjust eraser radius
- **SAVE**: Save current drawing to PNG file
- **CLEAR**: Erase the entire canvas

## 🎯 How It Works

### Gesture Recognition
The app uses MediaPipe's hand detection model to track:
- **Landmark positions**: 21 points per hand (wrist, fingers, joints)
- **Finger states**: Detects if each finger is extended or curled
- **Hand movements**: Tracks smooth motion for drawing

### Drawing Pipeline
1. Capture webcam frame
2. Detect hand landmarks using MediaPipe
3. Analyze finger positions to determine gesture
4. Draw on canvas based on gesture (draw, erase, pan)
5. Composite canvas onto camera feed
6. Render UI sidebar and status HUD
7. Display to user

### Performance Optimization
- **Lower resolution processing**: Hand detection runs on 424×240 (instead of full resolution)
- **Frame skipping**: If FPS drops below 18, detection is skipped every other frame
- **ROI-based alpha blending**: UI elements only update their region, not entire frame
- **Multi-threading**: OpenCV optimized for 4-thread parallelism

## 🛠️ Configuration

Edit the top of `file.py` to customize:

```python
# Camera settings
CAMERA_WIDTH   = 640      # Raw camera capture width
CAMERA_HEIGHT  = 480      # Raw camera capture height
REQUESTED_FPS  = 30       # Target FPS

# Processing settings
PROCESS_WIDTH  = 424      # Hand detection resolution
PROCESS_HEIGHT = 240
USE_FULLSCREEN = False    # Set True for fullscreen mode

# Gesture smoothing
SMOOTH_DRAW = 0.28        # Drawing smoothing (higher = more lag)
SMOOTH_MOVE = 0.24        # Panning smoothing
```

## 🐛 Troubleshooting

**"Error: Could not access webcam"**
- Ensure your webcam is connected and not used by another app
- Try changing `CAMERA_INDEX` from `0` to `1`, `2`, etc.

**Low FPS / Laggy performance**
- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT`
- Reduce `PROCESS_WIDTH` and `PROCESS_HEIGHT`
- Close other applications

**Hand not detected**
- Ensure adequate lighting
- Show full hand with fingers visible
- Adjust `min_detection_confidence` (lower = more detections, but less accurate)

**Jittery drawing**
- Increase `SMOOTH_DRAW` value (0.3-0.4 for more smoothing)
- Slow down your hand movements

## 📁 Project Structure

```
Air writer/
├── file.py          # Main application (fully commented)
├── README.md            # This file
├── .gitignore          # Git ignore rules
```

## 💡 Tips & Tricks

1. **Better lighting = better detection**: Ensure good overhead lighting
2. **Steady your hand**: Keep your wrist relatively still for precise drawing
3. **Pan before drawing**: If canvas goes off-screen, use two-finger pan
4. **Color numbers**: Press `1-9` to quickly switch colors without touching the sidebar
5. **Save often**: Save your work frequently to avoid losing drawings

## 🎓 Learning Resource

This project demonstrates:
- Real-time video processing with OpenCV
- ML-based hand pose estimation (MediaPipe)
- Gesture recognition and state machines
- UI/UX design for touch-free interfaces
- Performance optimization techniques
- Smooth motion tracking using exponential moving average

## 📝 License

Free to use and modify for personal and educational purposes.

## 🤝 Contributing

Found a bug or have an idea? Feel free to improve this project!

## 📧 Questions?

Check the code comments for detailed explanations of every function and algorithm. The code is written to be human-friendly and educational.

---

**Happy drawing!** 🎨✨
