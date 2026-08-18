# Smart Intercom Face Unlock

A small end-to-end computer vision project that recognizes an authorized user in an Android intercom app and triggers the **Open Door** action through UI automation.

The project was built for a practical constraint: the intercom application does not expose a public API, so the integration works by observing the emulator window and interacting with the visible UI.

## How it works

```text
Android emulator window
        |
        v
Screen capture (MSS)
        |
        v
Face detection (OpenCV Haar Cascade)
        |
        v
Face embedding (pretrained ArcFace ONNX)
        |
        v
Cosine similarity against a local reference image
        |
        v
Button detection (OpenCV template matching)
        |
        v
Mouse automation (PyAutoGUI)
```

The recognition model is **pretrained**. This project focuses on integrating existing CV components into a working real-time pipeline rather than training a face-recognition model from scratch.

## Features

- real-time capture of a selected emulator window;
- face detection and ArcFace embedding inference with ONNX Runtime;
- cosine-similarity identity check;
- configurable recognition and template-matching thresholds;
- several consecutive authorized frames required before an action can be triggered;
- visual face/similarity overlay for debugging;
- template-based detection of the door button;
- cooldown between automated actions.

## Project structure

```text
.
├── main.py               # application loop and orchestration
├── config.py             # paths, thresholds and runtime settings
├── face_recognition.py   # face detection and ArcFace inference
├── intercom_ui.py        # window capture, button detection and UI automation
├── requirements.txt
└── README.md
```

The ONNX model, reference face image and UI button template are intentionally kept local and excluded from version control.

## Tech stack

- Python
- OpenCV
- ONNX Runtime
- NumPy
- MSS
- PyAutoGUI
- PyGetWindow

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Mahan1341/smart-intercom-face-unlock.git
cd smart-intercom-face-unlock
pip install -r requirements.txt
```

If GPU inference is needed, replace `onnxruntime` with the appropriate `onnxruntime-gpu` package for your environment.

### 2. Download the pretrained ArcFace model

The project expects an InsightFace ArcFace ONNX model at:

```text
models/w600k_r50.onnx
```

The model file is not distributed with this repository.

### 3. Add a reference image

Place a local image of the authorized user at:

```text
reference.jpg
```

The image is excluded through `.gitignore` and should not be committed.

### 4. Add a button template

Create a small crop containing only the intercom's door-opening button and save it locally as:

```text
button.png
```

This file is also ignored by Git because a real application screenshot may contain implementation-specific or private UI data.

### 5. Configure the application

Runtime parameters are stored in `config.py`:

```python
WINDOW_NAME = "LDPlayer"
FACE_THRESHOLD = 0.5
BUTTON_THRESHOLD = 0.8
GLOBAL_COOLDOWN_SECONDS = 20
AUTHORIZED_FRAMES_REQUIRED = 3
```

Thresholds are application-specific and should be calibrated for the actual camera, emulator layout and reference image.

### 6. Run

```bash
python main.py
```

Press `Esc` to stop the application.

## Design choices

### Emulator integration

The original mobile application has no public API available for this use case. Instead of modifying the app, the project treats the emulator as a visual interface: it captures the window, analyzes the current frame and performs the same interaction a user would perform manually.

### Face recognition

A pretrained ArcFace model produces an embedding for the detected face. The embedding is L2-normalized and compared with the reference embedding using cosine similarity.

### UI detection

The door control is located with normalized template matching. This is intentionally simple and works for a stable UI, but it is sensitive to changes in scale, theme and button appearance.

## Limitations

- Haar Cascade detection is lightweight but substantially less robust than modern face detectors.
- The current pipeline crops the detected face but does not perform landmark-based face alignment before ArcFace inference.
- There is no liveness or anti-spoofing check, so this should **not** be treated as a security-grade access-control system.
- Identity verification uses a manually selected similarity threshold.
- Template matching depends on the intercom UI remaining visually similar.
- The implementation currently assumes a desktop environment where PyGetWindow and PyAutoGUI can control the emulator window.

## Demo

A short sanitized demo will be added here showing detection, similarity scoring, button localization and the final automated click without exposing private intercom or residential information.

## Motivation

The goal of this project was to connect computer-vision inference with a real application that had no convenient programmatic integration point. The interesting part is the complete pipeline: capture, inference, decision logic, UI detection and automation working together in real time.
