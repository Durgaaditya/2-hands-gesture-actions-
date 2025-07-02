--> Dual-Hand Gesture Control System

This project uses **MediaPipe**, **OpenCV**, and **PyAutoGUI** to recognize **two-hand gestures** via webcam and simulate keyboard/mouse actions. It supports separate gestures for the right and left hands, as well as combined gestures involving both hands.


--> Features

--> **Right Hand Gestures**:
  - Thumb + Index → Play/Pause
  - Thumb + Middle → Volume Up
  - Thumb + Ring → Volume Down
  - Thumb + Pinky → Fullscreen Toggle
  - Finger Gun → Next Tab

--> **Both Hand Gestures**:
  - Thumbs Touch → Scroll Down
  - Index Fingers Touch → Scroll Up

---

--> Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

Install with:

```bash
**pip install opencv-python mediapipe pyautogui numpy
> How to Run
Connect your webcam.

Run the script:

bash
Copy
Edit
python gesture_control.py
Perform gestures in front of the camera!

[] Notes
Ensure your hands are well lit and visible.

Right and left hands are distinguished by x-position (left = Hand 1, right = Hand 2).

Adjust distance thresholds if needed.**
