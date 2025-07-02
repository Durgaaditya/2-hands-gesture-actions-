import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 1  # seconds
current_gesture = ""

# Right hand gesture logic
def get_right_hand_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = lm[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = lm[mp_hands.HandLandmark.PINKY_MCP]

    if np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) < 0.05:
        return 'CLICK'
    if (index_tip.y < index_mcp.y and thumb_tip.x > thumb_ip.x) and \
       (middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y):
        return 'FINGER_GUN'
    if np.hypot(thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y) < 0.05:
        return 'RING_TOUCH'
    if np.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y) < 0.05:
        return 'MIDDLE_TOUCH'
    if np.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y) < 0.05:
        return 'PINKY_TOUCH'

    return None  # Removed scroll-up from right hand

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand = None
    right_hand = None
    current_gesture = ""

    if results.multi_hand_landmarks:
        hand_coords = []
        for hand_landmarks in results.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            hand_coords.append((hand_landmarks, wrist_x))

        hand_coords.sort(key=lambda x: x[1])

        if len(hand_coords) >= 2:
            left_hand = hand_coords[0][0]
            right_hand = hand_coords[1][0]
        elif len(hand_coords) == 1:
            right_hand = hand_coords[0][0]

        for idx, (handLms, _) in enumerate(hand_coords):
            label = f"Hand {idx + 1}"
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, str(id), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            wrist = handLms.landmark[0]
            cv2.putText(frame, label, (int(wrist.x * w), int(wrist.y * h) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # ▶️ Right hand gestures
    if right_hand:
        gesture = get_right_hand_gesture(right_hand)
        if gesture and time.time() - last_action_time > cooldown:
            if gesture == 'CLICK':
                pyautogui.press('space')
                current_gesture = "CLICK"
            elif gesture == 'FINGER_GUN':
                pyautogui.hotkey('ctrl', 'tab')
                current_gesture = "FINGER_GUN"
            elif gesture == 'RING_TOUCH':
                pyautogui.press('volumedown')
                current_gesture = "VOLUME DOWN"
            elif gesture == 'MIDDLE_TOUCH':
                pyautogui.press('volumeup')
                current_gesture = "VOLUME UP"
            elif gesture == 'PINKY_TOUCH':
                pyautogui.press('f')
                current_gesture = "FULLSCREEN"
            last_action_time = time.time()

    # 🔀 Combined-hand gestures
    if left_hand and right_hand:
        # Scroll down - both thumbs meet
        left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_dist = np.hypot(left_thumb.x - right_thumb.x, left_thumb.y - right_thumb.y)

        if thumb_dist < 0.05 and time.time() - last_action_time > cooldown:
            pyautogui.scroll(-300)
            current_gesture = "SCROLL DOWN"
            last_action_time = time.time()

        # Scroll up - both index fingers meet
        left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dist = np.hypot(left_index.x - right_index.x, left_index.y - right_index.y)

        if index_dist < 0.05 and time.time() - last_action_time > cooldown:
            pyautogui.scroll(300)
            current_gesture = "SCROLL UP"
            last_action_time = time.time()

    # Display current gesture
    if current_gesture:
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
