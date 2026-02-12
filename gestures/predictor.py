import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GesturePredictor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.history = {}  # Історія позицій (wrist_x, wrist_y) для кожної руки (key — індекс руки)
        self.max_history_len = 15

    def predict_gestures(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        gestures = []
        if not result.multi_hand_landmarks:
            self.history.clear()
            return gestures

        # Очищаємо історію для рук, яких немає
        current_hands = set(range(len(result.multi_hand_landmarks)))
        for hand_idx in list(self.history.keys()):
            if hand_idx not in current_hands:
                del self.history[hand_idx]

        # Обробляємо кожну руку
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            if i not in self.history:
                self.history[i] = []
            self.history[i].append((wrist_x, wrist_y))
            if len(self.history[i]) > self.max_history_len:
                self.history[i].pop(0)

            is_open = self._is_hand_open(hand_landmarks.landmark)
            wave_detected = self._is_wave(self.history[i])
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            if is_open and wave_detected:
                gesture = "wave"
                self.history[i].clear()  # очищаємо історію після розпізнавання маху
            else:
                gesture = self._classify_static_gesture(landmarks)

            bbox = self._get_bbox(landmarks)
            gestures.append({
                "gesture": gesture,
                "bbox": bbox,
                "landmarks": landmarks
            })

        return gestures

    def _is_hand_open(self, landmarks):
        """
        Визначає, чи відкрита долоня.
        Порівнюємо кінчики пальців (8, 12, 16, 20) з їхніми основами.
        """
        tips = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]

        opened_fingers = 0
        for t, b in zip(tips, bases):
            if landmarks[t].y < landmarks[b].y:
                opened_fingers += 1

        return opened_fingers >= 3

    def _is_wave(self, history):
        if len(history) < 10:
            return False

        xs = [pos[0] for pos in history]
        ys = [pos[1] for pos in history]

        def analyze_axis(values):
            diffs = np.diff(values)
            significant_diffs = [d for d in diffs if abs(d) > 0.004]
            if len(significant_diffs) < 5:
                return 0, 0, 0
            changes = 0
            for i in range(len(significant_diffs) - 1):
                if significant_diffs[i] * significant_diffs[i + 1] < 0:
                    changes += 1
            amplitude = max(values) - min(values)
            avg_speed = np.mean(np.abs(diffs))
            return changes, amplitude, avg_speed

        changes_x, amp_x, speed_x = analyze_axis(xs)
        changes_y, amp_y, speed_y = analyze_axis(ys)

        if (changes_x >= 2 and amp_x > 0.05 and speed_x > 0.01) or \
           (changes_y >= 2 and amp_y > 0.05 and speed_y > 0.01):
            return True
        return False

    def _classify_static_gesture(self, lm):
        wrist = lm[0]
        
        # Кінчики пальців (Tips)
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]
        
        # Суглоби (MCP/PIP) для порівняння
        thumb_ip = lm[3]
        index_mcp = lm[6]
        middle_mcp = lm[10]
        ring_mcp = lm[14]
        pinky_mcp = lm[18]

        # 1. 👍 thumbs_up: великий палець вгору, решта зігнуті
        is_thumb_up = thumb_tip[1] < thumb_ip[1] < wrist[1]
        fingers_folded = all(lm[i][1] > lm[i - 2][1] for i in [8, 12, 16, 20])

        if is_thumb_up and fingers_folded:
            return "thumbs_up"

        # 2. ✌️ victory: вказівний та середній вгорі, підмізинний та мізинець зігнуті
        # В координатах MediaPipe Y зменшується при русі вгору
        is_index_up = index_tip[1] < index_mcp[1]
        is_middle_up = middle_tip[1] < middle_mcp[1]
        is_ring_down = ring_tip[1] > ring_mcp[1]
        is_pinky_down = pinky_tip[1] > pinky_mcp[1]

        if is_index_up and is_middle_up and is_ring_down and is_pinky_down:
            return "victory"

        return None

    def _get_bbox(self, landmarks):
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        return min(xs), min(ys), max(xs), max(ys)
    
