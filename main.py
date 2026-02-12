import cv2
import os
import numpy as np
from face.embedder import FaceEmbedder
from face.recognizer import FaceRecognizer
from gestures.predictor import GesturePredictor
from core.controller import start_voice_assistant, handle_event
from PIL import Image, ImageDraw, ImageFont
from data.data import USERS_DATA

def center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_hand_of_face(hand_bbox, face_bbox):
    hx, hy = center(hand_bbox)
    fx1, fy1, fx2, fy2 = face_bbox
    padding_x = 200 
    padding_y = 300
    return (fx1 - padding_x <= hx <= fx2 + padding_x) and \
           (fy1 <= hy <= fy2 + padding_y)

def draw_text_ua(frame, text, position, color=(0, 255, 0), font_size=24):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Шлях до шрифту (Windows: arial.ttf)
    font_path = "C:/Windows/Fonts/arial.ttf" 
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # PIL: (R, G, B)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def load_users_from_dict(data, embedder, recognizer, base_path="data"):
    for user_id, images in data.items():
        print(f"[INFO] Завантаження користувача: {user_id}")
        for img_rel_path in images:
            img_path = os.path.join(base_path, img_rel_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = embedder.get_embeddings(img)
            if faces:
                recognizer.add_user(user_id, faces[0]["embedding"])
        print(f"[OK] {user_id} готовий\n")

def main():
    cap = cv2.VideoCapture(0)
    embedder = FaceEmbedder(device="cpu")
    recognizer = FaceRecognizer(threshold=0.4)
    gesture_predictor = GesturePredictor()

    load_users_from_dict(data=USERS_DATA, embedder=embedder, recognizer=recognizer)
    start_voice_assistant()

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = embedder.get_embeddings(frame)
        recognized_users = []
        last_found_user = None

        # Обробка облич
        for face in faces:
            user_id = recognizer.recognize(face["embedding"])
            bbox = face["bbox"]
            recognized_users.append({"user_id": user_id, "bbox": bbox})

            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if user_id else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = user_id if user_id else "Невідомо"
            frame = draw_text_ua(frame, label, (x1, y1 - 35), color, font_size=28)

        # Обробка жестів
        gestures = gesture_predictor.predict_gestures(frame)
        for g in gestures:
            if not g["gesture"]: continue
            
            gx1, gy1, gx2, gy2 = g["bbox"]
            cv2.putText(frame, g["gesture"], (gx1, gy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            for user in recognized_users:
                if user["user_id"] and is_hand_of_face(g["bbox"], user["bbox"]):
                    handle_event(user_id=user["user_id"], gesture=g["gesture"], frame=frame)

        cv2.imshow("AI Assistant", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

main()