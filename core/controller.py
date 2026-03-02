import asyncio
import platform
import sqlite3
import cv2
import time
import webbrowser
import os
import threading
import edge_tts
import numpy as np
import pygame
from PIL import ImageFont, ImageDraw, Image
import speech_recognition as sr
import subprocess
from utils.finder import find_app_path
from utils.voice_engine import speak_async, speak_task

# Ініціалізація pygame mixer
pygame.mixer.init()

# Контейнер для стану (спільна пам'ять між потоками)
class AssistantState:
    def __init__(self):
        self.user_id = None        # Останній впізнаний (для авто-вітання)
        self.active_user = None    # Той, хто звернувся через жест
        self.last_active_time = 0

state = AssistantState()

# Таймери та змінні
last_event_time = {"greet": 0, "gesture": 0, "action": 0}
last_user_id = None
current_message = ""
COOLDOWN_SEC = 2
ACTION_COOLDOWN = 5
current_message_color = (0, 255, 0)
greeted_users = set()
message_expiry_time = 0
spoken_gestures = {"thumbs_up": False, "stop": False}

recognizer = sr.Recognizer()
mic = sr.Microphone()

def voice_callback(rec, audio):
    """Callback функція: викликається автоматично при виявленні голосу"""
    try:
        # 1. Розпізнаємо текст
        command = rec.recognize_google(audio, language='uk-UA').lower()
        
        # 2. Викликаємо обробник, ПЕРЕДАЮЧИ актуальний user_id зі стану
        process_voice_command(command, state.user_id)
    except sr.UnknownValueError:
        print("Не розпізнано слів")
    except Exception as e:
        print(f"🎙 Помилка розпізнавання: {e}")

def start_voice_assistant():
    """Запуск прослуховування"""
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        # Важливо: передаємо посилання на функцію voice_callback
        recognizer.listen_in_background(mic, voice_callback)
        print("👂 Голосовий асистент запущено...")
    except Exception as e:
        print(f"❌ Не вдалося запустити мікрофон: {e}")

def process_voice_command(command, user_id):
    print(f"⚙️ Виконую команду '{command}'", flush=True)
    current_focus = state.active_user if state.active_user else user_id
    
    conn = sqlite3.connect('assistant.db')
    cursor = conn.cursor()

    # 1. Перевірка простих команд із бази
    cursor.execute("SELECT response FROM commands WHERE ? LIKE '%' || keyword || '%'", (command,))
    cmd_row = cursor.fetchone()
    if cmd_row:
        response = cmd_row[0].replace("{user}", str(current_focus))
        speak_async(response)
        conn.close()
        return

    # 2. Перевірка додатків із бази
    cursor.execute("SELECT app_name, path FROM apps WHERE ? LIKE '%' || keyword || '%'", (command,))
    app_row = cursor.fetchone()
    
    if app_row:
        app_name, saved_path = app_row
        if not current_focus:
            speak_async("Будь ласка, авторизуйтесь жестом для доступу.")
            conn.close()
            return

        speak_async(f"Відкриваю {app_name}")
        
        # Логіка для Mac
        if platform.system() == "Darwin":
            clean_name = app_name.replace(".exe", "")
            subprocess.Popen(['/usr/bin/open', '-a', clean_name])
        
        # Логіка для Windows
        else:
            path = saved_path if saved_path and os.path.exists(saved_path) else find_app_path(app_name)
            if path:
                os.startfile(path)
                # Оновлюємо шлях у БД, щоб наступного разу не шукати довго
                cursor.execute("UPDATE apps SET path = ? WHERE app_name = ?", (path, app_name))
                conn.commit()
            else:
                speak_async(f"Я не знайшла {app_name} на диску")

    # 3. Спеціальна команда виходу
    elif "вихід" in command:
        conn.close()
        speak_task("Бувайте!")
        os._exit(0)

    conn.close()

# --- ВІЗУАЛІЗАЦІЯ ТА ПОДІЇ ---
def draw_ukr_text(img, text, position, font_size=35, color=(0, 255, 0)):
    h, w = img.shape[:2]
    current_os = platform.system() # 'Windows', 'Darwin' (Mac) або 'Linux'

    # 1. Адаптивне масштабування
    if current_os == "Darwin":
        # На Mac Retina щільність пікселів вища, тому множимо сильніше
        retina_scale = w / 640 
        final_font_size = int(font_size * retina_scale * 1.2)
    else:
        # На Windows зазвичай достатньо стандартного масштабування
        scale = w / 1280
        final_font_size = int(font_size * (scale if scale > 1 else 1))

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. Вибір шляху до шрифту залежно від ОС
    if current_os == "Windows":
        font_path = "C:/Windows/Fonts/arial.ttf"
    elif current_os == "Darwin":
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Для Linux

    try:
        font = ImageFont.truetype(font_path, final_font_size)
    except:
        # Якщо файл не знайдено, намагаємося завантажити системний шрифт за назвою
        try:
            font = ImageFont.truetype("arial.ttf", final_font_size)
        except:
            font = ImageFont.load_default()
    
    pil_color = (color[2], color[1], color[0])
    
    # 3. Малюємо тінь (чорна підкладка)
    x, y = position
    offset = max(1, int(final_font_size / 20)) # Товщина тіні залежить від розміру
    draw.text((x + offset, y + offset), text, font=font, fill=(0, 0, 0))
    
    # Малюємо основний текст
    draw.text(position, text, font=font, fill=pil_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def handle_event(user_id, gesture, frame):
    global last_user_id, current_message, current_message_color, message_expiry_time, greeted_users
    current_time = time.time()
    
    can_update_text = current_time > message_expiry_time

    # --- ЛОГІКА ОБЛИЧЧЯ (Тільки один раз для кожного) ---
    if user_id:
        # Оновлюємо активного юзера для голосу (це ми вже робили)
        state.user_id = user_id
        
        # ПЕРЕВІРКА: Чи ми вже вітали цю конкретну людину?
        if user_id not in greeted_users:
            current_message = f"Привіт, {user_id}!"
            current_message_color = (0, 255, 0)
            speak_async(f"Привіт, {user_id}. Рада тебе бачити")
            
            # Додаємо в список "привітаних", щоб не повторювати
            greeted_users.add(user_id)
            
            # Заморожуємо повідомлення на екрані
            message_expiry_time = current_time + 4.0
            last_user_id = user_id

    # --- ЛОГІКА ЖЕСТІВ ---
    if gesture and (current_time - last_event_time["gesture"] > COOLDOWN_SEC):
        last_event_time["gesture"] = current_time
        
        # Оновлюємо активного користувача для команд, якщо є жест
        state.active_user = user_id
        state.last_active_time = current_time

        if can_update_text:
            if gesture == "wave":
                webbrowser.open("https://www.youtube.com")
                current_message = "Відкриваю YouTube..."
                current_message_color = (0, 255, 255)
                speak_async("Відкриваю ютуб")
                message_expiry_time = current_time + 2.0
                
            elif gesture == "thumbs_up":
                current_message = "Круто!"
                current_message_color = (0, 255, 0)
                speak_async("Це просто круто")
                message_expiry_time = current_time + 1.5
            
            elif gesture == "victory":
                current_message = "Перемога!"
                current_message_color = (255, 0, 255)
                speak_async("Все буде Україна")
                message_expiry_time = current_time + 2.0

    # Очищення тексту після завершення таймера
    if gesture is None and can_update_text:
        current_message = ""

    # Малювання (current_message малюється тільки якщо воно не порожнє)
    if current_message:
        temp_frame = draw_ukr_text(frame, current_message, (20, 40), font_size=40, color=current_message_color)
        np.copyto(frame, temp_frame)