import asyncio
import os
import threading
import time
import edge_tts
import pygame

def speak_task(text):
    filename = f"voice_{int(time.time())}.mp3"
    try:
        VOICE = "uk-UA-PolinaNeural" 
        
        async def generate():
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(filename)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()

        if os.path.exists(filename):
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.unload()  # Звільняємо файл
            time.sleep(0.2)             # Коротка пауза для ОС, щоб відпустити дескриптор
            os.remove(filename)
            print(f"🗑 Тимчасовий файл {filename} видалено")
    except Exception as e:
        print(f"🔊 Помилка звуку: {e}")
        if os.path.exists(filename):
            try: os.remove(filename)
            except: pass

def speak_async(text):
    threading.Thread(target=speak_task, args=(text,), daemon=True).start()