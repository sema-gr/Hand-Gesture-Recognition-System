import os
import platform
import subprocess
from utils.voice_engine import speak_async

def find_app_path(app_name):
    if platform.system() != "Windows":
        return None
    
    try:
        speak_async("Зачекайте! Відбувається пошук")
        print("Відбувається пошук програми, не хвилюйтесь!!")
        print("Голосові команди не працюють!")
        # Швидка перевірка через команду де (працює для багатьох системних програм)
        result = subprocess.check_output(f"where /R C:\\ {app_name}", shell=True, stderr=subprocess.STDOUT)
        paths = result.decode('cp1251').split('\r\n')
        return paths[0]
    except:
        # Якщо не знайшло на диску C, можна додати пошук у конкретних папках
        common_folders = [
            os.path.expanduser("~/AppData/Roaming"),
            "C:/Program Files",
            "C:/Program Files (x86)"
        ]
        for folder in common_folders:
            for root, dirs, files in os.walk(folder):
                if app_name in files:
                    return os.path.join(root, app_name)
    return None