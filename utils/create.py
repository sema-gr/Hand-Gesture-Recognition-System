import sqlite3

def create_base():
    conn = sqlite3.connect('assistant.db')
    cursor = conn.cursor()

    # 1. Таблиця програм
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            app_name TEXT NOT NULL,
            path TEXT
        )
    ''')

    # Таблиця фраз
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            response TEXT NOT NULL
        )
    ''')

    apps_data = [
        ('телеграм', 'Telegram.exe', ''), 
        ('нотатки', 'notepad.exe', 'C:/Windows/System32/notepad.exe'),
        ('браузер', 'chrome.exe', '')
    ]
    
    commands_data = [
        ('хто я', 'Ви — {user}. Я впізнала вас за жестом.'),
        ('як справи', 'У мене все чудово, готова до роботи!'),
        ('що ти вмієш', 'Я можу впізнавати вас, відкривати програми та виконувати голосові команди.')
    ]

    cursor.executemany("INSERT INTO apps (keyword, app_name, path) VALUES (?, ?, ?)", apps_data)
    cursor.executemany("INSERT INTO commands (keyword, response) VALUES (?, ?)", commands_data)

    conn.commit()
    conn.close()
    print("Базу даних 'assistant.db' успішно створено та наповнено!")

create_base()