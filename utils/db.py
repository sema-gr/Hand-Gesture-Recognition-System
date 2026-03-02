import sqlite3

def init_db():
    conn = sqlite3.connect('assistant.db')
    cursor = conn.cursor()
    
    cursor.execute("INSERT OR IGNORE INTO apps VALUES (6, 'facebook', 'Facebook.exe', '')")
    cursor.execute("DELETE FROM apps WHERE id = 4")
    
    conn.commit()
    conn.close()

init_db()