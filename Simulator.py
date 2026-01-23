import requests
import time
import random

# L'adresse de ton serveur FastAPI
URL = "http://127.0.0.1:8000/update"

# Liste de capteurs imaginaires
SENSORS = ["A1", "A2", "B1", "B2"]

print("--- Demarrage de la simulation des capteurs ---")

while True:
    sensor_id = random.choice(SENSORS)
    status = random.choice(["free", "occupied"])
    
    payload = {
        "id": sensor_id,
        "status": status
    }

    try:
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print(f"[Capteur {sensor_id}] envoye : {status}")
        else:
            print(f"Erreur serveur : {response.status_code}")
            
    except Exception as e:
        print(f"Erreur : Le serveur est-il allume ? {e}")

    # On attend 3 secondes avant le prochain envoi
    time.sleep(5)