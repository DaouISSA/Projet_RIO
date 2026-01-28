import requests
import time
import random

# ===== CONFIGURATION =====
SERVER_HOST = "127.0.0.1"  # Change en "0.0.0.0" pour réseau
SERVER_PORT = 8000
URL = f"http://{SERVER_HOST}:{SERVER_PORT}/update"

# Places disponibles (selon votre configuration : A1-A3, B1-B3)
SENSORS = ["A1", "A2", "A3", "B1", "B2", "B3"]

# Simuler l'état réel des capteurs
SENSOR_STATES = {spot: "free" for spot in SENSORS}

# ===== PARAMÈTRES DE SIMULATION =====
UPDATE_INTERVAL = 3  # Secondes entre chaque mise à jour
CHANGE_PROBABILITY = 0.3  # 30% de chance de changement d'état

print("--- Démarrage du simulateur de capteurs ---")
print(f"Serveur : {URL}")
print(f"Places monitrées : {SENSORS}")
print()

update_count = 0

while True:
    update_count += 1
    
    # Aléatoirement changer l'état de certains capteurs
    for sensor_id in SENSORS:
        if random.random() < CHANGE_PROBABILITY:
            # Inverser l'état (free ↔ occupied)
            SENSOR_STATES[sensor_id] = "occupied" if SENSOR_STATES[sensor_id] == "free" else "free"
    
    # Envoyer TOUS les capteurs (ou juste celui modifié)
    for sensor_id in SENSORS:
        status = SENSOR_STATES[sensor_id]
        payload = {
            "id": sensor_id,
            "status": status
        }

        try:
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                print(f"[{update_count:03d}] {sensor_id} → {status:8s} ✓")
            else:
                print(f"[{update_count:03d}] {sensor_id} → Erreur {response.status_code}")
                
        except Exception as e:
            print(f"[{update_count:03d}] {sensor_id} → Erreur connexion : {e}")

    print()
    time.sleep(UPDATE_INTERVAL)