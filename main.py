from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Serveur Parking Issa - SQLite")

# --- CONFIGURATION CORS (Indispensable pour test.html) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect("parking.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_spots (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_update TEXT,
            pos_x INTEGER,
            pos_y INTEGER
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM parking_spots")
    if cursor.fetchone()[0] == 0:
        places = []

        # Artère principale
        MAIN_Y = 100

        # Allées perpendiculaires (X fixe)
        allees_x = {
            "A": 40,
            "B": 80,
            "C": 120,
            "D": 160,
        }

        # Paramètres des places
        spacing_y = 12  # espacement entre places
        places_per_side = 4

        for allee, x in allees_x.items():
            # Places à gauche 
            for i in range(places_per_side):
                y = MAIN_Y - (i + 1) * spacing_y
                places.append((
                    f"{allee}{i + 1}",
                    "free",
                    "2024-01-01 00:00:00",
                    x,
                    y
                ))

            # Places à droite
            for i in range(places_per_side):
                y = MAIN_Y + (i + 1) * spacing_y
                places.append((
                    f"{allee}{i + 5}",
                    "free",
                    "2024-01-01 00:00:00",
                    x,
                    y
                ))

    cursor.executemany(
        "INSERT INTO parking_spots VALUES (?,?,?,?,?)",
        places
    )

    conn.commit()
    conn.close()

init_db()

# Positions par défaut pour les places connues (fallback si DB incomplète)
DEFAULT_POSITIONS = {}
for i in range(1, 4):
    DEFAULT_POSITIONS[f'A{i}'] = (0 + (i - 1) * 18, 20)
    DEFAULT_POSITIONS[f'B{i}'] = (0 + (i - 1) * 18, 70)

class SensorData(BaseModel):
    id: str
    status: str

@app.get("/")
async def read_root():
    """Servir la page HTML du parking"""
    return FileResponse("test.html", media_type="text/html")

@app.post("/update")
async def update_spot(sensor: SensorData):
    conn = sqlite3.connect("parking.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Utilisation de UPDATE pour ne pas écraser les coordonnées x, y existantes
    cursor.execute(
        "UPDATE parking_spots SET status = ?, last_update = ? WHERE id = ?",
        (sensor.status, now, sensor.id),
    )
    conn.commit()
    conn.close()
    
    return {"message": f"Place {sensor.id} mise à jour", "status": sensor.status}

@app.get("/parking/status")
async def get_all_status():
    conn = sqlite3.connect("parking.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM parking_spots")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- ROUTE UNIQUE POUR LA RÉSERVATION (Version Guillaume avec coordonnées) ---
@app.post("/booking/request")
async def request_spot():
    conn = sqlite3.connect("parking.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # On cherche la place libre la plus proche
    cursor.execute("SELECT * FROM parking_spots WHERE status = 'free' ORDER BY pos_x, pos_y LIMIT 1")
    spot = cursor.fetchone()
    
    if spot:
        # S'assurer que les coordonnées existent (backfill si nécessaire)
        pos_x = spot["pos_x"]
        pos_y = spot["pos_y"]
        if pos_x is None or pos_y is None:
            default = DEFAULT_POSITIONS.get(spot["id"])
            if default is not None:
                pos_x, pos_y = default
                cursor.execute(
                    "UPDATE parking_spots SET pos_x = ?, pos_y = ? WHERE id = ?",
                    (pos_x, pos_y, spot["id"]),
                )
                conn.commit()
            else:
                # Dernier recours: valeurs neutres
                pos_x, pos_y = 0, 0
        # Marquer la place comme occupée et mettre à jour la date
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "UPDATE parking_spots SET status = ?, last_update = ? WHERE id = ?",
            ("occupied", now, spot["id"]),
        )
        conn.commit()
        
        # Calculer l'itinéraire précis
        itinerary_waypoints = calculate_itinerary(pos_x, pos_y, spot["id"])
        
        response = {
            "status": "success",
            "assigned_spot": spot["id"],
            "message": f"La place {spot['id']} vous est attribuée.",
            "coordinates": {"x": pos_x, "y": pos_y},
            "assigned_coordinates": {"x": pos_x, "y": pos_y},
            "request_received_at": now,
            "itinerary_steps": itinerary_waypoints["steps"],
            "waypoints": itinerary_waypoints["waypoints"]
        }
        conn.close()
        return response
    conn.close()
    return {"status": "error", "message": "Parking complet"}

def calculate_itinerary(target_x, target_y, spot_id):
    """Calcule un itinéraire réaliste vers la place spécifiée"""
    
    # Entrée du parking (en bas au centre)
    entry_x = 100
    entry_y = 100  # En bas du parking
    
    # Allée principale (centre entre rangées A et B)
    allee_y = 45
    
    # Déterminer la rangée (A=haut ou B=bas)
    rangee = spot_id[0]
    is_rangee_a = (rangee == 'A')
    
    # Créer les points de passage simplifié
    waypoints = [
        {"x": entry_x, "y": entry_y},  # 1. Entrée du parking
        {"x": entry_x, "y": allee_y},  # 2. Monter à l'allée principale
        {"x": target_x, "y": allee_y},  # 3. Avancer jusqu'à l'alignement de la place
        {"x": target_x, "y": target_y}     # 4. Place finale
    ]
    
    # Direction selon la rangée
    direction = "haut" if is_rangee_a else "bas"
    
    steps = [
        f"Point 1 : Entrée du parking",
        f"Point 2 : Monter à l'allée principale",
        f"Point 3 : Avancer jusqu'à l'alignement de la place {spot_id} ({direction})",
        f"Point 4 : Place finale {spot_id}"
    ]
    
    return {
        "waypoints": waypoints,
        "steps": steps
    }

# --- UTILITAIRE: remettre toutes les places en 'free' ---
def free_all_spots() -> int:
    conn = sqlite3.connect("parking.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "UPDATE parking_spots SET status = ?, last_update = ?",
        ("free", now),
    )
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM parking_spots WHERE status = 'free'")
    freed_count = cursor.fetchone()[0]
    conn.close()
    return freed_count

@app.post("/parking/reset")
async def reset_parking():
    count = free_all_spots()
    
    return {
        "status": "success",
        "freed_count": count,
        "message": "Toutes les places sont maintenant libres."
    }