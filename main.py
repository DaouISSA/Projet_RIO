from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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
        places = [
            ('A1', 'free', '2024-01-01 00:00:00', 10, 20),
            ('A2', 'free', '2024-01-01 00:00:00', 10, 40),
            ('B1', 'free', '2024-01-01 00:00:00', 50, 20),
            ('B2', 'free', '2024-01-01 00:00:00', 50, 40)
        ]
        cursor.executemany("INSERT INTO parking_spots VALUES (?,?,?,?,?)", places)
    conn.commit()
    conn.close()

init_db()

# Positions par défaut pour les places connues (fallback si DB incomplète)
DEFAULT_POSITIONS = {
    "A1": (10, 20),
    "A2": (10, 40),
    "B1": (50, 20),
    "B2": (50, 40),
}

class SensorData(BaseModel):
    id: str
    status: str

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur le serveur de gestion du parking", "owner": "Issa"}

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
        response = {
            "status": "success",
            "assigned_spot": spot["id"],
            "message": f"La place {spot['id']} vous est attribuée.",
            "coordinates": {"x": pos_x, "y": pos_y},
            "assigned_coordinates": {"x": pos_x, "y": pos_y},
            "request_received_at": now,
            "itinerary_steps": [
                "Entrez dans le parking",
                f"Suivez l'itinéraire vers x={pos_x}, y={pos_y}"
            ]
        }
        conn.close()
        return response
    conn.close()
    return {"status": "error", "message": "Parking complet"}

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