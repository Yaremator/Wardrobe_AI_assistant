import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import bcrypt


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "wardrobe.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clothes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS set_clothes_updated_at
            AFTER UPDATE ON clothes
            FOR EACH ROW
            BEGIN
                UPDATE clothes
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = OLD.id;
            END;
            """
        )
        try:
            conn.execute("ALTER TABLE clothes ADD COLUMN season TEXT DEFAULT 'All Seasons';")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE clothes ADD COLUMN ai_description TEXT DEFAULT '';")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    password_hash = _hash_password(password)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?);",
                (username, password_hash),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "Username is already taken."

    return True, "Registration successful."


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?;",
            (username.strip(),),
        ).fetchone()

    if row is None:
        return None
    if not _check_password(password, row["password_hash"]):
        return None

    return {"id": row["id"], "username": row["username"]}


def add_clothing_item(user_id: int, name: str, image_path: str, category: str, season: str = "All Seasons", ai_description: str = "") -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO clothes (user_id, name, image_path, category, season, ai_description)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (user_id, name.strip(), image_path, category.strip(), season.strip(), ai_description.strip()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_clothes(user_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, image_path, category, season, ai_description, created_at, updated_at
            FROM clothes
            WHERE user_id = ?
            ORDER BY created_at DESC;
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_clothing_item(item_id: int, user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, image_path, category, season, ai_description
            FROM clothes
            WHERE id = ? AND user_id = ?;
            """,
            (item_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def update_clothing_item(
    item_id: int,
    user_id: int,
    name: str | None = None,
    category: str | None = None,
    image_path: str | None = None,
    season: str | None = None,
    ai_description: str | None = None,
) -> bool:
    updates = []
    values: list[Any] = []

    if name is not None:
        updates.append("name = ?")
        values.append(name.strip())
    if category is not None:
        updates.append("category = ?")
        values.append(category.strip())
    if image_path is not None:
        updates.append("image_path = ?")
        values.append(image_path)
    if season is not None:
        updates.append("season = ?")
        values.append(season.strip())
    if ai_description is not None:
        updates.append("ai_description = ?")
        values.append(ai_description.strip())

    if not updates:
        return False

    values.extend([item_id, user_id])
    query = f"UPDATE clothes SET {', '.join(updates)} WHERE id = ? AND user_id = ?;"

    with get_connection() as conn:
        cursor = conn.execute(query, values)
        conn.commit()
        return cursor.rowcount > 0


def delete_clothing_item(item_id: int, user_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM clothes WHERE id = ? AND user_id = ?;",
            (item_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
