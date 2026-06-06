import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import bcrypt
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "wardrobe.db"


class UserProfileData(BaseModel):
    user_id: int
    full_name: str = ""
    gender: str = ""
    style_preferences: str = ""
    chest_cm: float = Field(default=0, ge=0)
    waist_cm: float = Field(default=0, ge=0)
    hips_cm: float = Field(default=0, ge=0)
    shoulder_cm: float = Field(default=0, ge=0)
    sleeve_cm: float = Field(default=0, ge=0)
    inseam_cm: float = Field(default=0, ge=0)
    foot_length_cm: float = Field(default=0, ge=0)


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
                season TEXT DEFAULT 'All Seasons',
                ai_description TEXT DEFAULT '',
                color TEXT DEFAULT '',
                style_tag TEXT DEFAULT '',
                is_in_laundry INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                style_preferences TEXT DEFAULT '',
                chest_cm REAL DEFAULT 0,
                waist_cm REAL DEFAULT 0,
                hips_cm REAL DEFAULT 0,
                shoulder_cm REAL DEFAULT 0,
                sleeve_cm REAL DEFAULT 0,
                inseam_cm REAL DEFAULT 0,
                foot_length_cm REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        _migrate_clothes_schema(conn)
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
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS set_user_profiles_updated_at
            AFTER UPDATE ON user_profiles
            FOR EACH ROW
            BEGIN
                UPDATE user_profiles
                SET updated_at = CURRENT_TIMESTAMP
                WHERE user_id = OLD.user_id;
            END;
            """
        )
        conn.commit()


def _migrate_clothes_schema(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(clothes);").fetchall()
    }
    if "color" not in existing_columns:
        conn.execute("ALTER TABLE clothes ADD COLUMN color TEXT DEFAULT '';")
    if "style_tag" not in existing_columns:
        conn.execute("ALTER TABLE clothes ADD COLUMN style_tag TEXT DEFAULT '';")
    if "is_in_laundry" not in existing_columns:
        conn.execute("ALTER TABLE clothes ADD COLUMN is_in_laundry INTEGER NOT NULL DEFAULT 0;")
    if "season" not in existing_columns:
        conn.execute("ALTER TABLE clothes ADD COLUMN season TEXT DEFAULT 'All Seasons';")
    if "ai_description" not in existing_columns:
        conn.execute("ALTER TABLE clothes ADD COLUMN ai_description TEXT DEFAULT '';")


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


def add_clothing_item(
    user_id: int,
    name: str,
    image_path: str,
    category: str,
    season: str = "All Seasons",
    ai_description: str = "",
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO clothes (user_id, name, image_path, category, season, ai_description)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                user_id,
                name.strip(),
                image_path,
                category.strip(),
                season.strip(),
                ai_description.strip(),
            ),
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


def get_user_profile(user_id: int) -> UserProfileData:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                user_id, full_name, gender, style_preferences,
                chest_cm, waist_cm, hips_cm, shoulder_cm, sleeve_cm, inseam_cm, foot_length_cm
            FROM user_profiles
            WHERE user_id = ?;
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return UserProfileData(user_id=user_id)
    return UserProfileData.model_validate(dict(row))


def upsert_user_profile(profile: UserProfileData) -> None:
    data = profile.model_dump()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, full_name, gender, style_preferences,
                chest_cm, waist_cm, hips_cm, shoulder_cm, sleeve_cm, inseam_cm, foot_length_cm
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                full_name = excluded.full_name,
                gender = excluded.gender,
                style_preferences = excluded.style_preferences,
                chest_cm = excluded.chest_cm,
                waist_cm = excluded.waist_cm,
                hips_cm = excluded.hips_cm,
                shoulder_cm = excluded.shoulder_cm,
                sleeve_cm = excluded.sleeve_cm,
                inseam_cm = excluded.inseam_cm,
                foot_length_cm = excluded.foot_length_cm;
            """,
            (
                data["user_id"],
                data["full_name"],
                data["gender"],
                data["style_preferences"],
                data["chest_cm"],
                data["waist_cm"],
                data["hips_cm"],
                data["shoulder_cm"],
                data["sleeve_cm"],
                data["inseam_cm"],
                data["foot_length_cm"],
            ),
        )
        conn.commit()
