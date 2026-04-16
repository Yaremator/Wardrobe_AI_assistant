import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Створюємо папку для фотографій, якщо її ще немає
os.makedirs("uploads", exist_ok=True)

# Налаштування SQLite бази даних (файл wardrobe.db з'явиться у папці проєкту)
engine = create_engine('sqlite:///wardrobe.db', echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Визначаємо модель (таблицю) для речі
class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)        # Наприклад: "Улюблена синя сорочка"
    category = Column(String, nullable=False)    # Наприклад: "Сорочки", "Штани", "Взуття"
    color = Column(String)                       # Колір
    season = Column(String)                      # "Літо", "Зима", "Всесезон"
    image_path = Column(String, nullable=False)  # Шлях до збереженого фото

# Створюємо таблицю в базі даних (якщо вона ще не створена)
Base.metadata.create_all(bind=engine)

def get_db():
    """Функція для безпечного отримання сесії бази даних"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()