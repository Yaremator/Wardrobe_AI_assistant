import os
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

# Константи для LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40