import os
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

# Константи для LLM
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40

def get_api_key():
    load_dotenv(override=True)
    return os.getenv("GEMINI_API_KEY")