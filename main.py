import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

class Item(BaseModel):
    item_id: str = Field(description="ID речі")
    description: str = Field(description="Опис речі")

class Outfit(BaseModel):
    top: Item
    bottom: Item
    outerwear: Optional[Item] = Field(default=None, description="Верхній одяг")
    shoes: Item
    accessories: List[Item]

class AssistantResponse(BaseModel):
    is_outfit_generated: bool = Field(description="True, якщо ти згенерував лук. False, якщо це просто розмова.")
    text_reply: str = Field(description="Твоя відповідь: звичайна розмова, порада стиліста або коментар до луку.")
    outfit_data: Optional[Outfit] = Field(default=None, description="Заповнюй ТІЛЬКИ якщо is_outfit_generated == True.")

def start_wardrobie():
    print("--- Wardrobie Assistant: ONLINE ---")
    print("(напиши 'вихід', щоб зупинити чат)\n")

    system_prompt_text = """
    Ти — Smart Wardrobe Assistant, дружній і висококласний персональний AI-стиліст.

    У тебе є два режими роботи:
    1. Режим розмови: Якщо користувач просто вітається, ставить питання про моду, тренди або просить поради, відповідай як звичайний співрозмовник. 
    2. Режим генерації луку: Якщо користувач прямо просить зібрати образ, підібрати одяг на сьогодні або передає дані про погоду/гардероб, тоді збирай лук.

    Правила для луку: використовуй ТІЛЬКИ ті речі, що є в гардеробі. Враховуй погоду і багатошаровість.
    """

    chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt_text,
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=AssistantResponse,
        )
    )

    while True:
        user_input = input("Ви: ")

        if user_input.lower() in ['вихід', 'exit', 'stop', 'quit']:
            print("Wardrobie: До зустрічі! Заходь за порадами ще.")
            break

        try:
            response = chat_session.send_message(user_input)

            parsed_data = response.parsed

            print(f"\nWardrobie: {parsed_data.text_reply}\n")

            if parsed_data.is_outfit_generated:
                print(f"⚙️ [Система]: Бот згенерував лук! Дані для фронтенду:")
                print(f"Верх: {parsed_data.outfit_data.top.description} (ID: {parsed_data.outfit_data.top.item_id})")
                print(
                    f"Низ: {parsed_data.outfit_data.bottom.description} (ID: {parsed_data.outfit_data.bottom.item_id})")
                print(
                    f"Взуття: {parsed_data.outfit_data.shoes.description} (ID: {parsed_data.outfit_data.shoes.item_id})")
                print("-" * 30 + "\n")

        except Exception as e:
            print(f"\n❌ Помилка: {e}")


if __name__ == "__main__":
    start_wardrobie()