from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import base64
import json

import config
import prompts

def get_llm():
    """Ініціалізує та повертає об'єкт моделі."""
    api_key = config.get_api_key()
    if not api_key:
        raise ValueError("API ключ не знайдено. Будь ласка, введіть його в налаштуваннях.")
        
    return ChatGoogleGenerativeAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        api_key=api_key
    )

def get_conversation_chain():
    """Створює ланцюжок для обробки чату."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()

def analyze_image_with_vision(image_path: str) -> dict:
    """Аналізує зображення речі та повертає JSON-словник з деталями."""
    llm = get_llm()
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Визначаємо MIME-тип на основі розширення (для простоти беремо jpeg або png)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompts.VISION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            }
        ]
    )
    
    response = llm.invoke([message])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"Помилка парсингу: {e}")
        return {
            "category": "Аксесуари",
            "color": "Невідомо",
            "season": "Будь-який",
            "ai_description": f"Не вдалося проаналізувати: {str(e)}"
        }