from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
import prompts

def get_llm():
    """Ініціалізує та повертає об'єкт моделі."""
    return ChatGoogleGenerativeAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        api_key=config.GEMINI_API_KEY
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