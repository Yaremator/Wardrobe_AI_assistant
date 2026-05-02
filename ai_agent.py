import os
from typing import Any

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from db import get_user_clothes


SYSTEM_PROMPT = (
    "You are Smart Wardrobe Assistant. "
    "You are friendly, practical, and concise. "
    "Use tool results when available. "
    "When the user asks what to wear, use weather and wardrobe data before advice."
)


def _history_to_messages(chat_history: list[dict[str, str]], memory_window: int) -> list[Any]:
    if not chat_history:
        return []

    max_history_items = max(memory_window * 2, 0)
    cropped = chat_history[-max_history_items:]
    messages: list[Any] = []
    for item in cropped:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _build_tools(user_id: int):
    @tool
    def get_current_weather(location: str) -> str:
        """Get current weather for the specified city/location."""
        api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
        if not api_key:
            return (
                "OpenWeather API key is not configured. "
                "Please set OPENWEATHER_API_KEY in .env."
            )

        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": api_key, "units": "metric", "lang": "en"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            description = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]
            city = data.get("name", location)
            return (
                f"Weather in {city}: {description}. "
                f"Temperature {temp}C (feels like {feels_like}C). "
                f"Humidity {humidity}%, wind {wind} m/s."
            )
        except requests.RequestException as exc:
            return f"Could not fetch weather for '{location}': {exc}"

    @tool
    def get_user_wardrobe() -> str:
        """Get current user's wardrobe items from the SQLite database."""
        clothes = get_user_clothes(user_id)
        if not clothes:
            return "User wardrobe is empty."

        lines = [
            f"- {item['name']} ({item['category']})"
            for item in clothes
        ]
        return "User wardrobe items:\n" + "\n".join(lines)

    return [get_current_weather, get_user_wardrobe]


def generate_stylist_reply(
    user_id: int,
    user_message: str,
    chat_history: list[dict[str, str]],
    memory_window: int = 10,
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return (
            "Gemini API key is missing. Add GOOGLE_API_KEY to your .env file "
            "to enable AI stylist responses."
        )

    tools = _build_tools(user_id)
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.3,
        google_api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_to_messages(chat_history, memory_window))
    messages.append(HumanMessage(content=user_message))

    # Handle multi-step tool calling loop.
    for _ in range(3):
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)
        tool_calls = getattr(ai_response, "tool_calls", None) or []
        if not tool_calls:
            content = ai_response.content

            # Якщо content — це список (як у вашому лозі), витягуємо текст
            if isinstance(content, list):
                # Проходимо по списку і збираємо всі 'text' частини
                text_parts = [item['text'] for item in content if isinstance(item, dict) and 'text' in item]
                return "".join(text_parts)

            # Якщо це звичайний рядок
            return str(content)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool is None:
                tool_result = f"Tool '{tool_name}' is unavailable."
            else:
                try:
                    tool_result = selected_tool.invoke(tool_args)
                except Exception as exc:  # noqa: BLE001
                    tool_result = f"Tool '{tool_name}' failed: {exc}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "I could not complete the full tool workflow right now. Please try again."
