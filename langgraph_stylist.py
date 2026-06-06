import asyncio
import base64
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import requests
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

from db import get_clothing_item, get_user_clothes
from langfuse_tracing import langfuse_trace, observe_operation


BASE_DIR = Path(__file__).resolve().parent
MAX_VISION_IMAGES = 8
MAX_IMAGE_SIZE = (768, 768)


class ClothingSelection(BaseModel):
    id: int = Field(description="Clothing item ID selected as relevant to the request.")
    reasoning: str = Field(description="Short reason why this item matches user intent.")


class SelectorOutput(BaseModel):
    selected_items: list[ClothingSelection] = Field(default_factory=list)
    reasoning: str = Field(description="Overall short selection reasoning.")


class WeatherRequest(BaseModel):
    should_check_weather: bool = Field(description="True when weather is useful for the request.")
    location: str = Field(default="", description="City/location to check. Use Lviv if user asks for today but gives no city.")
    date: str = Field(default="", description="Optional date in YYYY-MM-DD format. Empty for current weather.")
    reasoning: str = Field(default="", description="Short reason for the decision.")


class ValidatorOutput(BaseModel):
    status: Literal["ok", "needs_revision"] = "ok"
    feedback: str = ""


class WardrobeItem(BaseModel):
    id: int
    name: str
    category: str
    image_path: str
    season: str = "All Seasons"
    ai_description: str = ""


class StylistState(BaseModel):
    user_id: int
    user_query: str
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    wardrobe_items: list[WardrobeItem] = Field(default_factory=list)
    selected_items: list[ClothingSelection] = Field(default_factory=list)
    selection_reasoning: str = ""
    weather_summary: str = ""
    weather_reasoning: str = ""
    vision_notes: str = ""
    final_advice: str = ""
    validator_feedback: str = ""
    missing_images: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _ensure_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")
    return api_key


@lru_cache(maxsize=1)
def _get_fast_llm() -> ChatGoogleGenerativeAI:
    api_key = _ensure_google_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        google_api_key=api_key,
    )


@lru_cache(maxsize=1)
def _get_selector_llm() -> Any:
    return _get_fast_llm().with_structured_output(SelectorOutput)


@lru_cache(maxsize=1)
def _get_weather_router_llm() -> Any:
    return _get_fast_llm().with_structured_output(WeatherRequest)


@lru_cache(maxsize=1)
def _get_validator_llm() -> Any:
    return _get_fast_llm().with_structured_output(ValidatorOutput)


def _reset_cached_llm_clients() -> None:
    """Drop cached async LLM clients bound to a previous asyncio event loop."""
    _get_fast_llm.cache_clear()
    _get_selector_llm.cache_clear()
    _get_weather_router_llm.cache_clear()
    _get_validator_llm.cache_clear()


def _safe_chat_history_context(chat_history: list[dict[str, str]], max_messages: int = 10) -> str:
    if not chat_history:
        return "No previous chat history."

    cropped = chat_history[-max_messages:]
    lines = []

    for message in cropped:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No previous chat history."


def _extract_text_from_llm_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)

        return "\n\n".join(part.strip() for part in parts if part.strip())

    return str(content).strip()


def _parse_feels_like_c(weather_summary: str) -> float | None:
    patterns = [
        r"feels like\s*(-?\d+(?:\.\d+)?)\s*C",
        r"Temperature\s*(-?\d+(?:\.\d+)?)\s*C",
    ]

    for pattern in patterns:
        match = re.search(pattern, weather_summary, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def _weather_requires_outerwear(state: StylistState) -> bool:
    text = f"{state.user_query} {state.weather_summary}".lower()
    feels_like = _parse_feels_like_c(state.weather_summary)

    if feels_like is not None and feels_like <= 14:
        return True

    cold_words = [
        "cold",
        "cool",
        "chilly",
        "windy",
        "rain",
        "холод",
        "прохолод",
        "вітер",
        "вітр",
        "дощ",
        "змерз",
    ]

    return any(word in text for word in cold_words)


def _outerwear_score(item: WardrobeItem) -> int:
    text = f"{item.name} {item.season} {item.ai_description}".lower()
    score = 0

    if item.season == "Winter":
        score += 5
    elif item.season == "Autumn":
        score += 4
    elif item.season == "Spring":
        score += 3
    elif item.season == "All Seasons":
        score += 2

    good_words = [
        "куртка",
        "пальто",
        "тренч",
        "плащ",
        "бомбер",
        "піджак",
        "жакет",
        "coat",
        "jacket",
        "outerwear",
        "warm",
        "тепл",
        "вітр",
    ]

    for word in good_words:
        if word in text:
            score += 2

    return score


def _ensure_outerwear_for_weather(state: StylistState) -> None:
    if not _weather_requires_outerwear(state):
        return

    selected_ids = {item.id for item in state.selected_items}
    item_by_id = {item.id: item for item in state.wardrobe_items}

    selected_has_outerwear = any(
        item_by_id[item_id].category == "Outerwear"
        for item_id in selected_ids
        if item_id in item_by_id
    )

    if selected_has_outerwear:
        return

    outerwear_items = [
        item for item in state.wardrobe_items
        if item.category == "Outerwear"
    ]

    if not outerwear_items:
        return

    best_outerwear = max(outerwear_items, key=_outerwear_score)

    if best_outerwear.id in selected_ids:
        return

    state.selected_items.append(
        ClothingSelection(
            id=best_outerwear.id,
            reasoning=(
                "Додано верхній одяг, бо за погодою для прогулянки може бути прохолодно."
            ),
        )
    )

    state.selected_items = state.selected_items[:MAX_VISION_IMAGES]


@lru_cache(maxsize=32)
def _fetch_weatherapi_summary(location: str, date: str = "") -> str:
    api_key = os.getenv("WEATHERAPI_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WEATHERAPI_KEY is not configured.")

    base_url = "https://api.weatherapi.com/v1"
    endpoint = "/forecast.json" if not date else "/history.json"

    params = {
        "key": api_key,
        "q": location,
    }

    if date:
        params["dt"] = date
    else:
        params["days"] = 3

    response = requests.get(f"{base_url}{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "current" in data:
        current = data["current"]
        condition = current["condition"]["text"]
        temp = current["temp_c"]
        feels_like = current.get("feelslike_c", temp)
        wind = current.get("wind_kph", 0)

        return (
            f"Weather in {location}: {condition}. "
            f"Temperature {temp}C, feels like {feels_like}C, wind {wind} kph."
        )

    forecast_day = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
    condition = forecast_day.get("condition", {}).get("text", "unknown")
    avg_temp = forecast_day.get("avgtemp_c", "unknown")
    max_temp = forecast_day.get("maxtemp_c", "unknown")
    min_temp = forecast_day.get("mintemp_c", "unknown")

    return (
        f"Weather in {location} on {date}: {condition}. "
        f"Avg {avg_temp}C, min {min_temp}C, max {max_temp}C."
    )


@lru_cache(maxsize=32)
def _fetch_openweather_summary(location: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured.")

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "en",
        },
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
        f"Weather in {city}: {description}. Temperature {temp}C "
        f"(feels like {feels_like}C). Humidity {humidity}%, wind {wind} m/s."
    )


def get_weather_summary(location: str, date: str = "") -> str:
    location = location.strip() or "Lviv"
    date = date.strip()

    try:
        return _fetch_weatherapi_summary(location, date)
    except Exception as weatherapi_error:
        if date:
            return f"Could not fetch weather for '{location}' on '{date}': {weatherapi_error}"

        try:
            return _fetch_openweather_summary(location)
        except Exception as openweather_error:
            return (
                f"Could not fetch weather for '{location}'. "
                f"WeatherAPI error: {weatherapi_error}. "
                f"OpenWeather error: {openweather_error}"
            )


@tool
def check_weather(location: str, date: str = "") -> str:
    """Get weather for a location. Date is optional and should be YYYY-MM-DD."""
    return get_weather_summary(location, date)


async def weather_node(state: StylistState, config: RunnableConfig) -> StylistState:
    messages = [
        SystemMessage(
            content=(
                "You are Weather Agent for a wardrobe app. Decide whether weather is useful. "
                "Check weather when the user asks what to wear today/tomorrow, mentions going outside, "
                "mentions a city/date/weather, or asks for a practical outfit. "
                "If city is missing but weather is needed, use Lviv."
            )
        ),
        HumanMessage(
            content=(
                f"Chat history:\n{_safe_chat_history_context(state.chat_history)}\n\n"
                f"Current user query: {state.user_query}"
            )
        ),
    ]

    try:
        result: WeatherRequest = await _get_weather_router_llm().ainvoke(messages, config=config)
        state.weather_reasoning = result.reasoning

        if result.should_check_weather:
            location = result.location.strip() or "Lviv"
            date = result.date.strip()

            if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                date = ""

            state.weather_summary = await asyncio.to_thread(get_weather_summary, location, date)

    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"Weather Agent failed: {exc}")

    return state


async def _load_wardrobe_items(user_id: int) -> list[WardrobeItem]:
    rows = await asyncio.to_thread(get_user_clothes, user_id)

    return [
        WardrobeItem(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            image_path=row["image_path"],
            season=row.get("season") or "All Seasons",
            ai_description=row.get("ai_description") or "",
        )
        for row in rows
    ]


def _build_selector_metadata(items: list[WardrobeItem], max_items: int = 60) -> str:
    compact = items[:max_items]
    lines = []

    for item in compact:
        description = item.ai_description.replace("\n", "; ").strip()

        if len(description) > 180:
            description = description[:177] + "..."

        lines.append(
            f"{item.id}|{item.category}|{item.season}|{item.name}|{description}"
        )

    return "\n".join(lines)


async def selector_node(state: StylistState, config: RunnableConfig) -> StylistState:
    if not state.wardrobe_items:
        state.wardrobe_items = await _load_wardrobe_items(state.user_id)

    if not state.wardrobe_items:
        state.final_advice = "Your wardrobe is empty. Add some items first, then I can suggest outfits."
        return state

    selector_llm = _get_selector_llm()
    metadata = _build_selector_metadata(state.wardrobe_items)

    messages = [
        SystemMessage(
            content=(
                "You are Selector Agent for Smart Wardrobe Assistant. "
                "Select only item IDs from the user's wardrobe. "
                "For a walking/outdoor outfit, select a complete practical outfit: "
                "Top, Bottom, Shoes, and Outerwear if the weather is cold, cool, windy, rainy, or feels like 14C or lower. "
                "Do not assume the user can add clothing that is not selected from the wardrobe. "
                "Do not select random items. Prefer comfort, weather suitability, and visual compatibility."
            )
        ),
        HumanMessage(
            content=(
                f"Chat history:\n{_safe_chat_history_context(state.chat_history)}\n\n"
                f"User query: {state.user_query}\n"
                f"Weather info: {state.weather_summary or 'No weather info.'}\n\n"
                "Available items in format id|category|season|name|ai_description:\n"
                f"{metadata}\n\n"
                "Return selected IDs with concise reasoning. "
                "If weather requires outerwear and there is Outerwear in the wardrobe, include it."
            )
        ),
    ]

    result: SelectorOutput = await selector_llm.ainvoke(messages, config=config)

    state.selected_items = result.selected_items[:MAX_VISION_IMAGES]
    state.selection_reasoning = result.reasoning

    _ensure_outerwear_for_weather(state)

    return state


def _resize_and_encode_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        prepared = ImageOps.exif_transpose(img).convert("RGB")
        prepared.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        prepared.save(buf, format="JPEG", quality=85, optimize=True)

        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


async def _load_selected_items_with_images(
    user_id: int,
    selected_items: list[ClothingSelection],
) -> tuple[list[WardrobeItem], list[str]]:
    resolved_items: list[WardrobeItem] = []
    missing: list[str] = []

    for selected in selected_items:
        row = await asyncio.to_thread(get_clothing_item, selected.id, user_id)

        if row is None:
            missing.append(f"Item ID {selected.id} not found for current user.")
            continue

        resolved_items.append(
            WardrobeItem(
                id=row["id"],
                name=row["name"],
                category=row["category"],
                image_path=row["image_path"],
                season=row.get("season") or "All Seasons",
                ai_description=row.get("ai_description") or "",
            )
        )

    return resolved_items, missing


async def vision_node(state: StylistState, config: RunnableConfig) -> StylistState:
    if state.final_advice:
        return state

    if not state.selected_items:
        state.final_advice = (
            "I could not find clearly relevant items for this request. "
            "Try refining your prompt with occasion, weather, and style."
        )
        return state

    selected_rows, missing_errors = await _load_selected_items_with_images(
        state.user_id,
        state.selected_items,
    )
    state.missing_images.extend(missing_errors)

    multimodal_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are Vision Agent. Analyze selected wardrobe item photos and produce one coherent outfit advice.\n"
                f"Chat history:\n{_safe_chat_history_context(state.chat_history)}\n\n"
                f"User query: {state.user_query}\n"
                f"Weather info: {state.weather_summary or 'No weather info.'}\n\n"
                "Rules:\n"
                "- Be practical and concise.\n"
                "- Do not use emojis.\n"
                "- Do not add poetic filler about coffee, atmosphere, city vibes, etc.\n"
                "- Recommend only selected wardrobe items by name and ID.\n"
                "- Do not invent coats, jackets, accessories, or shoes that were not selected.\n"
                "- If outerwear is needed but no outerwear item is selected, say that no suitable outerwear was found in the wardrobe.\n"
                "- Format: short intro, selected outfit, why it works, one practical note.\n"            ),
        }
    ]

    for item in selected_rows:
        image_path = Path(item.image_path)

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        if not image_path.exists():
            state.missing_images.append(f"Missing image for item ID {item.id}: {image_path}")
            continue

        try:
            data_url = await asyncio.to_thread(_resize_and_encode_image, image_path)
        except Exception as exc:  # noqa: BLE001
            state.missing_images.append(f"Image processing failed for ID {item.id}: {exc}")
            continue

        multimodal_parts.append(
            {
                "type": "text",
                "text": (
                    f"Item {item.id}: {item.name} ({item.category}, {item.season})\n"
                    f"Stored AI description: {item.ai_description or 'No description.'}"
                ),
            }
        )

        multimodal_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    if len(multimodal_parts) == 1:
        state.final_advice = (
            "I found candidate items, but images are unavailable. "
            "Please re-upload missing photos and try again."
        )
        return state

    llm = _get_fast_llm()
    response = await llm.ainvoke([HumanMessage(content=multimodal_parts)], config=config)

    state.vision_notes = _extract_text_from_llm_content(response.content)
    state.final_advice = state.vision_notes

    return state


async def validator_node(state: StylistState, config: RunnableConfig) -> StylistState:
    if not state.final_advice:
        state.final_advice = "Unable to generate advice right now. Please try again."
        return state

    validator_llm = _get_validator_llm()

    messages = [
        SystemMessage(
            content=(
                "You are Validator Agent. Check briefly if stylist answer is safe, coherent, and aligned "
                "with user intent, chat history, weather, and available wardrobe items. Return status + short feedback."
            )
        ),
        HumanMessage(
            content=(
                f"Chat history:\n{_safe_chat_history_context(state.chat_history)}\n\n"
                f"User query: {state.user_query}\n"
                f"Weather info: {state.weather_summary or 'No weather info.'}\n"
                f"Stylist answer: {state.final_advice}\n"
                f"Missing image errors: {state.missing_images}"
            )
        ),
    ]

    result: ValidatorOutput = await validator_llm.ainvoke(messages, config=config)
    state.validator_feedback = result.feedback

    if result.status == "needs_revision":
        state.final_advice = (
            f"{state.final_advice}\n\n"
            f"Note: {result.feedback}"
        )

    return state


def _build_graph():
    graph = StateGraph(StylistState)

    graph.add_node("weather", weather_node)
    graph.add_node("selector", selector_node)
    graph.add_node("vision", vision_node)
    graph.add_node("validator", validator_node)

    graph.set_entry_point("weather")

    graph.add_edge("weather", "selector")
    graph.add_edge("selector", "vision")
    graph.add_edge("vision", "validator")
    graph.add_edge("validator", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_stylist_workflow():
    return _build_graph()


async def run_stylist_workflow(
    user_id: int,
    user_query: str,
    chat_history: list[dict[str, str]] | None = None,
    session_id: str | None = None,
) -> StylistState:
    _reset_cached_llm_clients()

    initial = StylistState(
        user_id=user_id,
        user_query=user_query,
        chat_history=chat_history or [],
    )

    workflow = get_stylist_workflow()

    try:
        with langfuse_trace(
            name="stylist-workflow",
            user_id=user_id,
            session_id=session_id or f"user-{user_id}",
            tags=["langgraph", "stylist", "feature:ai-stylist"],
            trace_input={"user_query": user_query},
        ) as trace:
            result = await workflow.ainvoke(
                initial,
                config=trace.config if trace else None,
            )

            if isinstance(result, StylistState):
                validated = result
            elif isinstance(result, dict):
                validated = StylistState.model_validate(result)
            else:
                validated = StylistState(
                    user_id=user_id,
                    user_query=user_query,
                    chat_history=chat_history or [],
                    final_advice="Unexpected workflow output format.",
                    errors=["Unexpected workflow output type."],
                )

            if trace:
                trace.set_output(
                    {
                        "final_advice": (validated.final_advice or "")[:500],
                        "errors": validated.errors,
                        "selected_item_count": len(validated.selected_items),
                    }
                )

            return validated

    except Exception as exc:  # noqa: BLE001
        return StylistState(
            user_id=user_id,
            user_query=user_query,
            chat_history=chat_history or [],
            final_advice="Workflow failed. Please retry.",
            errors=[str(exc)],
        )


VALID_CATEGORIES = ["Top", "Bottom", "Shoes", "Outerwear", "Dress", "Accessory", "Other"]
VALID_SEASONS = ["All Seasons", "Spring", "Summer", "Autumn", "Winter"]


def _clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return text


def _normalize_category(value: str, name: str = "", description: str = "") -> str:
    text = f"{value} {name} {description}".lower()

    category_keywords = {
        "Top": [
            "top", "shirt", "t-shirt", "tshirt", "tee", "футболка", "сорочка",
            "рубашка", "кофта", "светр", "худі", "hoodie", "sweater",
            "лонгслів", "longsleeve", "blouse", "блузка", "polo", "поло",
        ],
        "Bottom": [
            "bottom", "pants", "trousers", "jeans", "shorts", "skirt",
            "штани", "джинси", "шорти", "спідниця", "брюки",
        ],
        "Shoes": [
            "shoes", "sneakers", "boots", "trainers", "loafers", "sandals",
            "взуття", "кросівки", "кеди", "черевики", "туфлі", "ботинки",
            "сандалі", "slippers",
        ],
        "Outerwear": [
            "outerwear", "jacket", "coat", "parka", "blazer", "куртка",
            "пальто", "плащ", "піджак", "бомбер", "жилетка", "вест",
            "windbreaker", "анорак",
        ],
        "Dress": [
            "dress", "сукня", "плаття",
        ],
        "Accessory": [
            "accessory", "cap", "hat", "belt", "bag", "scarf", "gloves",
            "watch", "necklace", "ring", "окуляри", "шапка", "кепка",
            "ремінь", "сумка", "шарф", "рукавиці", "годинник", "прикраса",
        ],
    }

    value_clean = value.strip()
    if value_clean in VALID_CATEGORIES:
        return value_clean

    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category

    return "Other"


def _normalize_season(value: str, name: str = "", description: str = "") -> str:
    text = f"{value} {name} {description}".lower()

    value_clean = value.strip()
    if value_clean in VALID_SEASONS:
        return value_clean

    if any(word in text for word in ["winter", "зима", "зим", "тепла", "утеплена", "шерсть", "пуховик", "пальто"]):
        return "Winter"

    if any(word in text for word in ["summer", "літо", "літн", "легка", "тонка", "шорти", "майка", "сандалі"]):
        return "Summer"

    if any(word in text for word in ["spring", "весна", "веснян"]):
        return "Spring"

    if any(word in text for word in ["autumn", "fall", "осінь", "осінн", "демісезон"]):
        return "Autumn"

    # Футболки, джинси, кросівки, базові худі часто реально всесезонні
    return "All Seasons"


def analyze_clothing_image_auto(
    image_path: str,
    user_id: int | None = None,
) -> dict[str, str]:
    """Analyze uploaded clothing image and return normalized metadata for wardrobe item."""
    path = Path(image_path)
    trace_input = {"image_filename": path.name}

    with observe_operation(
        name="analyze-clothing-image",
        input_data=trace_input,
        user_id=user_id,
        tags=["feature:wardrobe-upload"],
    ) as observation:
        result = _analyze_clothing_image_impl(image_path)

        if observation is not None:
            observation.update(
                output={
                    "name": result.get("name", ""),
                    "category": result.get("category", "Other"),
                    "season": result.get("season", "All Seasons"),
                }
            )

        return result


def _analyze_clothing_image_impl(image_path: str) -> dict[str, str]:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        return {
            "name": "",
            "category": "Other",
            "season": "All Seasons",
            "description": "",
        }

    try:
        client = genai.Client(api_key=api_key)
        path = Path(image_path)

        if not path.exists():
            return {
                "name": "",
                "category": "Other",
                "season": "All Seasons",
                "description": "",
            }

        with open(path, "rb") as file:
            image_bytes = file.read()

        prompt = """
        You are a precise clothing classifier for a wardrobe app.

        Analyze ONLY the clothing item in the image.
        Return STRICT JSON without markdown.

        Required JSON fields:
        {
          "name": "...",
          "category": "...",
          "season": "...",
          "description": "..."
        }

        Rules:

        1. "name":
        - Use Ukrainian.
        - Short but specific.
        - Examples:
          "Чорна футболка oversize"
          "Сині джинси прямого крою"
          "Білі кросівки"
          "Чорна зимова куртка"

        2. "category":
        Must be exactly ONE of:
        ["Top", "Bottom", "Shoes", "Outerwear", "Dress", "Accessory", "Other"]

        Category guide:
        - Top: футболка, сорочка, худі, светр, кофта, лонгслів, поло
        - Bottom: штани, джинси, шорти, спідниця
        - Shoes: кросівки, кеди, черевики, туфлі, сандалі
        - Outerwear: куртка, пальто, плащ, піджак, бомбер, жилетка
        - Dress: сукня
        - Accessory: шапка, кепка, шарф, сумка, ремінь, окуляри, годинник
        - Other: use ONLY if it is impossible to classify

        3. "season":
        Must be exactly ONE of:
        ["All Seasons", "Spring", "Summer", "Autumn", "Winter"]

        Season guide:
        - Summer: light/thin clothes, shorts, tank tops, sandals
        - Winter: warm/thick clothes, coat, winter jacket, wool, insulated items
        - Spring: light jackets, spring clothes
        - Autumn: hoodies, sweaters, demi-season jackets, boots
        - All Seasons: basic t-shirts, jeans, sneakers, neutral everyday items

        4. "description":
        Use Ukrainian.
        Write each property on a new line.
        Format:
        Колір: ...
        Матеріал: ...
        Крій: ...
        Шар одягу: ...
        Стиль: ...
        Візерунок: ...

        Important:
        Do NOT default to "Other".
        Do NOT default to "All Seasons" unless it is truly a basic all-season item.
        """

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        raw_text = _clean_json_text(response.text)
        parsed = json.loads(raw_text)

        if not isinstance(parsed, dict):
            parsed = {}

        name = str(parsed.get("name", "")).strip()
        raw_category = str(parsed.get("category", "")).strip()
        raw_season = str(parsed.get("season", "")).strip()
        description = str(parsed.get("description", "")).strip()

        category = _normalize_category(raw_category, name, description)
        season = _normalize_season(raw_season, name, description)

        return {
            "name": name,
            "category": category,
            "season": season,
            "description": description,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"Failed to generate AI description: {exc}")
        return {
            "name": "",
            "category": "Other",
            "season": "All Seasons",
            "description": "",
        }