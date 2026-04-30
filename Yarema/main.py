import streamlit as st
import os
import uuid
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

# Імпорти з наших модулів
from llm_service import get_conversation_chain
from database import Item, get_db
import config
from dotenv import set_key, unset_key

# --- Налаштування сторінки ---
st.set_page_config(page_title="Wardrobe AI", page_icon="👗", layout="wide") # Змінили layout на wide для зручної галереї
st.title("Wardrobe AI Assistant")

# --- Ініціалізація стану сесії ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Бічна панель: Налаштування ---
env_file = os.path.join(os.path.dirname(__file__), ".env")

with st.sidebar:
    st.header("⚙️ Налаштування")
    
    current_key = config.get_api_key()
    if current_key:
        st.success("✅ API ключ встановлено")
    else:
        st.warning("⚠️ API ключ не встановлено")
        
    with st.form("api_key_form"):
        new_key = st.text_input("Gemini API Key", value=current_key if current_key else "", type="password")
        col1, col2 = st.columns(2)
        with col1:
            save_btn = st.form_submit_button("Зберегти")
        with col2:
            delete_btn = st.form_submit_button("Видалити")
            
        if save_btn:
            if new_key:
                if not os.path.exists(env_file):
                    open(env_file, 'w').close()
                set_key(env_file, "GEMINI_API_KEY", new_key)
                st.success("Ключ збережено!")
                st.rerun()
            else:
                st.error("Введіть ключ перед збереженням.")
        
        if delete_btn:
            if os.path.exists(env_file):
                unset_key(env_file, "GEMINI_API_KEY")
            os.environ.pop("GEMINI_API_KEY", None)
            st.success("Ключ видалено!")
            st.rerun()
            
    st.markdown("---")
    st.markdown("[Отримати API ключ Gemini](https://aistudio.google.com/app/apikey)")

# --- Створення вкладок ---
tab1, tab2, tab3 = st.tabs(["💬 Чат з ШІ", "👕 Мій гардероб", "➕ Додати річ"])

# ==========================================
# Вкладка 1: Чат з ШІ (Твій попередній код)
# ==========================================
with tab1:
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        if st.button("🗑️ Очистити чат"):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if user_query := st.chat_input("Напишіть про свій гардероб..."):
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # 1. Дістаємо всі речі з БД
            db = next(get_db())
            items = db.query(Item).all()
            
            # 2. Формуємо текстовий опис гардероба для ШІ
            if not items:
                wardrobe_text = "Гардероб поки порожній."
            else:
                # Робимо гарний список
                items_list = [f"- {item.category}: {item.name} (Колір: {item.color}, Сезон: {item.season}, Деталі: {item.ai_description or 'Відсутні'})" for item in items]
                wardrobe_text = "\n".join(items_list)
            
            # 3. Беремо історію
            history_context = st.session_state.chat_history[-11:-1] 
            
            # 4. Передаємо і запит, і історію, І ГАРДЕРОБ!
            try:
                chain = get_conversation_chain()
                response = st.write_stream(
                    chain.stream({
                        "input": user_query,
                        "history": history_context,
                        "wardrobe_context": wardrobe_text # Передаємо змінні для промпту
                    })
                )
                st.session_state.chat_history.append(AIMessage(content=response))
            except ValueError as e:
                st.error(str(e))
                st.session_state.chat_history.pop() # Видаляємо запит користувача, якщо сталася помилка

# ==========================================
# Вкладка 2: Галерея Гардероба
# ==========================================
with tab2:
    st.header("Твої речі")
    
    # Отримуємо всі речі з бази
    db = next(get_db())
    items = db.query(Item).all()
    
    if not items:
        st.info("Твій гардероб поки порожній. Перейди у вкладку 'Додати річ'!")
    else:
        # Виводимо речі сіткою по 4 в ряд
        cols = st.columns(4)
        for idx, item in enumerate(items):
            with cols[idx % 4]:
                st.image(item.image_path, width="stretch")
                st.markdown(f"**{item.name}**")
                st.caption(f"{item.category} | Колір: {item.color} | Сезон: {item.season}")
                if getattr(item, "ai_description", None):
                    st.caption(f"✨ *{item.ai_description}*")
                
                with st.popover("✏️ Редагувати"):
                    with st.form(f"edit_form_{item.id}"):
                        new_name = st.text_input("Назва", value=item.name)
                        new_category = st.text_input("Категорія", value=item.category)
                        new_color = st.text_input("Колір", value=item.color)
                        new_season = st.text_input("Сезон", value=item.season)
                        new_ai_desc = st.text_area("Опис", value=getattr(item, "ai_description", "") or "")
                        if st.form_submit_button("Зберегти"):
                            try:
                                db_session = next(get_db())
                                item_to_edit = db_session.query(Item).get(item.id)
                                if item_to_edit:
                                    item_to_edit.name = new_name
                                    item_to_edit.category = new_category
                                    item_to_edit.color = new_color
                                    item_to_edit.season = new_season
                                    item_to_edit.ai_description = new_ai_desc
                                    db_session.commit()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Помилка: {e}")

                if st.button("🗑️ Видалити", key=f"del_{item.id}"):
                    if os.path.exists(item.image_path):
                        try:
                            os.remove(item.image_path)
                        except OSError as e:
                            st.error(f"Помилка видалення файлу: {e}")
                    
                    try:
                        db_session = next(get_db())
                        item_to_delete = db_session.query(Item).get(item.id)
                        if item_to_delete:
                            db_session.delete(item_to_delete)
                            db_session.commit()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка видалення з БД: {e}")

# ==========================================
# Вкладка 3: Форма додавання речі
# ==========================================
with tab3:
    st.header("Додати нову річ")
    
    with st.form("add_item_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Назва (напр. Улюблена чорна футболка)*")
            st.info("💡 ШІ автоматично визначить категорію, колір, сезон та згенерує ключові фрази з фото!")
            
        with col2:
            uploaded_file = st.file_uploader("Завантаж фото речі*", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                st.image(uploaded_file, width=200)
                
        submitted = st.form_submit_button("Зберегти в гардероб")
        
        if submitted:
            if not name or not uploaded_file:
                st.error("Будь ласка, заповніть назву та завантажте фото.")
            else:
                # Генеруємо унікальне ім'я для файлу, щоб вони не перезаписували одне одного
                file_extension = uploaded_file.name.split('.')[-1]
                unique_filename = f"{uuid.uuid4()}.{file_extension}"
                file_path = os.path.join("uploads", unique_filename)
                
                # Зберігаємо фото на диск
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Викликаємо ШІ для аналізу
                with st.spinner("ШІ аналізує фотографію... 🪄"):
                    from llm_service import analyze_image_with_vision
                    try:
                        ai_data = analyze_image_with_vision(file_path)
                    except ValueError as e:
                        st.error(str(e))
                        ai_data = {
                            "category": "Невідомо",
                            "color": "Невідомо",
                            "season": "Невідомо",
                            "ai_description": ""
                        }
                
                # Записуємо дані в базу
                db = next(get_db())
                new_item = Item(
                    name=name,
                    category=ai_data.get("category", "Аксесуари"),
                    color=ai_data.get("color", "Невідомо"),
                    season=ai_data.get("season", "Будь-який"),
                    ai_description=ai_data.get("ai_description", ""),
                    image_path=file_path
                )
                db.add(new_item)
                db.commit()
                
                st.success(f"Річ '{name}' успішно додана! (Визначено: {new_item.category}, {new_item.color})")
                st.rerun() # Оновлюємо сторінку, щоб річ одразу з'явилася в галереї