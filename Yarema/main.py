import streamlit as st
import os
import uuid
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

# Імпорти з наших модулів
from llm_service import get_conversation_chain
from database import Item, get_db

# --- Налаштування сторінки ---
st.set_page_config(page_title="Wardrobe AI", page_icon="👗", layout="wide") # Змінили layout на wide для зручної галереї
st.title("Wardrobe AI Assistant")

# --- Ініціалізація стану сесії ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = get_conversation_chain()

# --- Створення вкладок ---
tab1, tab2, tab3 = st.tabs(["💬 Чат з ШІ", "👕 Мій гардероб", "➕ Додати річ"])

# ==========================================
# Вкладка 1: Чат з ШІ (Твій попередній код)
# ==========================================
with tab1:
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
                # Робимо гарний список, наприклад: "- Сорочки/Футболки: Улюблена чорна футболка (Колір: чорний, Сезон: Літо)"
                items_list = [f"- {item.category}: {item.name} (Колір: {item.color}, Сезон: {item.season})" for item in items]
                wardrobe_text = "\n".join(items_list)
            
            # 3. Беремо історію
            history_context = st.session_state.chat_history[-11:-1] 
            
            # 4. Передаємо і запит, і історію, І ГАРДЕРОБ!
            response = st.write_stream(
                st.session_state.chain.stream({
                    "input": user_query,
                    "history": history_context,
                    "wardrobe_context": wardrobe_text # Передаємо змінні для промпту
                })
            )
        
        st.session_state.chat_history.append(AIMessage(content=response))

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
                st.caption(f"Категорія: {item.category} | Колір: {item.color}")

# ==========================================
# Вкладка 3: Форма додавання речі
# ==========================================
with tab3:
    st.header("Додати нову річ")
    
    with st.form("add_item_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Назва (напр. Улюблена чорна футболка)*")
            category = st.selectbox("Категорія*", ["Верхній одяг", "Светри/Худі", "Сорочки/Футболки", "Штани/Джинси", "Взуття", "Аксесуари"])
            color = st.text_input("Колір")
            season = st.selectbox("Сезон", ["Будь-який", "Зима", "Весна", "Літо", "Осінь"])
            
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
                
                # Записуємо дані в базу
                db = next(get_db())
                new_item = Item(
                    name=name,
                    category=category,
                    color=color,
                    season=season,
                    image_path=file_path
                )
                db.add(new_item)
                db.commit()
                
                st.success(f"Річ '{name}' успішно додана!")
                st.rerun() # Оновлюємо сторінку, щоб річ одразу з'явилася в галереї