print("--- 1. Початок app.py ---")
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ai_agent import generate_stylist_reply, analyze_clothing_image_auto

from db import (
    add_clothing_item,
    authenticate_user,
    create_user,
    delete_clothing_item,
    get_clothing_item,
    get_user_clothes,
    init_db,
    update_clothing_item,
)
from utils import process_and_save_uploaded_image
print("--- 2. Імпорти завершено ---")

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
CHAT_MEMORY_MESSAGES = 10


def init_app_state() -> None:
    if "user" not in st.session_state:
        st.session_state.user = None


def render_auth_page() -> None:
    st.subheader("Login / Registration")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

        if submitted:
            user = authenticate_user(username, password)
            if user is None:
                st.error("Invalid username or password.")
            else:
                st.session_state.user = user
                st.success(f"Welcome back, {user['username']}!")
                st.rerun()

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            username = st.text_input("New username", key="register_username")
            password = st.text_input("New password", type="password", key="register_password")
            confirm_password = st.text_input(
                "Confirm password", type="password", key="register_password_confirm"
            )
            submitted = st.form_submit_button("Create account")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(username, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def render_add_item_page() -> None:
    user = st.session_state.user
    st.subheader("Add New Clothing Item")

    if "draft_item" not in st.session_state:
        st.session_state.draft_item = None

    if st.session_state.draft_item is None:
        item_image = st.file_uploader(
            "Upload image to auto-fill details via AI", type=["jpg", "jpeg", "png", "webp"], key="auto_new_item_image"
        )
        if item_image is not None:
            with st.spinner("AI is analyzing the photo..."):
                image_path = process_and_save_uploaded_image(item_image, UPLOADS_DIR)
                metadata = analyze_clothing_image_auto(image_path)
                
                st.session_state.draft_item = {
                    "image_path": image_path,
                    "name": metadata.get("name", ""),
                    "category": metadata.get("category", "Other"),
                    "season": metadata.get("season", "All Seasons"),
                    "ai_description": metadata.get("description", "")
                }
            st.rerun()
    else:
        draft = st.session_state.draft_item
        
        col_img, col_form = st.columns([0.4, 0.6])
        with col_img:
            st.image(draft["image_path"], width='stretch')
            if st.button("Cancel / Upload Another", width='stretch'):
                st.session_state.draft_item = None
                st.rerun()
                
        with col_form:
            with st.form("save_clothing_form"):
                item_name = st.text_input("Name", value=draft["name"])
                
                categories = ["Top", "Bottom", "Shoes", "Outerwear", "Dress", "Accessory", "Other"]
                cat_idx = categories.index(draft["category"]) if draft["category"] in categories else 6
                item_category = st.selectbox("Category", categories, index=cat_idx)
                
                seasons = ["All Seasons", "Spring", "Summer", "Autumn", "Winter"]
                sea_idx = seasons.index(draft["season"]) if draft["season"] in seasons else 0
                item_season = st.selectbox("Season", seasons, index=sea_idx)
                
                item_desc = st.text_area("AI Details (Properties)", value=draft["ai_description"], height=150)
                
                submitted = st.form_submit_button("Save Item", type="primary", width='stretch')
                    
            if submitted:
                if not item_name.strip():
                    st.error("Name is required.")
                else:
                    add_clothing_item(
                        user_id=user["id"], 
                        name=item_name, 
                        image_path=draft["image_path"], 
                        category=item_category,
                        season=item_season,
                        ai_description=item_desc
                    )
                    st.session_state.draft_item = None
                    st.success("Item added successfully!")
                    st.rerun()

def render_wardrobe_page() -> None:
    user = st.session_state.user
    st.subheader("My Wardrobe")

    clothes = get_user_clothes(user["id"])
    if not clothes:
        st.info("No items yet. Add your first item in the 'Add Item' tab.")
        return

    columns_per_row = 3
    for row_start in range(0, len(clothes), columns_per_row):
        row_items = clothes[row_start : row_start + columns_per_row]
        cols = st.columns(columns_per_row)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                image_path = Path(item["image_path"])
                if image_path.exists():
                    st.image(str(image_path), width='stretch')
                else:
                    st.warning("Image file is missing.")
                st.caption(f"{item['name']} ({item['category']})")
                st.write(f"**Season:** {item.get('season', 'All Seasons')}")
                if item.get("ai_description"):
                    st.write(f"**AI Details:** {item['ai_description']}")

                with st.expander("Edit / Delete", expanded=False):
                    with st.form(f"edit_item_{item['id']}", clear_on_submit=False):
                        new_name = st.text_input("Name", value=item["name"], key=f"edit_name_{item['id']}")
                        new_category = st.text_input(
                            "Category", value=item["category"], key=f"edit_category_{item['id']}"
                        )
                        
                        season_options = ["All Seasons", "Spring", "Summer", "Autumn", "Winter"]
                        curr_season = item.get("season", "All Seasons")
                        season_idx = season_options.index(curr_season) if curr_season in season_options else 0
                        new_season = st.selectbox(
                            "Season", 
                            season_options,
                            index=season_idx,
                            key=f"edit_season_{item['id']}"
                        )
                        new_ai_desc = st.text_area(
                            "AI Description", value=item.get("ai_description", ""), key=f"edit_ai_{item['id']}"
                        )
                        
                        replace_image = st.file_uploader(
                            "Replace image (optional)",
                            type=["jpg", "jpeg", "png", "webp"],
                            key=f"replace_image_{item['id']}",
                        )
                        update_clicked = st.form_submit_button("Update")

                    if update_clicked:
                        new_image_path = None
                        if replace_image is not None:
                            new_image_path = process_and_save_uploaded_image(replace_image, UPLOADS_DIR)
                        updated = update_clothing_item(
                            item_id=item["id"],
                            user_id=user["id"],
                            name=new_name,
                            category=new_category,
                            image_path=new_image_path,
                            season=new_season,
                            ai_description=new_ai_desc
                        )
                        if updated:
                            st.success("Item updated.")
                            st.rerun()
                        st.error("Could not update item.")

                    if st.button("Delete item", type="primary", key=f"delete_btn_{item['id']}"):
                        item_to_delete = get_clothing_item(item["id"], user["id"])
                        if item_to_delete:
                            deleted = delete_clothing_item(item["id"], user["id"])
                            if deleted:
                                image_file = Path(item_to_delete["image_path"])
                                if image_file.exists():
                                    image_file.unlink(missing_ok=True)
                                st.success("Item deleted.")
                                st.rerun()
                        st.error("Could not delete item.")


def render_ai_stylist_page() -> None:
    st.subheader("AI Stylist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask your stylist...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_reply = generate_stylist_reply(
                    user_id=st.session_state.user["id"],
                    user_message=user_prompt,
                    chat_history=st.session_state.chat_history[:-1],
                    memory_window=CHAT_MEMORY_MESSAGES,
                )
            st.markdown(assistant_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.chat_history = st.session_state.chat_history[-(CHAT_MEMORY_MESSAGES * 2) :]


def main() -> None:
    print("--- 3. Запуск main() ---")
    load_dotenv()
    st.set_page_config(page_title="Smart Wardrobe Assistant", page_icon="👔", layout="wide")
    print("--- 4. Ініціалізація БД ---")
    init_db()
    print("--- 5. Ініціалізація стану ---")
    init_app_state()

    user = st.session_state.user
    print(f"--- 6. Перевірка юзера: {user} ---")
    if user is None:
        st.title("Smart Wardrobe Assistant")
        print("--- 7. Рендеримо сторінку авторизації ---")
        render_auth_page()
        return

    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.title("Smart Wardrobe Assistant")
    with col2:
        st.markdown(f"<div style='text-align: right; color: #888; font-size: 0.9em; padding-top: 15px;'>👤 {user['username']}</div>", unsafe_allow_html=True)
        if st.button("Logout", width='stretch'):
            st.session_state.user = None
            st.rerun()

    print("--- 8. Рендеримо Вкладки ---")
    tab_stylist, tab_wardrobe, tab_add = st.tabs(["AI Stylist", "My Wardrobe", "Add Item"])

    with tab_stylist:
        print("--- 9. Рендеримо AI Stylist ---")
        render_ai_stylist_page()

    with tab_wardrobe:
        print("--- 9.5. Рендеримо Гардероб ---")
        render_wardrobe_page()

    with tab_add:
        print("--- 10. Рендеримо Додавання ---")
        render_add_item_page()


if __name__ == "__main__":
    main()
