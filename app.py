print("--- 1. Початок app.py ---")
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ai_agent import generate_stylist_reply

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


def render_wardrobe_page() -> None:
    user = st.session_state.user
    st.subheader("My Wardrobe")

    with st.expander("Add new clothing item", expanded=True):
        with st.form("add_clothing_form", clear_on_submit=True):
            item_name = st.text_input("Name")
            item_category = st.selectbox(
                "Category",
                ["Top", "Bottom", "Shoes", "Outerwear", "Dress", "Accessory", "Other"],
            )
            item_image = st.file_uploader(
                "Upload image", type=["jpg", "jpeg", "png", "webp"], key="new_item_image"
            )
            submitted = st.form_submit_button("Save item")

        if submitted:
            if not item_name.strip():
                st.error("Name is required.")
            elif item_image is None:
                st.error("Image is required.")
            else:
                image_path = process_and_save_uploaded_image(item_image, UPLOADS_DIR)
                add_clothing_item(user_id=user["id"], name=item_name, image_path=image_path, category=item_category)
                st.success("Item added successfully.")
                st.rerun()

    clothes = get_user_clothes(user["id"])
    if not clothes:
        st.info("No items yet. Add your first item above.")
        return

    columns_per_row = 3
    for row_start in range(0, len(clothes), columns_per_row):
        row_items = clothes[row_start : row_start + columns_per_row]
        cols = st.columns(columns_per_row)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                image_path = Path(item["image_path"])
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning("Image file is missing.")
                st.caption(f"{item['name']} ({item['category']})")

                with st.expander("Edit / Delete", expanded=False):
                    with st.form(f"edit_item_{item['id']}", clear_on_submit=False):
                        new_name = st.text_input("Name", value=item["name"], key=f"edit_name_{item['id']}")
                        new_category = st.text_input(
                            "Category", value=item["category"], key=f"edit_category_{item['id']}"
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

    st.title("Smart Wardrobe Assistant")

    user = st.session_state.user
    print(f"--- 6. Перевірка юзера: {user} ---")
    if user is None:
        print("--- 7. Рендеримо сторінку авторизації ---")
        render_auth_page()
        return

    sidebar = st.sidebar
    sidebar.success(f"Logged in as `{user['username']}`")
    if sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    page = sidebar.radio("Navigation", ["My Wardrobe", "AI Stylist"])
    print(f"--- 8. Поточна сторінка: {page} ---")

    if page == "My Wardrobe":
        print("--- 9. Рендеримо Гардероб ---")
        render_wardrobe_page()
    else:
        print("--- 10. Рендеримо AI Stylist ---")
        render_ai_stylist_page()


if __name__ == "__main__":
    main()
