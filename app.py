from pathlib import Path
import asyncio

import streamlit as st

from langgraph_stylist import analyze_clothing_image_auto, run_stylist_workflow

from db import (
    add_clothing_item,
    authenticate_user,
    create_user,
    delete_clothing_item,
    get_clothing_item,
    get_user_profile,
    get_user_clothes,
    upsert_user_profile,
    update_clothing_item,
    UserProfileData,
)
from utils import process_and_save_uploaded_image


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
CHAT_MEMORY_MESSAGES = 10


def run_stylist_workflow_sync(
    user_id: int,
    user_query: str,
    chat_history: list[dict[str, str]] | None = None,
):
    return asyncio.run(
        run_stylist_workflow(
            user_id=user_id,
            user_query=user_query,
            chat_history=chat_history or [],
        )
    )


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
                "Confirm password",
                type="password",
                key="register_password_confirm",
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

    categories = ["Top", "Bottom", "Shoes", "Outerwear", "Dress", "Accessory", "Other"]
    seasons = ["All Seasons", "Spring", "Summer", "Autumn", "Winter"]

    if "draft_item" not in st.session_state:
        st.session_state.draft_item = None

    with st.expander("Add new clothing item", expanded=True):
        if st.session_state.draft_item is None:
            item_image = st.file_uploader(
                "Upload image to auto-fill details via AI",
                type=["jpg", "jpeg", "png", "webp"],
                key="auto_new_item_image",
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
                        "ai_description": metadata.get("description", ""),
                    }

                st.rerun()

        else:
            draft = st.session_state.draft_item
            col_img, col_form = st.columns([0.4, 0.6])

            with col_img:
                st.image(draft["image_path"], use_container_width=True)

                if st.button("Cancel / Upload another", use_container_width=True):
                    st.session_state.draft_item = None
                    st.rerun()

            with col_form:
                with st.form("save_clothing_form", clear_on_submit=False):
                    item_name = st.text_input("Name", value=draft.get("name", ""))

                    category_value = draft.get("category", "Other")
                    category_index = categories.index(category_value) if category_value in categories else 6
                    item_category = st.selectbox("Category", categories, index=category_index)

                    season_value = draft.get("season", "All Seasons")
                    season_index = seasons.index(season_value) if season_value in seasons else 0
                    item_season = st.selectbox("Season", seasons, index=season_index)

                    item_ai_description = st.text_area(
                        "AI Details",
                        value=draft.get("ai_description", ""),
                        height=150,
                    )

                    submitted = st.form_submit_button("Save item", type="primary")

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
                            ai_description=item_ai_description,
                        )

                        st.session_state.draft_item = None
                        st.success("Item added successfully.")
                        st.rerun()

    clothes = get_user_clothes(user["id"])
    if not clothes:
        st.info("No items yet. Add your first item above.")
        return

    columns_per_row = 3
    for row_start in range(0, len(clothes), columns_per_row):
        row_items = clothes[row_start: row_start + columns_per_row]
        cols = st.columns(columns_per_row)

        for idx, item in enumerate(row_items):
            with cols[idx]:
                image_path = Path(item["image_path"])

                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning("Image file is missing.")

                st.caption(f"{item['name']} ({item['category']})")
                st.write(f"**Season:** {item.get('season', 'All Seasons')}")

                if item.get("ai_description"):
                    st.markdown("**AI Details:**")
                    st.text(item["ai_description"])

                with st.expander("Edit / Delete", expanded=False):
                    with st.form(f"edit_item_{item['id']}", clear_on_submit=False):
                        new_name = st.text_input(
                            "Name",
                            value=item["name"],
                            key=f"edit_name_{item['id']}",
                        )

                        category_value = item.get("category", "Other")
                        category_index = categories.index(category_value) if category_value in categories else 6
                        new_category = st.selectbox(
                            "Category",
                            categories,
                            index=category_index,
                            key=f"edit_category_{item['id']}",
                        )

                        season_value = item.get("season", "All Seasons")
                        season_index = seasons.index(season_value) if season_value in seasons else 0
                        new_season = st.selectbox(
                            "Season",
                            seasons,
                            index=season_index,
                            key=f"edit_season_{item['id']}",
                        )

                        new_ai_description = st.text_area(
                            "AI Details",
                            value=item.get("ai_description", ""),
                            key=f"edit_ai_description_{item['id']}",
                            height=130,
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
                            new_image_path = process_and_save_uploaded_image(
                                replace_image,
                                UPLOADS_DIR,
                            )

                        updated = update_clothing_item(
                            item_id=item["id"],
                            user_id=user["id"],
                            name=new_name,
                            category=new_category,
                            image_path=new_image_path,
                            season=new_season,
                            ai_description=new_ai_description,
                        )

                        if updated:
                            st.success("Item updated.")
                            st.rerun()
                        else:
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


def render_workflow_details(workflow_state) -> None:
    with st.expander("LangGraph workflow details", expanded=False):
        st.markdown("**Selector reasoning**")
        st.write(workflow_state.selection_reasoning or "No selector reasoning.")

        st.markdown("**Weather info**")
        st.write(workflow_state.weather_summary or "No weather info.")

        st.markdown("**Weather reasoning**")
        st.write(workflow_state.weather_reasoning or "No weather reasoning.")

        st.markdown("**Selected items**")
        if workflow_state.selected_items:
            selected_items = [
                {
                    "id": item.id,
                    "reasoning": item.reasoning,
                }
                for item in workflow_state.selected_items
            ]
            st.table(selected_items)
        else:
            st.write("No selected items.")

        st.markdown("**Vision notes**")
        st.write(workflow_state.vision_notes or "No vision notes.")

        st.markdown("**Validator feedback**")
        st.write(workflow_state.validator_feedback or "No validator feedback.")

        if workflow_state.missing_images:
            st.warning(workflow_state.missing_images)

        if workflow_state.errors:
            st.error(workflow_state.errors)


def render_ai_stylist_page() -> None:
    st.subheader("AI Stylist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask your stylist...")

    if user_prompt:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with LangGraph agents..."):
                workflow_state = run_stylist_workflow_sync(
                    user_id=st.session_state.user["id"],
                    user_query=user_prompt,
                    chat_history=st.session_state.chat_history[:-1],
                )

            assistant_reply = workflow_state.final_advice or (
                "I could not generate outfit advice right now. Please try again."
            )

            st.markdown(assistant_reply)
            render_workflow_details(workflow_state)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": assistant_reply,
            }
        )

        st.session_state.chat_history = st.session_state.chat_history[-(CHAT_MEMORY_MESSAGES * 2):]


def render_profile_page() -> None:
    st.subheader("Profile")
    user = st.session_state.user
    profile = get_user_profile(user["id"])

    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        st.markdown("### Як визначити свій розмір")
        st.markdown(
            """
            **A. Обхват грудей**  
            Використовуючи вимірювальну стрічку, здійсніть замір найширшої частини ваших грудей.

            **B. Обхват талії**  
            Виконайте заміри найвужчої частини вашої талії. Стрічка не має бути занадто натягнутою.

            **C. Обхват стегон**  
            Заміряйте найповнішу частину ваших стегон, тримаючи стрічку горизонтально.
            """
        )

        img_col1, img_col2 = st.columns(2)

        try:
            img_col1.image(
                str(BASE_DIR / "pics" / "male.jpg"),
                caption="Чоловік",
                use_container_width=True,
            )
            img_col2.image(
                str(BASE_DIR / "pics" / "female.jpg"),
                caption="Жінка",
                use_container_width=True,
            )
        except Exception:
            st.error("Зображення не знайдені в папці 'pics'.")

    with right_col:
        default_styles = [
            style.strip()
            for style in profile.style_preferences.split(",")
            if style.strip()
        ]

        gender_options = ["Не вказувати", "Жінка", "Чоловік", "Небінарна особа"]

        legacy_gender_map = {
            "": "Не вказувати",
            "Female": "Жінка",
            "Male": "Чоловік",
            "Non-binary": "Небінарна особа",
            "Prefer not to say": "Не вказувати",
        }

        current_gender = legacy_gender_map.get(profile.gender, profile.gender)
        gender_index = (
            gender_options.index(current_gender)
            if current_gender in gender_options
            else 0
        )

        with st.form("profile_form", clear_on_submit=False):
            st.markdown("### Ваші дані")

            full_name = st.text_input("Ім'я", value=profile.full_name)

            gender = st.selectbox(
                "Стать",
                options=gender_options,
                index=gender_index,
            )

            selected_styles = st.multiselect(
                "Бажані стилі",
                ["Business", "Casual", "Old Money", "Streetwear", "Minimalist", "Sport", "Elegant"],
                default=default_styles,
            )

            st.markdown("##### Основні мірки (см)")

            chest_cm = st.number_input(
                "Обхват грудей (A)",
                min_value=0.0,
                value=float(profile.chest_cm),
                step=0.5,
            )

            waist_cm = st.number_input(
                "Обхват талії (B)",
                min_value=0.0,
                value=float(profile.waist_cm),
                step=0.5,
            )

            hips_cm = st.number_input(
                "Обхват стегон (C)",
                min_value=0.0,
                value=float(profile.hips_cm),
                step=0.5,
            )

            submitted = st.form_submit_button("Зберегти зміни")

        if submitted:
            is_valid = True

            if not (50 <= chest_cm <= 200):
                st.error("Введені дані не валідні.")
                is_valid = False
            elif not (40 <= waist_cm <= 180):
                st.error("Введені дані не валідні.")
                is_valid = False
            elif not (60 <= hips_cm <= 200):
                st.error("Введені дані не валідні.")
                is_valid = False

            if is_valid:
                payload = UserProfileData(
                    user_id=user["id"],
                    full_name=full_name.strip(),
                    gender=gender,
                    style_preferences=", ".join(selected_styles),
                    chest_cm=chest_cm,
                    waist_cm=waist_cm,
                    hips_cm=hips_cm,
                    shoulder_cm=profile.shoulder_cm,
                    sleeve_cm=profile.sleeve_cm,
                    inseam_cm=profile.inseam_cm,
                    foot_length_cm=profile.foot_length_cm,
                )

                upsert_user_profile(payload)
                st.success("Дані профілю успішно оновлені!")