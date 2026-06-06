import os
import subprocess
import sys
from pathlib import Path


STREAMLIT_ENV_FLAG = "WARDROBIE_RUNNING_IN_STREAMLIT"


def launch_with_streamlit() -> None:
    current_file = Path(__file__).resolve()

    env = os.environ.copy()
    env[STREAMLIT_ENV_FLAG] = "1"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(current_file),
        ],
        env=env,
        check=False,
    )


def run_app() -> None:
    from dotenv import load_dotenv
    import streamlit as st

    from app import (
        init_app_state,
        render_ai_stylist_page,
        render_auth_page,
        render_profile_page,
        render_wardrobe_page,
    )
    from db import init_db

    load_dotenv()

    st.set_page_config(
        page_title="Smart Wardrobe Assistant",
        layout="wide",
    )

    init_db()
    init_app_state()

    st.title("Wardrobie - Your Smart Wardrobe Assistant")

    user = st.session_state.user

    if user is None:
        render_auth_page()
        return

    sidebar = st.sidebar
    sidebar.success(f"Logged in as `{user['username']}`")

    from langfuse_tracing import is_langfuse_enabled, verify_langfuse_connection

    if is_langfuse_enabled() and not verify_langfuse_connection():
        sidebar.warning("Langfuse tracing is configured but authentication failed.")

    if sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    page = sidebar.radio(
        "Navigation",
        ["Profile", "My Wardrobe", "AI Stylist"],
    )

    if page == "My Wardrobe":
        render_wardrobe_page()
    elif page == "AI Stylist":
        render_ai_stylist_page()
    else:
        render_profile_page()


def main() -> None:
    if os.environ.get(STREAMLIT_ENV_FLAG) == "1":
        run_app()
    else:
        launch_with_streamlit()


main()