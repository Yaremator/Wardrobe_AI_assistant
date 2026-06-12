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
    st.stop()

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
