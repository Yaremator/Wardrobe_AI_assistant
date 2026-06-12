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

# Hide Deploy button and Streamlit toolbar
st.markdown("""
<style>
    [data-testid="stToolbar"] {
        display: none;
    }
    
    .stDeployButton {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

init_db()
init_app_state()

st.title("Wardrobie - Your Smart Wardrobe Assistant")

user = st.session_state.user

if user is None:
    render_auth_page()
    st.stop()

# User info and logout in sidebar
sidebar = st.sidebar
sidebar.success(f"Logged in as `{user['username']}`")

from langfuse_tracing import is_langfuse_enabled, verify_langfuse_connection

if is_langfuse_enabled() and not verify_langfuse_connection():
    sidebar.warning("Langfuse tracing is configured but authentication failed.")

if sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# Navigation tabs at the top
tab_profile, tab_wardrobe, tab_stylist = st.tabs(["Profile", "My Wardrobe", "AI Stylist"])

with tab_profile:
    render_profile_page()

with tab_wardrobe:
    render_wardrobe_page()

with tab_stylist:
    render_ai_stylist_page()
