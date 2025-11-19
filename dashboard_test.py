"""
Minimal test version to verify Streamlit can start on Streamlit Cloud.
"""
import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Test Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Streamlit Test Dashboard")
st.success("✅ Streamlit is working!")
st.info("If you can see this, Streamlit started successfully.")

st.markdown("---")
st.markdown("**Next steps:** If this works, we'll add back functionality incrementally.")

