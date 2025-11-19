"""
Incremental test version - adding features one by one to find the blocker.
Start with this if dashboard_test.py works.
"""
import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Incremental Test Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Incremental Test Dashboard")
st.info("Testing imports and basic functionality...")

# Test 1: Basic imports
try:
    import pandas as pd
    import numpy as np
    st.success("✅ Basic imports (pandas, numpy) work")
except Exception as e:
    st.error(f"❌ Basic imports failed: {e}")
    st.stop()

# Test 2: Additional imports
try:
    from scipy import stats
    import plotly.graph_objects as go
    import plotly.express as px
    st.success("✅ Additional imports (scipy, plotly) work")
except Exception as e:
    st.error(f"❌ Additional imports failed: {e}")
    st.stop()

# Test 3: Custom module imports
try:
    import data_processing as dp
    st.success("✅ data_processing module imported")
except Exception as e:
    st.error(f"❌ data_processing import failed: {e}")
    st.exception(e)
    st.stop()

try:
    import sentiment_analysis as sa
    st.success("✅ sentiment_analysis module imported")
except Exception as e:
    st.error(f"❌ sentiment_analysis import failed: {e}")
    st.exception(e)
    st.stop()

try:
    import visualizations as viz
    from visualizations import COLOR_PALETTE, get_group_color
    st.success("✅ visualizations module imported")
except Exception as e:
    st.error(f"❌ visualizations import failed: {e}")
    st.exception(e)
    st.stop()

# Test 4: Check for data file
import os
data_file = "OJ CSAT Trial.csv"
if os.path.exists(data_file):
    st.success(f"✅ Data file '{data_file}' found")
else:
    st.warning(f"⚠️ Data file '{data_file}' not found (this is OK for testing)")
    st.info(f"Current directory: {os.getcwd()}")
    try:
        files = os.listdir('.')
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            st.write("CSV files found:", ', '.join(csv_files))
    except Exception as e:
        st.write(f"Error listing files: {e}")

# Test 5: Try to define a simple function
@st.cache_data
def test_function():
    return "Test data"

try:
    result = test_function()
    st.success(f"✅ Cached function works: {result}")
except Exception as e:
    st.error(f"❌ Cached function failed: {e}")
    st.exception(e)

st.markdown("---")
st.success("🎉 All tests passed! If you see this, the basic structure works.")
st.info("Next: Try loading actual data and see where it fails.")

