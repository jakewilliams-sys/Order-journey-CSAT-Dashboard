# Testing Instructions for Streamlit Cloud Debug

## Step 1: Test Minimal Version

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app" (or edit your existing app)
3. Change the **Main file path** from `dashboard.py` to `dashboard_test.py`
4. Click "Deploy!" or "Save"
5. Wait for deployment

**Expected Result:**
- If you see "✅ Streamlit is working!" - Streamlit can start, the issue is in dashboard.py
- If you still get "connection refused" - there's an environment issue

## Step 2: If Minimal Test Works

We'll add back features incrementally to find what's blocking startup.

## Step 3: Switch Back to Main Dashboard

Once we identify and fix the issue, change the Main file path back to `dashboard.py`

