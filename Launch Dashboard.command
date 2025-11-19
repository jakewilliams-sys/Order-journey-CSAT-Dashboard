#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Please run: pip install -r requirements.txt"
    echo ""
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

# Run the dashboard
echo "Starting Streamlit Dashboard..."
echo "The dashboard will open in your browser automatically."
echo "To stop the dashboard, close this window or press Ctrl+C"
echo ""
streamlit run dashboard.py

