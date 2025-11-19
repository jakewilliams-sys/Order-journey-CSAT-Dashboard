# Deployment Guide for Streamlit Cloud

## Project Status
✅ Git repository: Not initialized (will be initialized in next steps)
✅ .gitignore: Created
✅ All necessary files: Present
✅ Data file: `OJ CSAT Trial.csv` (will be included in repository)

## Step 1: Initialize Git and Create First Commit

Run these commands in your terminal (from the project directory):

```bash
# Initialize Git repository
git init

# Add all files (CSV will be included since it's needed for the dashboard)
git add .

# Create initial commit
git commit -m "Initial commit - CSAT Survey Analysis Dashboard"
```

## Step 2: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository name: `order-journey-csat-dashboard` (or your preferred name)
4. Choose **Public** or **Private** (Public is free for Streamlit Cloud)
5. **DO NOT** initialize with README, .gitignore, or license (you already have these)
6. Click **"Create repository"**

## Step 3: Push Code to GitHub

After creating the repository, GitHub will show you commands. Use these (replace `YOUR_USERNAME` and `YOUR_REPO_NAME`):

```bash
# Set the main branch name
git branch -M main

# Add GitHub as remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** and authorize with your GitHub account
3. Click **"New app"**
4. Fill in the form:
   - **Repository**: Select your repository (`order-journey-csat-dashboard`)
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
5. Click **"Deploy!"**

## Step 5: Wait for Deployment

- Streamlit Cloud will install dependencies from `requirements.txt`
- It will run `dashboard.py`
- The first deployment may take 2-3 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`

## Important Notes

- **Data File**: The `OJ CSAT Trial.csv` file is included in the repository. If this contains sensitive data, consider:
  - Making the repository private
  - Using Streamlit's secrets management for sensitive data
  - Or modifying the dashboard to use a file uploader instead

- **Updates**: To update the deployed app:
  ```bash
  git add .
  git commit -m "Your update message"
  git push
  ```
  Streamlit Cloud will automatically redeploy when you push changes.

## Troubleshooting

If deployment fails:
- Check that `requirements.txt` includes all dependencies
- Verify `dashboard.py` is the correct main file
- Check the logs in Streamlit Cloud dashboard
- Ensure the CSV file path in `dashboard.py` matches the file name exactly

