# Complete Guide: Sharing Your Dashboard with Others

This guide will walk you through sharing your dashboard so anyone can access it with just a web link - no installation needed!

---

## What You'll Need

- A computer (Mac, Windows, or Linux)
- An internet connection
- About 15-20 minutes
- A free GitHub account (we'll create this together)

---

## Part 1: Setting Up GitHub

### Step 1.1: Create a GitHub Account (if you don't have one)

1. **Open your web browser** (Chrome, Safari, Firefox, etc.)

2. **Go to:** [https://github.com](https://github.com)

3. **Click the green button** that says "Sign up" (top right corner)

4. **Fill in the form:**
   - Enter your email address
   - Create a password
   - Choose a username (this will be part of your dashboard URL)
   - Solve the puzzle to prove you're human

5. **Click "Create account"**

6. **Verify your email** (check your inbox and click the verification link)

### Step 1.2: Create a New Repository on GitHub

A "repository" is just a fancy word for a folder where your code lives online.

1. **Make sure you're signed in** to GitHub (you should see your profile picture in the top right)

2. **Look for a "+" icon** in the top right corner (next to your profile picture)

3. **Click the "+" icon**, then click **"New repository"**

4. **Fill in the repository details:**
   - **Repository name:** Type `order-journey-csat-dashboard` (or any name you like - no spaces, use hyphens)
   - **Description:** (Optional) Type "CSAT Survey Analysis Dashboard"
   - **Choose Public or Private:**
     - **Public** = Anyone with the link can see your code (free for Streamlit Cloud)
     - **Private** = Only you (and people you invite) can see it
     - **Recommendation:** Choose **Public** (it's free and easier)

5. **IMPORTANT:** Do NOT check any of these boxes:
   - ❌ "Add a README file"
   - ❌ "Add .gitignore"
   - ❌ "Choose a license"
   
   (We already have these files, so leave them unchecked)

6. **Click the green button** that says **"Create repository"**

7. **You'll see a page with instructions** - DON'T follow those yet! We'll do that in the next section.

---

## Part 2: Uploading Your Code to GitHub

This part uses something called "Terminal" (Mac) or "Command Prompt" (Windows). Don't worry - we'll walk through it step by step!

### Step 2.1: Open Terminal (Mac) or Command Prompt (Windows)

**On Mac:**
1. Press `Command + Space` (hold both keys together)
2. Type "Terminal"
3. Press Enter
4. A black window will open - this is Terminal!

**On Windows:**
1. Press `Windows Key + R`
2. Type "cmd" and press Enter
3. A black window will open - this is Command Prompt!

### Step 2.2: Navigate to Your Project Folder

You need to tell Terminal/Command Prompt where your dashboard files are.

**On Mac:**
1. In Terminal, type this (copy and paste it):
   ```bash
   cd "/Users/jakewilliams/Desktop/Order Journey CSAT - Value"
   ```
2. Press Enter

**On Windows:**
1. In Command Prompt, type this (replace with your actual folder path):
   ```cmd
   cd "C:\Users\YourName\Desktop\Order Journey CSAT - Value"
   ```
2. Press Enter

**How to check if you're in the right place:**
- Type: `ls` (Mac) or `dir` (Windows) and press Enter
- You should see files like `dashboard.py`, `requirements.txt`, etc.
- If you see those files, you're in the right place! ✅

### Step 2.3: Set Up Git (One-Time Setup)

Git is the tool that uploads your code. We need to tell it who you are.

**Type these commands one at a time** (press Enter after each):

```bash
git config user.name "Your Name"
```

(Replace "Your Name" with your actual name, like "Jake")

```bash
git config user.email "your.email@example.com"
```

(Replace with the email you used for GitHub)

**Press Enter after each command.**

### Step 2.4: Initialize Git and Create Your First Commit

**Type these commands one at a time** (press Enter after each):

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "Initial commit - CSAT Dashboard"
```

```bash
git branch -M main
```

**What these commands do:**
- `git init` = Starts tracking your files
- `git add .` = Selects all your files to upload
- `git commit` = Saves a snapshot of your files
- `git branch -M main` = Names your branch "main"

### Step 2.5: Connect to Your GitHub Repository

**Type this command** (replace `YOUR_USERNAME` with your GitHub username):

```bash
git remote add origin https://github.com/YOUR_USERNAME/order-journey-csat-dashboard.git
```

**Example:** If your username is "jakewilliams-sys" and your repo is "Order-journey-CSAT-Dashboard", it would be:
```bash
git remote add origin https://github.com/jakewilliams-sys/Order-journey-CSAT-Dashboard.git
```

Press Enter.

### Step 2.6: Upload Your Code to GitHub

**Type this command:**

```bash
git push -u origin main
```

Press Enter.

**You'll be asked to sign in:**
- **Username:** Enter your GitHub username
- **Password:** Enter a **Personal Access Token** (NOT your GitHub password)

**How to get a Personal Access Token:**

1. Go to: [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name: "Streamlit Dashboard"
4. Check the box for **"repo"** (this gives it permission to upload code)
5. Scroll down and click **"Generate token"**
6. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
7. Paste it when Terminal asks for your password

**After entering your username and token:**
- You should see messages about "uploading" or "pushing"
- Wait for it to finish (may take 30 seconds to 2 minutes)
- When you see "done" or similar, you're finished! ✅

**Go back to your GitHub repository page** and refresh it - you should see all your files there!

---

## Part 3: Deploying to Streamlit Cloud

Now we'll make your dashboard accessible to anyone with a link!

### Step 3.1: Sign Up for Streamlit Cloud

1. **Go to:** [https://share.streamlit.io](https://share.streamlit.io)

2. **Click "Sign in"** (top right corner)

3. **Click "Continue with GitHub"** (this uses your GitHub account)

4. **Authorize Streamlit** - Click "Authorize streamlit" when asked

5. **You're now signed in!** You should see the Streamlit Cloud dashboard

### Step 3.2: Create a New App

1. **Click the big button** that says **"New app"** (usually in the top right or center of the page)

2. **Fill in the form:**

   **Repository:**
   - Click the dropdown menu
   - Find and select: `order-journey-csat-dashboard` (or whatever you named it)
   
   **Branch:**
   - Type: `main`
   - (This should be the default)
   
   **Main file path:**
   - Type: `dashboard.py`
   - (This is the main file that runs your dashboard)

3. **Click "Deploy!"** (big button at the bottom)

### Step 3.3: Wait for Deployment

- Streamlit Cloud will now:
  - Install all the required software
  - Set up your dashboard
  - Make it accessible online

- **This takes 2-5 minutes** - you'll see progress messages
- A spinning icon means it's working - be patient!
- When it's done, you'll see "Your app is live!" ✅

### Step 3.4: Get Your Dashboard URL

Once deployment is complete, you'll see:
- A green checkmark ✅
- A URL that looks like: `https://order-journey-csat-dashboard.streamlit.app`

**This is your dashboard link!** Copy it - you'll share this with others.

---

## Part 4: Sharing Your Dashboard

### How to Share

1. **Copy your dashboard URL** (from Streamlit Cloud)

2. **Share it with others:**
   - Email it
   - Send it in a message
   - Put it in a document
   - Share it however you like!

3. **Anyone with the link can:**
   - Open it in any web browser
   - View and interact with your dashboard
   - No installation needed!

### Making Updates

If you make changes to your dashboard:

1. **Make your changes** to the files on your computer

2. **Open Terminal/Command Prompt** and navigate to your folder (Step 2.2)

3. **Run these commands:**
   ```bash
   git add .
   git commit -m "Updated dashboard"
   git push
   ```

4. **Streamlit Cloud will automatically update** your dashboard (usually within 1-2 minutes)

5. **Refresh the dashboard URL** to see your changes!

---

## Troubleshooting

### Problem: "Git is not installed"

**Solution:**
- **Mac:** Git usually comes pre-installed. If not, install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- **Windows:** Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)

### Problem: "Permission denied" when pushing to GitHub

**Solution:**
- Make sure you're using a Personal Access Token (not your password)
- Make sure the token has "repo" permissions checked
- Try generating a new token

### Problem: "Repository not found" in Streamlit Cloud

**Solution:**
- Make sure your GitHub repository is set to **Public** (or you've given Streamlit Cloud access to private repos)
- Check that you spelled the repository name correctly
- Make sure you've pushed your code to GitHub first

### Problem: "Module not found" error in Streamlit Cloud

**Solution:**
- Check that `requirements.txt` includes all necessary packages
- Make sure all your Python files are in the repository
- Check the Streamlit Cloud logs for specific error messages

### Problem: Dashboard shows "File not found" error

**Solution:**
- Make sure `OJ CSAT Trial.csv` is in your repository
- Check that the file name in `dashboard.py` matches exactly (including capitalization)
- Make sure the CSV file was included when you ran `git add .`

### Problem: Terminal/Command Prompt says "command not found"

**Solution:**
- Make sure you're typing commands exactly as shown
- Check that you're in the correct folder (use `ls` or `dir` to see files)
- Make sure Git is installed (see first troubleshooting item)

### Problem: Can't find the "+" button on GitHub

**Solution:**
- Make sure you're signed in to GitHub
- The "+" icon is in the top right corner, next to your profile picture
- Try refreshing the page

---

## Quick Reference: Commands You'll Use

Here are all the commands in one place (copy and paste as needed):

```bash
# Navigate to your folder (Mac - adjust path as needed)
cd "/Users/jakewilliams/Desktop/Order Journey CSAT - Value"

# Set up Git (one-time, replace with your info)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Initial upload (do this once)
git init
git add .
git commit -m "Initial commit - CSAT Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# Making updates (use these when you make changes)
git add .
git commit -m "Your update message"
git push
```

---

## Need Help?

- **Streamlit Cloud Docs:** [https://docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **GitHub Help:** [https://docs.github.com](https://docs.github.com)
- **Streamlit Community:** [https://discuss.streamlit.io](https://discuss.streamlit.io)

---

## Summary

**What you've accomplished:**
1. ✅ Created a GitHub account and repository
2. ✅ Uploaded your dashboard code to GitHub
3. ✅ Deployed your dashboard to Streamlit Cloud
4. ✅ Got a shareable URL that anyone can access

**Your dashboard is now live and shareable!** 🎉

Anyone with the link can access it from any device, anywhere in the world - no installation needed!

