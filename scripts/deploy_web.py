#!/usr/bin/env python3
"""
Web Deployment Assistant for FinTech Agent
==========================================

Interactive script to help deploy the application to various cloud platforms.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.END}")

def print_header(text):
    """Print a header with decorations"""
    print("\n" + "="*60)
    print_colored(f"🚀 {text}", Colors.BOLD + Colors.BLUE)
    print("="*60)

def run_command(command, description="Running command"):
    """Run a shell command and return success status"""
    print_colored(f"⏳ {description}...", Colors.YELLOW)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print_colored(f"✅ {description} completed successfully", Colors.GREEN)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ {description} failed: {e}", Colors.RED)
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, e.stderr

def check_git_status():
    """Check if git is initialized and clean"""
    print_colored("🔍 Checking git status...", Colors.BLUE)
    
    # Check if git is initialized
    if not Path(".git").exists():
        print_colored("❌ Git not initialized. Initializing...", Colors.YELLOW)
        success, _ = run_command("git init", "Initializing git repository")
        if not success:
            return False
    
    # Check for uncommitted changes
    success, output = run_command("git status --porcelain", "Checking for uncommitted changes")
    if success and output.strip():
        print_colored("⚠️ You have uncommitted changes. Committing...", Colors.YELLOW)
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Prepare for deployment"', "Committing changes")
    
    return True

def deploy_streamlit_cloud():
    """Guide user through Streamlit Cloud deployment"""
    print_header("Streamlit Community Cloud Deployment")
    
    print("""
🌟 Streamlit Community Cloud is the easiest way to deploy!

Prerequisites:
✅ GitHub repository (public)
✅ Streamlit Cloud account

Steps:
""")
    
    print_colored("1. Push your code to GitHub:", Colors.BLUE)
    if check_git_status():
        # Check if origin exists
        success, _ = run_command("git remote get-url origin", "Checking git origin")
        if not success:
            print_colored("⚠️ No git origin found. Please add your GitHub repository:", Colors.YELLOW)
            print("   git remote add origin https://github.com/yourusername/FinTechAgent.git")
            return False
        
        run_command("git push origin main", "Pushing to GitHub")
    
    print_colored("\n2. Deploy to Streamlit Cloud:", Colors.BLUE)
    print("   • Go to https://share.streamlit.io")
    print("   • Sign in with GitHub")
    print("   • Click 'New app'")
    print("   • Select your repository")
    print("   • Set main file: src/streamlit_dashboard.py")
    print("   • Click 'Deploy'")
    
    print_colored("\n3. Set environment variables (optional):", Colors.BLUE)
    print("   In Streamlit Cloud settings, add:")
    print("   • GITHUB_AI_API_KEY (for enhanced AI features)")
    print("   • Other API keys as needed")
    
    print_colored("\n🎉 Your app will be live at: https://yourapp.streamlit.app", Colors.GREEN)

def deploy_heroku():
    """Guide user through Heroku deployment"""
    print_header("Heroku Deployment")
    
    # Check if Heroku CLI is installed
    success, _ = run_command("heroku --version", "Checking Heroku CLI")
    if not success:
        print_colored("❌ Heroku CLI not found. Please install it first:", Colors.RED)
        print("   https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    print_colored("🔧 Setting up Heroku deployment...", Colors.BLUE)
    
    # Login to Heroku
    print_colored("Logging in to Heroku (browser will open):", Colors.YELLOW)
    run_command("heroku login", "Heroku login")
    
    # Create Heroku app
    app_name = input("Enter your app name (or press Enter for auto-generated): ").strip()
    if app_name:
        command = f"heroku create {app_name}"
    else:
        command = "heroku create"
    
    success, output = run_command(command, "Creating Heroku app")
    if success:
        # Extract app URL from output
        lines = output.split('\n')
        app_url = next((line for line in lines if 'https://' in line and 'herokuapp.com' in line), "")
        
        # Set environment variables
        print_colored("\n🔑 Setting environment variables...", Colors.BLUE)
        env_file = Path(".env")
        if env_file.exists():
            print("Found .env file. Setting key environment variables...")
            # Set key environment variables (avoid setting sensitive keys in public logs)
            run_command("heroku config:set PYTHONPATH=/app", "Setting PYTHONPATH")
            run_command("heroku config:set STREAMLIT_SERVER_PORT=80", "Setting Streamlit port")
            
            print_colored("⚠️ Remember to set your API keys manually:", Colors.YELLOW)
            print("   heroku config:set GITHUB_AI_API_KEY=your_key")
            print("   heroku config:set REDDIT_CLIENT_ID=your_id")
            print("   heroku config:set REDDIT_CLIENT_SECRET=your_secret")
        
        # Deploy
        if check_git_status():
            success, _ = run_command("git push heroku main", "Deploying to Heroku")
            if success:
                print_colored(f"\n🎉 Deployment successful!", Colors.GREEN)
                print_colored(f"🌐 Your app is live at: {app_url}", Colors.BLUE)
                
                # Open the app
                run_command("heroku open", "Opening app in browser")

def deploy_railway():
    """Guide user through Railway deployment"""
    print_header("Railway Deployment")
    
    # Check if Railway CLI is installed
    success, _ = run_command("railway version", "Checking Railway CLI")
    if not success:
        print_colored("❌ Railway CLI not found. Installing...", Colors.YELLOW)
        if platform.system() == "Windows":
            print("Please install Railway CLI manually:")
            print("   npm install -g @railway/cli")
            return False
        else:
            success, _ = run_command("npm install -g @railway/cli", "Installing Railway CLI")
            if not success:
                print_colored("❌ Failed to install Railway CLI", Colors.RED)
                return False
    
    print_colored("🔧 Setting up Railway deployment...", Colors.BLUE)
    
    # Login and deploy
    run_command("railway login", "Logging in to Railway")
    run_command("railway init", "Initializing Railway project")
    
    if check_git_status():
        success, _ = run_command("railway up", "Deploying to Railway")
        if success:
            print_colored("\n🎉 Deployment successful!", Colors.GREEN)
            print_colored("🌐 Check your Railway dashboard for the live URL", Colors.BLUE)

def deploy_docker():
    """Guide user through Docker deployment options"""
    print_header("Docker Deployment Options")
    
    print("""
🐳 Docker deployment options:

1. Google Cloud Run (Recommended for production)
2. AWS ECS
3. Azure Container Instances
4. DigitalOcean App Platform
5. Self-hosted VPS

Choose your platform and follow the respective guides in DEPLOYMENT.md
""")
    
    # Build Docker image for testing
    print_colored("🔨 Building Docker image for testing...", Colors.BLUE)
    success, _ = run_command("docker build -t fintech-agent .", "Building Docker image")
    if success:
        print_colored("✅ Docker image built successfully!", Colors.GREEN)
        print("Test locally with: docker run -p 8501:8501 fintech-agent")

def main():
    """Main deployment assistant"""
    print_header("FinTech Agent Web Deployment Assistant")
    
    print("""
Welcome! This assistant will help you deploy your FinTech Agent to the web.

Choose your deployment platform:

1. 🌟 Streamlit Community Cloud (Free, Easy)
2. 🚀 Heroku (Popular, Reliable) 
3. 🚄 Railway (Modern, Fast)
4. 🐳 Docker/Cloud (Production, Scalable)
5. ❓ Help me choose

""")
    
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            deploy_streamlit_cloud()
            break
        elif choice == "2":
            deploy_heroku()
            break
        elif choice == "3":
            deploy_railway()
            break
        elif choice == "4":
            deploy_docker()
            break
        elif choice == "5":
            print("""
🤔 Help choosing a platform:

💰 **Free/Low Cost**: Streamlit Cloud (free), Railway (good free tier)
🏢 **Production/Business**: Heroku, Google Cloud Run, AWS
⚡ **Performance**: Railway, Google Cloud Run
🎓 **Learning/Demo**: Streamlit Cloud
🛠️ **Full Control**: Docker on VPS/Cloud

Recommendation: Start with Streamlit Cloud for demos, upgrade to Railway/Heroku for production.
""")
        else:
            print_colored("❌ Invalid choice. Please enter 1-5.", Colors.RED)

if __name__ == "__main__":
    main()