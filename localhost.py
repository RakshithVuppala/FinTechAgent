#!/usr/bin/env python3
"""
Localhost Configuration and Runner for FinTech Agent
==================================================

Simplified script specifically for running the application locally
with optimized settings for development and personal use.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_environment():
    """Check and setup local environment."""
    print("🔍 Checking local environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    
    # Check if we're in the right directory
    if not Path("src/streamlit_dashboard.py").exists():
        print("❌ Please run this script from the FinTechAgent root directory")
        return False
    
    print("✅ Directory structure - OK")
    return True

def setup_local_environment():
    """Setup local environment variables and directories."""
    print("🔧 Setting up local environment...")
    
    # Create data directories
    local_dirs = [
        "data/raw", "data/interim", "data/processed", 
        "data/structured", "data/vector_db", "logs"
    ]
    
    for directory in local_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Setup .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        if Path(".env.example").exists():
            env_file.write_text(Path(".env.example").read_text())
            print("✅ Created .env file from template")
        else:
            # Create basic .env file
            env_content = """# Local Development Configuration
# Add your API keys here for enhanced features (optional)

# GitHub AI API (Recommended for AI features)
# GITHUB_AI_API_KEY=your_github_ai_api_key_here

# OpenAI API (Alternative)
# OPENAI_API_KEY=your_openai_api_key_here

# Reddit API (Optional for sentiment analysis)
# REDDIT_CLIENT_ID=your_reddit_client_id
# REDDIT_CLIENT_SECRET=your_reddit_client_secret
# REDDIT_USER_AGENT=FinTechAgent/1.0

# Alpha Vantage API (Optional for additional data)
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Local development settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
CACHE_DURATION_MINUTES=30
"""
            env_file.write_text(env_content)
            print("✅ Created basic .env file")
    
    print("✅ Local environment setup complete")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Checking dependencies...")
    
    try:
        import streamlit
        print("✅ Dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run manually: pip install -r requirements.txt")
            return False

def start_localhost_server():
    """Start the Streamlit server optimized for localhost."""
    print("🚀 Starting FinTech Agent on localhost...")
    print()
    print("🌐 Application will be available at: http://localhost:8501")
    print("💡 Running in LOCAL MODE with the following features:")
    print("   • Local data storage (no cloud dependencies)")
    print("   • Optional API keys for enhanced features")
    print("   • Optimized for single-user development")
    print("   • Automatic browser opening")
    print()
    print("📋 Quick Usage Guide:")
    print("1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)")
    print("2. Choose 'Orchestrator Agent' for best performance")  
    print("3. Click 'Start AI Research'")
    print("4. Review the comprehensive analysis")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Change to src directory
    original_dir = os.getcwd()
    os.chdir("src")
    
    try:
        # Start Streamlit with localhost-optimized settings
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
            "--server.port=8501",
            "--server.address=localhost", 
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--global.developmentMode=true"
        ])
        
        # Open browser after a short delay
        import time
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Opened application in your default browser")
        except:
            print("💡 Please open http://localhost:8501 in your browser")
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down FinTech Agent...")
        process.terminate()
        print("✅ Server stopped successfully")
    except Exception as e:
        print(f"❌ Error running application: {e}")
    finally:
        os.chdir(original_dir)

def main():
    """Main function for localhost runner."""
    print("💻 FinTech Agent - Localhost Runner")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Setup local environment
    setup_local_environment()
    
    # Install dependencies if needed
    if not install_dependencies():
        sys.exit(1)
    
    # Start the server
    start_localhost_server()

if __name__ == "__main__":
    main()