#!/usr/bin/env python3
"""
Quick start script for AI-Powered Financial Research Agent
=========================================================

This script provides a simple way to run the application with automatic setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        print("Please upgrade Python and try again.")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies if not already installed."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    """Set up environment file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔧 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created")
        print("💡 You can edit .env file to add your API keys for enhanced features")

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw", "data/interim", "data/processed", 
        "data/structured", "data/vector_db", "models", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def run_application():
    """Run the Streamlit application."""
    print("🚀 Starting AI-Powered Financial Research Agent...")
    print()
    print("📋 Quick Start Guide:")
    print("1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)")
    print("2. Choose 'Orchestrator Agent' for best performance")
    print("3. Click 'Start AI Research'")
    print("4. Review the comprehensive analysis results")
    print()
    print("🌐 The application will open in your browser at:")
    print("   http://localhost:8501")
    print()
    print("💡 Running in LOCAL MODE - Perfect for development and testing!")
    print("   - All data stored locally in 'data/' directory")
    print("   - API keys loaded from .env file (if present)")
    print("   - Application optimized for localhost performance")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    # Change to src directory and run streamlit
    os.chdir("src")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function."""
    print("🚀 AI-Powered Financial Research Agent")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    create_directories()
    
    # Check and install dependencies
    if not check_dependencies():
        print("📦 Dependencies not found. Installing...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please run:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()