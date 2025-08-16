#!/usr/bin/env python3
"""
Setup script for AI-Powered Financial Research Agent
===================================================

This script helps set up the development environment and verify the installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip...")
    try:
        import pip
        print(f"✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available")
        return False

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    print("🏗️ Creating virtual environment...")
    
    # Create virtual environment
    command = f"{sys.executable} -m venv venv"
    return run_command(command, "Creating virtual environment")

def install_requirements():
    """Install Python requirements."""
    print("📚 Installing Python requirements...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    upgrade_cmd = f"{pip_cmd} install --upgrade pip"
    if not run_command(upgrade_cmd, "Upgrading pip"):
        return False
    
    # Install requirements
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    return run_command(install_cmd, "Installing requirements")

def setup_environment_file():
    """Set up the environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("🔧 .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    print("🔧 Creating .env file from template...")
    try:
        env_example.rename(env_file)
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/interim", 
        "data/processed",
        "data/structured",
        "data/vector_db",
        "models",
        "logs",
        "reports/figures"
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files
            gitkeep = Path(directory) / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        
        print("✅ Directories created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def test_installation():
    """Test the installation by importing key modules."""
    print("🧪 Testing installation...")
    
    # Determine python command based on OS
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    test_script = '''
import sys
try:
    import pandas
    import streamlit
    import yfinance
    import chromadb
    print("✅ All core modules imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run(
            [python_cmd, "-c", test_script],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e.stderr}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print()
    print("2. Edit .env file with your API keys (optional):")
    print("   - GITHUB_AI_API_KEY for enhanced AI features")
    print("   - REDDIT_CLIENT_ID/SECRET for sentiment analysis")
    print("   - ALPHA_VANTAGE_API_KEY for additional data")
    print()
    print("3. Run the application:")
    print("   streamlit run src/streamlit_dashboard.py")
    print()
    print("4. Access the dashboard at:")
    print("   http://localhost:8501")
    print()
    print("For more information, see README.md")

def main():
    """Main setup function."""
    print("="*60)
    print("🚀 AI-Powered Financial Research Agent Setup")
    print("="*60)
    print()
    
    # Change to project directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Checking pip", check_pip),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing requirements", install_requirements),
        ("Setting up environment file", setup_environment_file),
        ("Creating directories", create_directories),
        ("Testing installation", test_installation),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n❌ Setup failed. Failed steps: {', '.join(failed_steps)}")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    else:
        show_next_steps()

if __name__ == "__main__":
    main()