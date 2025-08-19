"""
AI Configuration Management
===========================

Centralized configuration for AI models across all agents.
Change the model in .env file to update all agents at once.
"""

import os
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_ai_model() -> str:
    """
    Get the AI model from environment variables with fallback.
    
    Returns:
        str: The AI model to use (e.g., 'gpt-4o-mini', 'gpt-4o')
    """
    return os.getenv('AI_MODEL', 'gpt-4o-mini')

def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment variables.
    
    Returns:
        Optional[str]: The OpenAI API key or None if not found
    """
    return os.getenv('OPENAI_API_KEY')

def get_github_ai_api_key() -> Optional[str]:
    """
    Get GitHub AI API key from environment variables.
    
    Returns:
        Optional[str]: The GitHub AI API key or None if not found
    """
    return os.getenv('GITHUB_AI_API_KEY')

def get_model_config() -> dict:
    """
    Get complete model configuration.
    
    Returns:
        dict: Configuration dictionary with model and API keys
    """
    return {
        'model': get_ai_model(),
        'openai_api_key': get_openai_api_key(),
        'github_ai_api_key': get_github_ai_api_key(),
    }

def validate_ai_config() -> bool:
    """
    Validate that AI configuration is properly set.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    model = get_ai_model()
    openai_key = get_openai_api_key()
    github_key = get_github_ai_api_key()
    
    # Check if we have at least one valid API key
    has_valid_key = bool(openai_key and openai_key != 'your_openai_api_key_here') or \
                   bool(github_key and github_key != 'your_github_ai_api_key_here')
    
    return bool(model) and has_valid_key

# Constants for easy import
AI_MODEL = get_ai_model()
OPENAI_API_KEY = get_openai_api_key()
GITHUB_AI_API_KEY = get_github_ai_api_key()

# For backward compatibility and easy debugging
def print_config():
    """Print current AI configuration (for debugging)"""
    config = get_model_config()
    print("=== AI Configuration ===")
    print(f"Model: {config['model']}")
    print(f"OpenAI API Key: {'Set' if config['openai_api_key'] else 'Not Set'}")
    print(f"GitHub AI API Key: {'Set' if config['github_ai_api_key'] else 'Not Set'}")
    print(f"Config Valid: {validate_ai_config()}")
    print("========================")

if __name__ == "__main__":
    # Test the configuration when run directly
    print_config()