#!/usr/bin/env python3
"""
Test exactly what Streamlit sees for LLM availability
"""

import os
import sys
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

# Mock Streamlit for testing
class MockSecrets:
    def get(self, key, default=None):
        if key == "api_keys":
            return {}
        return default

# Mock streamlit module
import types
st_mock = types.ModuleType('streamlit')
st_mock.secrets = MockSecrets()

# Add to sys.modules so imports work
sys.modules['streamlit'] = st_mock

def test_check_functions():
    """Test the exact functions used in Streamlit"""
    
    print("Testing check_api_status()...")
    
    # Import after mocking streamlit
    from streamlit_dashboard import check_api_status, check_llm_availability
    
    # Test API status
    api_status = check_api_status()
    print(f"API Status Result: {api_status}")
    
    # Test LLM availability
    is_available, status_msg = check_llm_availability()
    print(f"LLM Available: {is_available}")
    print(f"Status Message: {status_msg}")
    
    return is_available, status_msg

def test_direct_api_call():
    """Test the direct API call that should work"""
    print("\nTesting direct API call...")
    
    try:
        from openai import OpenAI
        
        github_api_key = os.getenv("GITHUB_AI_API_KEY")
        print(f"Using API key: {github_api_key[:10] if github_api_key else 'None'}...")
        
        client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=github_api_key,
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        print("Direct API call: SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"Direct API call: FAILED - {e}")
        return False

def main():
    print("Testing Streamlit LLM Check")
    print("=" * 40)
    
    # Test environment
    github_key = os.getenv("GITHUB_AI_API_KEY")
    print(f"GitHub API Key loaded: {bool(github_key)}")
    if github_key:
        print(f"Key starts with: {github_key[:10]}...")
    
    print("\n" + "=" * 40)
    
    # Test direct API call
    direct_works = test_direct_api_call()
    
    print("\n" + "=" * 40)
    
    # Test Streamlit functions
    streamlit_available, streamlit_msg = test_check_functions()
    
    print("\n" + "=" * 40)
    print("COMPARISON:")
    print(f"Direct API call works: {direct_works}")
    print(f"Streamlit check says: {streamlit_available}")
    
    if direct_works and not streamlit_available:
        print("\nPROBLEM IDENTIFIED:")
        print("Direct API works but Streamlit check fails")
        print("This suggests an issue in the check_api_status() function")
    elif direct_works and streamlit_available:
        print("\nEVERYTHING WORKS:")
        print("Both direct API and Streamlit check succeed")
        print("The issue might be in Streamlit session state or display")
    else:
        print("\nAPI ISSUE:")
        print("Direct API call is failing")

if __name__ == "__main__":
    main()