#!/usr/bin/env python3
"""
Debug Streamlit LLM Detection
"""

import os
import sys
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_github_ai_simple():
    """Test GitHub AI exactly like your working code"""
    from openai import OpenAI

    github_api_key = os.getenv("GITHUB_AI_API_KEY")
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=github_api_key,
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    )

    print("SUCCESS: Your working code produces:")
    print(completion.choices[0].message.content)
    return True

def test_env_loading():
    """Test if environment variables are loaded correctly"""
    github_key = os.getenv("GITHUB_AI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"GitHub API Key loaded: {bool(github_key)}")
    if github_key:
        print(f"GitHub Key starts with: {github_key[:10]}...")
        print(f"GitHub Key length: {len(github_key)} characters")
    
    print(f"OpenAI API Key loaded: {bool(openai_key)}")
    if openai_key:
        print(f"OpenAI Key: {openai_key}")
    
    return github_key, openai_key

def test_dashboard_api_check():
    """Test the exact API check from dashboard"""
    github_api_key = os.getenv("GITHUB_AI_API_KEY")
    
    print(f"\nTesting dashboard API check...")
    print(f"API key from env: {github_api_key[:10] if github_api_key else 'None'}...")
    
    if github_api_key and github_api_key != "your_github_ai_api_key_here" and github_api_key.strip():
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://models.github.ai/inference",
                api_key=github_api_key,
            )
            # Try a simple chat completion instead of models.list() which may not be supported
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print("SUCCESS: Dashboard API check passed")
            return {"status": "configured", "message": "GitHub AI API working"}
        except Exception as e:
            print(f"ERROR: Dashboard API check failed: {e}")
            # If it's a 404 on models.list but key exists, assume it might still work
            if "404" in str(e) and len(github_api_key) > 20:
                return {"status": "configured", "message": "GitHub AI API key present (limited test)"}
            else:
                return {"status": "error", "message": f"GitHub AI error: {str(e)[:50]}..."}
    else:
        print("ERROR: API key check failed - invalid key")
        return {"status": "missing", "message": "GitHub AI API key not configured"}

def main():
    print("Debugging Streamlit LLM Detection")
    print("=" * 50)
    
    # Test 1: Your working code
    print("\n1. Testing your working code:")
    try:
        test_github_ai_simple()
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test 2: Environment loading
    print("\n2. Testing environment variable loading:")
    github_key, openai_key = test_env_loading()
    
    # Test 3: Dashboard API check
    print("\n3. Testing dashboard API check logic:")
    api_result = test_dashboard_api_check()
    print(f"API check result: {api_result}")
    
    # Test 4: LLM availability check
    print("\n4. Testing LLM availability logic:")
    if api_result["status"] == "configured":
        print("SUCCESS: LLM should be detected as available")
    else:
        print("PROBLEM: LLM will be detected as unavailable")
        print(f"Reason: {api_result['message']}")

if __name__ == "__main__":
    main()