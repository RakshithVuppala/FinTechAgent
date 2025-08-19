#!/usr/bin/env python3
"""
LLM Connectivity Test Script
Tests all available LLM APIs and provides diagnosis
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for ai_config import
sys.path.append('src')
from ai_config import get_ai_model

def test_github_ai():
    """Test GitHub AI API connectivity"""
    print("\n=== Testing GitHub AI API ===")
    
    github_api_key = os.getenv('GITHUB_AI_API_KEY')
    
    if not github_api_key or github_api_key == 'your_github_ai_api_key_here':
        print("ERROR: GitHub AI API key not configured")
        print("SOLUTION: Get a GitHub AI API key from https://docs.github.com/en/github-models")
        return False
    
    print(f"API Key found: {github_api_key[:10]}...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=github_api_key
        )
        
        # Test chat completion instead of models.list() which may not be supported
        try:
            response = client.chat.completions.create(
                model=get_ai_model(),
                messages=[{"role": "user", "content": "Say 'GitHub AI is working'"}],
                max_tokens=10
            )
            print(f"SUCCESS: {response.choices[0].message.content}")
            return True
                
        except Exception as e:
            print(f"ERROR: API call failed: {e}")
            if "Unauthorized" in str(e):
                print("SOLUTION: Your GitHub AI API key appears to be invalid or expired")
                print("1. Check if the key is correct in your .env file")
                print("2. Generate a new key at https://github.com/settings/tokens")
                print("3. Ensure the key has proper permissions for GitHub Models")
            return False
            
    except ImportError:
        print("ERROR: OpenAI library not installed")
        print("SOLUTION: Run 'pip install openai'")
        return False

def test_openai_direct():
    """Test OpenAI API directly"""
    print("\n=== Testing OpenAI API ===")
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        print("ERROR: OpenAI API key not configured")
        print("SOLUTION: Get an OpenAI API key from https://platform.openai.com/api-keys")
        return False
    
    print(f"API Key found: {openai_api_key[:10]}...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai_api_key)
        
        # Test models list
        try:
            models = client.models.list()
            print(f"SUCCESS: Found {len(models.data)} models available")
            
            # Test chat completion
            response = client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {"role": "user", "content": "Say 'OpenAI is working'"}
                ],
                max_tokens=10
            )
            print(f"SUCCESS: {response.choices[0].message.content}")
            return True
            
        except Exception as e:
            print(f"ERROR: API call failed: {e}")
            if "Unauthorized" in str(e):
                print("SOLUTION: Your OpenAI API key appears to be invalid")
                print("1. Check if the key is correct in your .env file")
                print("2. Verify your OpenAI account has credits")
                print("3. Generate a new key at https://platform.openai.com/api-keys")
            return False
            
    except ImportError:
        print("ERROR: OpenAI library not installed")
        print("SOLUTION: Run 'pip install openai'")
        return False

def test_agent_initialization():
    """Test if our agents can initialize properly"""
    print("\n=== Testing Agent Initialization ===")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from agents.financial_agent import EnhancedFinancialAnalysisAgent
        from agents.market_agent import MarketIntelligenceAgent
        
        # Test financial agent
        print("Testing Financial Agent...")
        financial_agent = EnhancedFinancialAnalysisAgent(use_llm=True)
        if financial_agent.use_llm:
            print("SUCCESS: Financial Agent initialized with LLM")
        else:
            print("WARNING: Financial Agent initialized without LLM")
        
        # Test market agent  
        print("Testing Market Intelligence Agent...")
        market_agent = MarketIntelligenceAgent(use_llm=True)
        if market_agent.use_llm:
            print("SUCCESS: Market Agent initialized with LLM")
        else:
            print("WARNING: Market Agent initialized without LLM")
            
        return financial_agent.use_llm or market_agent.use_llm
        
    except Exception as e:
        print(f"ERROR: Agent initialization failed: {e}")
        return False

def main():
    """Run all LLM connectivity tests"""
    print("LLM Connectivity Diagnostic Tool")
    print("=" * 50)
    
    results = []
    
    # Test GitHub AI
    github_working = test_github_ai()
    results.append(("GitHub AI", github_working))
    
    # Test OpenAI
    openai_working = test_openai_direct()
    results.append(("OpenAI", openai_working))
    
    # Test agent initialization
    agents_working = test_agent_initialization()
    results.append(("Agents", agents_working))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for service, working in results:
        status = "WORKING" if working else "FAILED"
        print(f"{service:15}: {status}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if not any(result[1] for result in results[:2]):  # No LLM APIs working
        print("1. No LLM APIs are working. You need to configure at least one:")
        print("   - GitHub AI API: https://docs.github.com/en/github-models")
        print("   - OpenAI API: https://platform.openai.com/api-keys")
        print("2. Update your .env file with a valid API key")
        print("3. Restart the Streamlit application")
    elif github_working:
        print("1. GitHub AI is working - your application should work normally")
    elif openai_working:
        print("1. OpenAI is working - your application should work normally")
        print("2. Consider updating agents to use OpenAI directly if GitHub AI continues to fail")
    
    print(f"\n4. Run this script again after making changes: python {__file__}")

if __name__ == "__main__":
    main()