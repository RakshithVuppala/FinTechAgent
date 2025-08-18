#!/usr/bin/env python3
"""
LLM Configuration Fix Script
Updates the agents to use OpenAI directly instead of GitHub AI
"""

import os
import sys

def update_financial_agent():
    """Update financial agent to use OpenAI directly"""
    
    agent_file = "src/agents/financial_agent.py"
    
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Replace GitHub AI configuration with OpenAI
    content = content.replace(
        '''self.llm_client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv("GITHUB_AI_API_KEY"),
                )''',
        '''self.llm_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                )'''
    )
    
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {agent_file}")

def update_market_agent():
    """Update market agent to use OpenAI directly"""
    
    agent_file = "src/agents/market_agent.py"
    
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Replace GitHub AI configuration with OpenAI
    content = content.replace(
        '''self.llm_client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv('GITHUB_AI_API_KEY')
                    )''',
        '''self.llm_client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                    )'''
    )
    
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {agent_file}")

def update_dashboard():
    """Update dashboard to check OpenAI instead of GitHub AI"""
    
    dashboard_file = "src/streamlit_dashboard.py"
    
    with open(dashboard_file, 'r') as f:
        content = f.read()
    
    # Update the check_api_status function
    old_check = '''    # GitHub AI API  
    github_api_key = os.getenv("GITHUB_AI_API_KEY")
    # Also check Streamlit secrets format
    if not github_api_key:
        github_api_key = st.secrets.get("api_keys", {}).get("GITHUB_AI_API_KEY")
    
    if github_api_key and github_api_key != "your_github_ai_api_key_here" and github_api_key.strip():
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://models.github.ai/inference",
                api_key=github_api_key,
            )
            # Try a simple test call
            response = client.models.list()
            api_status["github_ai"] = {"status": "configured", "message": "GitHub AI API working"}
        except Exception as e:
            api_status["github_ai"] = {"status": "error", "message": f"GitHub AI error: {str(e)[:50]}..."}
    else:
        api_status["github_ai"] = {"status": "missing", "message": "GitHub AI API key not configured"}'''

    new_check = '''    # OpenAI API (primary)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Also check Streamlit secrets format
    if not openai_api_key:
        openai_api_key = st.secrets.get("api_keys", {}).get("OPENAI_API_KEY")
    
    if openai_api_key and openai_api_key != "your_openai_api_key_here" and openai_api_key.strip():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            # Try a simple test call
            response = client.models.list()
            api_status["openai"] = {"status": "configured", "message": "OpenAI API working"}
        except Exception as e:
            api_status["openai"] = {"status": "error", "message": f"OpenAI error: {str(e)[:50]}..."}
    else:
        api_status["openai"] = {"status": "missing", "message": "OpenAI API key not configured"}'''
    
    if old_check in content:
        content = content.replace(old_check, new_check)
        print("Updated dashboard API status check")
    else:
        print("Could not find exact API status check to replace")
    
    # Update the LLM availability check
    old_llm_check = '''    # Check GitHub AI first
    if api_status["github_ai"]["status"] == "configured":
        return True, "GitHub AI API ready"
    
    # Fallback to OpenAI
    if api_status["openai"]["status"] == "configured":
        return True, "OpenAI API ready"'''

    new_llm_check = '''    # Check OpenAI API
    if api_status["openai"]["status"] == "configured":
        return True, "OpenAI API ready"'''
    
    if old_llm_check in content:
        content = content.replace(old_llm_check, new_llm_check)
        print("Updated LLM availability check")
    
    with open(dashboard_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {dashboard_file}")

def main():
    """Run all updates"""
    print("Fixing LLM Configuration...")
    print("=" * 40)
    
    try:
        update_financial_agent()
        update_market_agent()
        update_dashboard()
        
        print("\n" + "=" * 40)
        print("SUCCESS: All files updated!")
        print("\nNext steps:")
        print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
        print("2. Update OPENAI_API_KEY in your .env file")
        print("3. Run: python test_llm_connectivity.py")
        print("4. Restart your Streamlit application")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()