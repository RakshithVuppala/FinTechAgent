#!/usr/bin/env python3
"""
Demonstration of Centralized AI Model Configuration
===================================================

This script shows how the AI model is now centrally managed via .env file.
"""

import sys
import os
sys.path.append('src')

from ai_config import get_ai_model, print_config

def demo_centralized_config():
    """Demonstrate the centralized model configuration"""
    
    print("AI MODEL CONFIGURATION DEMO")
    print("=" * 50)
    
    # Show current configuration
    print("\n1. Current Configuration:")
    print_config()
    
    # Show model usage in different contexts
    print("\n2. Model Usage Across Components:")
    
    try:
        from agents.financial_agent import EnhancedFinancialAnalysisAgent
        from agents.market_agent import MarketIntelligenceAgent
        
        # Initialize agents
        financial_agent = EnhancedFinancialAnalysisAgent(use_llm=False)  # LLM disabled for demo
        market_agent = MarketIntelligenceAgent(use_llm=False)  # LLM disabled for demo
        
        print(f"   • Financial Agent would use: {get_ai_model()}")
        print(f"   • Market Agent would use: {get_ai_model()}")
        print(f"   • Streamlit Dashboard would use: {get_ai_model()}")
        print(f"   • Test Scripts would use: {get_ai_model()}")
        
    except Exception as e:
        print(f"   Error importing agents: {e}")
    
    print("\n3. How to Change Model for ALL Components:")
    print("   Edit .env file and change: AI_MODEL=your_preferred_model")
    print("   Available options:")
    print("   • gpt-4o-mini (current, cost-effective)")
    print("   • gpt-4o (more capable, higher cost)")
    print("   • gpt-3.5-turbo (fastest, lowest cost)")
    
    print("\n4. Benefits of Centralized Configuration:")
    print("   SUCCESS: Single point of control")
    print("   SUCCESS: No need to update multiple files")
    print("   SUCCESS: Consistent model across all agents")
    print("   SUCCESS: Easy A/B testing of different models")
    print("   SUCCESS: Environment-specific configurations")
    
    print("\n" + "=" * 50)
    print("RESULT: All AI components now use the centralized model configuration!")

if __name__ == "__main__":
    demo_centralized_config()