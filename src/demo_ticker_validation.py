"""
Demo script showing the ticker validation in action
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from financial_data_collector import FinancialDataCollector
import asyncio
from agents.orchestrator_agent import InvestmentResearchOrchestrator

async def demo_ticker_validation():
    """Demo the ticker validation system"""
    
    print("Ticker Validation Demo")
    print("=" * 40)
    
    collector = FinancialDataCollector()
    orchestrator = InvestmentResearchOrchestrator(use_cache=False, use_llm=False)
    
    # Demo scenarios
    test_tickers = [
        "INVALIDTICKER",  # Invalid - too long
        "APPL",          # Invalid - close to AAPL  
        "TESLA",         # Invalid - should suggest TSLA
        "AAPL",          # Valid
    ]
    
    for ticker in test_tickers:
        print(f"\nTesting ticker: '{ticker}'")
        print("-" * 30)
        
        # Test validation
        is_valid, message, suggestions = collector.validate_ticker(ticker)
        
        if is_valid:
            print(f"Valid ticker: {message}")
            
            # For valid tickers, show that research would proceed
            print("  - Research would proceed normally")
        else:
            print(f"Invalid ticker: {message}")
            
            if suggestions:
                print(f"  - Suggestions: {', '.join(suggestions[:3])}")
                
                # Demo what happens in orchestrator
                print("  - Testing orchestrator response...")
                try:
                    await orchestrator.research_company(ticker, "Demo Company")
                except ValueError as e:
                    error_msg = str(e)
                    if "Suggested alternatives:" in error_msg:
                        alts = error_msg.split("Suggested alternatives:")[-1].strip()
                        print(f"  - Orchestrator suggests: {alts}")
                    else:
                        print(f"  - Orchestrator error: {error_msg}")
        
        print()

def demo_user_experience():
    """Show what the user experience would be like"""
    
    print("\n" + "=" * 50)
    print("User Experience Demo")
    print("=" * 50)
    
    print("Scenario: User enters 'TESLA' expecting to research Tesla Inc.")
    print()
    
    collector = FinancialDataCollector()
    is_valid, message, suggestions = collector.validate_ticker("TESLA")
    
    print("System Response:")
    print(f"Invalid Ticker: TESLA")
    print(f"{message}")
    print()
    print("Did you mean one of these?")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    print()
    print("User clicks on 'TSLA' - System auto-fills TSLA and starts research")
    
    print("\n" + "-" * 50)
    print("Scenario: User enters 'AAPL'")
    print()
    
    is_valid, message, suggestions = collector.validate_ticker("AAPL")
    print("System Response:")
    if is_valid:
        print(f"Valid ticker: {message}")
        print("Research proceeds immediately")
    
if __name__ == "__main__":
    print("Running Ticker Validation Demo")
    print("This shows how the system handles invalid tickers")
    print("=" * 60)
    
    # Run async demo
    asyncio.run(demo_ticker_validation())
    
    # Run user experience demo
    demo_user_experience()
    
    print("\nDemo completed!")
    print("The system now validates tickers and provides helpful suggestions!")