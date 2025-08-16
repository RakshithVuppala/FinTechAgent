"""
Test script for ticker validation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from financial_data_collector import FinancialDataCollector
import asyncio
from agents.orchestrator_agent import InvestmentResearchOrchestrator

def test_ticker_validation():
    """Test the ticker validation functionality"""
    
    print("Testing Ticker Validation System")
    print("=" * 50)
    
    collector = FinancialDataCollector()
    
    # Test cases
    test_cases = [
        # Valid tickers
        ("AAPL", True),
        ("MSFT", True),
        ("GOOGL", True),
        
        # Invalid tickers
        ("INVALIDTICKER", False),
        ("NOTREAL", False),
        ("XYZ123", False),
        ("", False),
        ("APL", False),  # Close to AAPL
        ("MSFTT", False),  # Close to MSFT
        ("TESLA", False),  # Should suggest TSLA
        ("AMAZON", False),  # Should suggest AMZN
        ("APPLE", False),  # Should suggest AAPL
        
        # Edge cases
        ("123", False),
        ("!@#", False),
        ("AVERYLONGTICKERNAME", False),
    ]
    
    print("Testing individual ticker validation:")
    print("-" * 40)
    
    for ticker, expected_valid in test_cases:
        try:
            is_valid, message, suggestions = collector.validate_ticker(ticker)
            
            status = "PASS" if is_valid == expected_valid else "FAIL"
            print(f"{status} {ticker:15} | Valid: {is_valid:5} | {message}")
            
            if suggestions:
                print(f"{'':20} | Suggestions: {', '.join(suggestions[:3])}")
                
        except Exception as e:
            print(f"ERROR {ticker:15} | Exception: {str(e)}")
        
        print()

async def test_orchestrator_validation():
    """Test ticker validation in the orchestrator"""
    
    print("\n" + "=" * 50)
    print("Testing Orchestrator Ticker Validation")
    print("=" * 50)
    
    orchestrator = InvestmentResearchOrchestrator(
        use_cache=False,
        use_llm=False,
        parallel_execution=False
    )
    
    # Test invalid ticker
    print("Testing invalid ticker 'INVALIDTICKER':")
    try:
        await orchestrator.research_company("INVALIDTICKER", "Invalid Company")
        print("FAIL - Should have failed with invalid ticker")
    except ValueError as e:
        print(f"PASS - Correctly caught invalid ticker: {str(e)}")
        
        # Check if suggestions are provided
        if "Suggested alternatives:" in str(e):
            print("PASS - Suggestions provided in error message")
        else:
            print("FAIL - No suggestions in error message")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
    
    print()
    
    # Test valid ticker
    print("Testing valid ticker 'AAPL':")
    try:
        # Just test validation part - don't run full research
        is_valid, message, suggestions = orchestrator.data_collector.validate_ticker("AAPL")
        if is_valid:
            print("PASS - AAPL is correctly identified as valid")
        else:
            print(f"FAIL - AAPL should be valid: {message}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

def test_suggestion_quality():
    """Test the quality of ticker suggestions"""
    
    print("\n" + "=" * 50)
    print("Testing Suggestion Quality")
    print("=" * 50)
    
    collector = FinancialDataCollector()
    
    # Test suggestion quality
    suggestion_tests = [
        ("APL", ["AAPL"]),  # Should suggest Apple
        ("TESLA", ["TSLA"]),  # Should suggest Tesla
        ("AMAZON", ["AMZN"]),  # Should suggest Amazon
        ("MSFTT", ["MSFT"]),  # Should suggest Microsoft
        ("GOOGEL", ["GOOGL"]),  # Should suggest Google
    ]
    
    for invalid_ticker, expected_suggestions in suggestion_tests:
        is_valid, message, suggestions = collector.validate_ticker(invalid_ticker)
        
        print(f"Input: {invalid_ticker}")
        print(f"Suggestions: {suggestions}")
        
        # Check if any expected suggestion is in the actual suggestions
        found_expected = any(exp in suggestions for exp in expected_suggestions)
        
        status = "PASS" if found_expected else "FAIL"
        print(f"{status} - Expected one of {expected_suggestions}")
        print()

if __name__ == "__main__":
    # Run all tests
    print("Running Ticker Validation Tests")
    print("=" * 60)
    
    # Test 1: Basic validation
    test_ticker_validation()
    
    # Test 2: Orchestrator integration
    asyncio.run(test_orchestrator_validation())
    
    # Test 3: Suggestion quality
    test_suggestion_quality()
    
    print("\nAll ticker validation tests completed!")
    print("=" * 60)