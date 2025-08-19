#!/usr/bin/env python3
"""
Test Different Tickers - Verify that agents produce different scores for different companies
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_ticker_variations():
    """Test that different ticker data produces different scores"""
    print("Testing Financial Agent with Different Company Data")
    print("=" * 60)
    
    try:
        from agents.financial_agent import EnhancedFinancialAnalysisAgent
        
        # Test data for different types of companies
        companies = {
            "AAPL": {  # Large tech company - should score high
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "financial_metrics": {
                    "market_cap": 2800000000000,  # $2.8T
                    "pe_ratio": 22.5,
                    "beta": 1.1,
                    "current_price": 180.50
                }
            },
            "TSLA": {  # High growth, high risk - different score
                "ticker": "TSLA", 
                "company_name": "Tesla Inc.",
                "financial_metrics": {
                    "market_cap": 800000000000,  # $800B
                    "pe_ratio": 45.2,  # High PE
                    "beta": 2.1,  # High beta (risky)
                    "current_price": 250.30
                }
            },
            "STARTUP": {  # Small company - should score differently
                "ticker": "STARTUP",
                "company_name": "Small Startup Corp.",
                "financial_metrics": {
                    "market_cap": 500000000,  # $500M (small)
                    "pe_ratio": 8.5,  # Low PE
                    "beta": 0.7,  # Low beta
                    "current_price": 15.75
                }
            }
        }
        
        agent = EnhancedFinancialAnalysisAgent(use_llm=False)
        scores = {}
        
        for ticker, data in companies.items():
            result = agent.analyze_company_financials(data)
            overall_score = result.get('overall_assessment', {}).get('overall_score', 0)
            scores[ticker] = overall_score
            print(f"{ticker}: Score = {overall_score:.1f}")
        
        # Check if scores are different
        unique_scores = len(set(scores.values()))
        print(f"\nUnique scores generated: {unique_scores}")
        print(f"Companies tested: {len(companies)}")
        
        if unique_scores > 1:
            print("SUCCESS: Different tickers produce different scores!")
            return True
        else:
            print("WARNING: All tickers produced the same score!")
            print("This suggests the scoring logic may need refinement.")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False

def main():
    """Run ticker variation test"""
    success = test_ticker_variations()
    
    print("\n" + "=" * 60)
    if success:
        print("RESULT: Recommendation system will produce different results for different stocks!")
    else:
        print("RESULT: Further investigation needed - scores may still be identical.")

if __name__ == "__main__":
    main()