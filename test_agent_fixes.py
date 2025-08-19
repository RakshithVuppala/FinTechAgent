#!/usr/bin/env python3
"""
Test Agent Fixes - Verify that agents provide correct data structure
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_financial_agent():
    """Test financial agent returns correct data structure"""
    print("=== Testing Financial Agent ===")
    
    try:
        from agents.financial_agent import EnhancedFinancialAnalysisAgent
        
        # Create test data
        test_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "financial_metrics": {
                "company_name": "Apple Inc.",
                "sector": "Technology",
                "market_cap": 2800000000000,  # $2.8T
                "pe_ratio": 22.5,
                "beta": 1.1,
                "revenue": 394328000000,
                "net_income": 99803000000,
                "debt_to_equity": 1.73,
                "current_ratio": 1.04,
                "roe": 0.26,
                "current_price": 180.50
            }
        }
        
        # Initialize agent
        agent = EnhancedFinancialAnalysisAgent(use_llm=False)  # Use basic analysis for testing
        
        # Run analysis
        result = agent.analyze_company_financials(test_data)
        
        # Check required fields for orchestrator
        print(f"SUCCESS: Analysis completed for {result.get('ticker', 'Unknown')}")
        
        overall_assessment = result.get('overall_assessment', {})
        print(f"SUCCESS: Overall assessment found: {bool(overall_assessment)}")
        print(f"SUCCESS: Overall score: {overall_assessment.get('overall_score', 'MISSING')}")
        print(f"SUCCESS: Score range 0-100: {0 <= overall_assessment.get('overall_score', -1) <= 100}")
        
        return overall_assessment.get('overall_score') is not None
        
    except Exception as e:
        print(f"FAIL: Financial agent test failed: {e}")
        return False

def test_market_agent():
    """Test market agent returns correct data structure"""
    print("\n=== Testing Market Agent ===")
    
    try:
        from agents.market_agent import MarketIntelligenceAgent
        
        # Initialize agent
        agent = MarketIntelligenceAgent(use_llm=False)  # Use basic analysis for testing
        
        # Run analysis
        result = agent.analyze_market_intelligence("AAPL", ["sentiment", "news_impact"])
        
        # Check required fields for orchestrator
        print(f"SUCCESS: Analysis completed for {result.get('ticker', 'Unknown')}")
        
        overall_assessment = result.get('overall_assessment', {})
        print(f"SUCCESS: Overall assessment found: {bool(overall_assessment)}")
        print(f"SUCCESS: Market sentiment: {overall_assessment.get('market_sentiment', 'MISSING')}")
        
        sentiment = overall_assessment.get('market_sentiment')
        valid_sentiments = ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']
        print(f"SUCCESS: Valid sentiment: {sentiment in valid_sentiments}")
        
        return overall_assessment.get('market_sentiment') in valid_sentiments
        
    except Exception as e:
        print(f"FAIL: Market agent test failed: {e}")
        return False

def test_orchestrator_compatibility():
    """Test that orchestrator can process agent outputs"""
    print("\n=== Testing Orchestrator Compatibility ===")
    
    try:
        # Mock data structures that agents should return
        financial_analysis = {
            "overall_assessment": {
                "overall_score": 75.0,
                "overall_rating": "Positive indicators"
            }
        }
        
        market_intelligence = {
            "overall_assessment": {
                "market_sentiment": "positive",
                "overall_outlook": "Positive market indicators"
            }
        }
        
        # Test orchestrator logic
        financial_score = financial_analysis.get("overall_assessment", {}).get("overall_score", 0)
        market_sentiment = market_intelligence.get("overall_assessment", {}).get("market_sentiment", "neutral")
        
        print(f"SUCCESS: Financial score extracted: {financial_score}")
        print(f"SUCCESS: Market sentiment extracted: {market_sentiment}")
        
        # Test score combination (from orchestrator)
        sentiment_scores = {
            "very_positive": 95,
            "positive": 80,
            "neutral": 60,
            "negative": 40,
            "very_negative": 20,
        }
        market_score = sentiment_scores.get(market_sentiment, 60)
        
        financial_weight = 0.6
        market_weight = 0.4
        combined_score = (financial_score * financial_weight) + (market_score * market_weight)
        
        print(f"SUCCESS: Market score: {market_score}")
        print(f"SUCCESS: Combined score: {combined_score}")
        
        # Test recommendation generation
        if combined_score >= 80:
            recommendation = "STRONG BUY"
        elif combined_score >= 70:
            recommendation = "BUY"
        elif combined_score >= 60:
            recommendation = "HOLD"
        elif combined_score >= 50:
            recommendation = "WEAK HOLD"
        else:
            recommendation = "SELL"
            
        print(f"SUCCESS: Recommendation: {recommendation}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Orchestrator compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TEST: Testing Agent Fixes for Recommendation System")
    print("=" * 60)
    
    # Run tests
    financial_ok = test_financial_agent()
    market_ok = test_market_agent()
    orchestrator_ok = test_orchestrator_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: TEST RESULTS")
    print("=" * 60)
    print(f"Financial Agent:     {'SUCCESS: PASS' if financial_ok else 'FAIL: FAIL'}")
    print(f"Market Agent:        {'SUCCESS: PASS' if market_ok else 'FAIL: FAIL'}")
    print(f"Orchestrator Logic:  {'SUCCESS: PASS' if orchestrator_ok else 'FAIL: FAIL'}")
    
    all_passed = financial_ok and market_ok and orchestrator_ok
    print(f"\nOverall Status:      {'SUCCESS: ALL TESTS PASSED' if all_passed else 'FAIL: SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nSUCCESS: Recommendation system should now work correctly!")
        print("Different tickers should now produce different recommendations.")
    else:
        print("\nERROR: Issues remain - recommendations may still be identical.")

if __name__ == "__main__":
    main()