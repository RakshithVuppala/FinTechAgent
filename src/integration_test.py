"""
Integration Test - Connect all Phase 1 components
Tests: Data Collection → Storage → Analysis pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from financial_data_collector import FinancialDataCollector
from data_manager import StructuredDataManager
from agents.financial_agent import EnhancedFinancialAnalysisAgent
import json
from datetime import datetime

def run_integration_test(ticker: str = "AAPL"):
    """
    End-to-end test of the investment research pipeline
    """
    print(f"🚀 Starting Integration Test for {ticker}")
    print("=" * 60)
    
    # Step 1: Collect Data
    print("📊 Step 1: Collecting financial data...")
    collector = FinancialDataCollector()
    separated_data = collector.collect_separated_data(ticker, f"{ticker} Company")
    
    if not separated_data:
        print("❌ Data collection failed")
        return False
    
    print(f"✅ Data collected: {separated_data['collection_metadata']}")
    
    # Step 2: Store Data  
    print("\n💾 Step 2: Storing structured data...")
    data_manager = StructuredDataManager()
    storage_success = data_manager.store_company_data(ticker, separated_data['structured'])
    
    if not storage_success:
        print("❌ Data storage failed")
        return False
    
    print("✅ Data stored successfully")
    
    # Step 3: Retrieve and Analyze
    print("\n🧠 Step 3: Analyzing with AI agent...")
    agent = EnhancedFinancialAnalysisAgent(use_llm=True)
    
    # Get stored data
    stored_data = data_manager.get_latest_company_data(ticker)
    if not stored_data:
        print("❌ Could not retrieve stored data")
        return False
    
    # Run analysis
    analysis = agent.analyze_company_financials(stored_data)
    
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return False
    
    print("✅ Analysis completed successfully")
    
    # Step 4: Display Results
    print("\n📈 Step 4: Investment Analysis Results")
    print("=" * 60)
    
    # Company Overview
    overview = analysis.get('company_overview', {})
    print(f"Company: {overview.get('company_name', 'N/A')} ({ticker})")
    print(f"Sector: {overview.get('sector', 'N/A')}")
    print(f"Current Price: ${overview.get('current_price', 0):,.2f}")
    print(f"Market Cap: ${overview.get('market_cap', 0):,.0f}")
    print(f"Category: {overview.get('market_cap_category', 'N/A')}")
    
    # Valuation
    valuation = analysis.get('valuation_analysis', {})
    print(f"\nP/E Ratio: {valuation.get('pe_ratio', 'N/A')}")
    print(f"P/E Assessment: {valuation.get('pe_assessment', 'N/A')}")
    print(f"Dividend Yield: {valuation.get('dividend_yield', 0)*100:.2f}%")
    
    # Risk
    risk = analysis.get('risk_assessment', {})
    print(f"\nBeta: {risk.get('beta', 'N/A')}")
    print(f"Risk Level: {risk.get('beta_assessment', 'N/A')}")
    print(f"Price Position: {risk.get('price_assessment', 'N/A')}")
    
    # Overall Assessment
    overall = analysis.get('overall_assessment', {})
    print(f"\nOverall Rating: {overall.get('overall_rating', 'N/A')}")
    print(f"Score: {overall.get('score', 0)}/{overall.get('max_score', 3)}")
    
    # LLM Insights (if available)
    llm_insights = analysis.get('llm_insights', {})
    if 'detailed_analysis' in llm_insights:
        print(f"\n🤖 AI Analyst Insights:")
        print("-" * 40)
        print(llm_insights['detailed_analysis'][:300] + "...")
    
    # Investment Recommendation (if available)
    recommendation = analysis.get('investment_recommendation', {})
    if 'recommendation_text' in recommendation:
        print(f"\n💡 Investment Recommendation:")
        print("-" * 40)
        print(recommendation['recommendation_text'][:200] + "...")
    
    # Step 5: Save Complete Report
    print(f"\n💾 Step 5: Saving complete report...")
    report_filename = f"reports/{ticker}_investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    complete_report = {
        'ticker': ticker,
        'analysis_date': datetime.now().isoformat(),
        'raw_data': separated_data,
        'analysis_results': analysis,
        'pipeline_metadata': {
            'data_sources_used': separated_data['collection_metadata'],
            'analysis_agent': 'EnhancedFinancialAnalysisAgent',
            'llm_enabled': 'detailed_analysis' in llm_insights
        }
    }
    
    with open(report_filename, 'w') as f:
        json.dump(complete_report, f, indent=2, default=str)
    
    print(f"✅ Complete report saved: {report_filename}")
    
    print(f"\n🎉 Integration Test SUCCESSFUL for {ticker}!")
    print("=" * 60)
    print("✅ Data Collection → Storage → Analysis → Report = WORKING")
    
    return True

if __name__ == "__main__":
    # Test with Apple
    success = run_integration_test("AAPL")
    
    if success:
        print("\n🚀 Phase 1 Complete! Ready for Phase 2")
        print("\nNext Steps:")
        print("- Build Market Intelligence Agent (news analysis)")
        print("- Create Orchestrator Agent")  
        print("- Add report generation")
    else:
        print("\n❌ Integration test failed. Check errors above.")