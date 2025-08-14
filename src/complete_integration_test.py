"""
Complete Integration Test - Full Investment Research Pipeline
Tests: Data Collection → Storage → Financial Analysis → Market Intelligence → Final Report
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from financial_data_collector import FinancialDataCollector
from data_manager import StructuredDataManager
from vector_manager import VectorDataManager
from agents.financial_agent import EnhancedFinancialAnalysisAgent
from agents.market_agent import MarketIntelligenceAgent
import json
from datetime import datetime
from typing import Dict, List, Any
def run_complete_investment_research(ticker: str = "AAPL", company_name: str = None):
    """
    Complete end-to-end investment research pipeline
    
    Pipeline:
    1. Collect all data (structured + unstructured)
    2. Store structured data in data manager
    3. Store unstructured data in vector database
    4. Run financial analysis on structured data
    5. Run market intelligence on unstructured data (RAG)
    6. Combine results into comprehensive investment report
    """
    
    print(f"🚀 COMPLETE INVESTMENT RESEARCH PIPELINE")
    print(f"🎯 Target: {ticker} ({company_name or ticker + ' Company'})")
    print("=" * 70)
    
    # Initialize all components
    print("\n🔧 Initializing Research Components...")
    collector = FinancialDataCollector()
    structured_manager = StructuredDataManager()
    vector_manager = VectorDataManager()
    financial_agent = EnhancedFinancialAnalysisAgent(use_llm=True)
    market_agent = MarketIntelligenceAgent(use_llm=True)
    
    research_results = {
        'ticker': ticker,
        'company_name': company_name,
        'research_timestamp': datetime.now().isoformat(),
        'pipeline_status': {},
        'financial_analysis': {},
        'market_intelligence': {},
        'combined_assessment': {}
    }
    
    # ==================== STEP 1: DATA COLLECTION ====================
    print(f"\n📊 STEP 1: Collecting Financial Data for {ticker}")
    print("-" * 50)
    
    try:
        separated_data = collector.collect_separated_data(ticker, company_name)
        
        if not separated_data:
            print("❌ Data collection failed")
            return None
        
        print(f"✅ Data collection successful")
        print(f"   📈 Structured sources: {separated_data['collection_metadata']['structured_sources']}")
        print(f"   📰 Unstructured sources: {separated_data['collection_metadata']['unstructured_sources']}")
        
        research_results['pipeline_status']['data_collection'] = 'success'
        research_results['data_summary'] = separated_data['collection_metadata']
        
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        research_results['pipeline_status']['data_collection'] = f'failed: {e}'
        return research_results
    
    # ==================== STEP 2: STORE STRUCTURED DATA ====================
    print(f"\n💾 STEP 2: Storing Structured Data")
    print("-" * 50)
    
    try:
        storage_success = structured_manager.store_company_data(ticker, separated_data['structured'])
        
        if storage_success:
            print("✅ Structured data stored successfully")
            research_results['pipeline_status']['structured_storage'] = 'success'
        else:
            print("⚠️ Structured data storage had issues")
            research_results['pipeline_status']['structured_storage'] = 'partial'
        
    except Exception as e:
        print(f"❌ Structured storage error: {e}")
        research_results['pipeline_status']['structured_storage'] = f'failed: {e}'
    
    # ==================== STEP 3: STORE UNSTRUCTURED DATA ====================
    print(f"\n🗂️ STEP 3: Storing Unstructured Data in Vector Database")
    print("-" * 50)
    
    try:
        vector_success = vector_manager.store_unstructured_data(ticker, separated_data['unstructured'])
        
        if vector_success:
            print("✅ Unstructured data vectorized and stored")
            
            # Show what was stored
            news_count = len(separated_data['unstructured'].get('news_articles', []))
            reddit_data = separated_data['unstructured'].get('reddit_sentiment', {})
            reddit_count = len(reddit_data.get('sample_posts', []))
            
            print(f"   📰 News articles stored: {news_count}")
            print(f"   🗨️ Reddit posts stored: {reddit_count}")
            
            research_results['pipeline_status']['vector_storage'] = 'success'
        else:
            print("⚠️ Vector storage had issues")
            research_results['pipeline_status']['vector_storage'] = 'partial'
        
    except Exception as e:
        print(f"❌ Vector storage error: {e}")
        research_results['pipeline_status']['vector_storage'] = f'failed: {e}'
    
    # ==================== STEP 4: FINANCIAL ANALYSIS ====================
    print(f"\n🧮 STEP 4: Running Financial Analysis (Structured Data)")
    print("-" * 50)
    
    try:
        # Get stored structured data
        stored_structured_data = structured_manager.get_latest_company_data(ticker)
        
        if stored_structured_data:
            financial_analysis = financial_agent.analyze_company_financials(stored_structured_data)
            
            if 'error' not in financial_analysis:
                print("✅ Financial analysis completed")
                
                # Display key results
                overview = financial_analysis.get('company_overview', {})
                valuation = financial_analysis.get('valuation_analysis', {})
                overall = financial_analysis.get('overall_assessment', {})
                
                print(f"   🏢 Company: {overview.get('company_name', 'N/A')}")
                print(f"   💰 Current Price: ${overview.get('current_price', 0):,.2f}")
                print(f"   📊 P/E Ratio: {valuation.get('pe_ratio', 'N/A')}")
                print(f"   🎯 Overall Rating: {overall.get('overall_rating', 'N/A')}")
                
                # Check for LLM insights
                llm_insights = financial_analysis.get('llm_insights', {})
                if 'detailed_analysis' in llm_insights:
                    print(f"   🤖 AI Insights: Available")
                else:
                    print(f"   🤖 AI Insights: {llm_insights.get('note', 'Not available')}")
                
                research_results['financial_analysis'] = financial_analysis
                research_results['pipeline_status']['financial_analysis'] = 'success'
            else:
                print(f"❌ Financial analysis error: {financial_analysis['error']}")
                research_results['pipeline_status']['financial_analysis'] = f"failed: {financial_analysis['error']}"
        else:
            print("❌ Could not retrieve stored structured data")
            research_results['pipeline_status']['financial_analysis'] = 'failed: no stored data'
        
    except Exception as e:
        print(f"❌ Financial analysis error: {e}")
        research_results['pipeline_status']['financial_analysis'] = f'failed: {e}'
    
    # ==================== STEP 5: MARKET INTELLIGENCE ====================
    print(f"\n📈 STEP 5: Running Market Intelligence Analysis (RAG + LLM)")
    print("-" * 50)
    
    try:
        market_intelligence = market_agent.analyze_market_intelligence(
            ticker=ticker,
            focus_areas=["sentiment", "news_impact", "risk_factors"]
        )
        
        if 'error' not in market_intelligence:
            print("✅ Market intelligence analysis completed")
            
            # Display key results
            market_intel = market_intelligence.get('market_intelligence', {})
            overall_assessment = market_intelligence.get('overall_assessment', {})
            
            print(f"   🔍 Focus areas analyzed: {len(market_intel)}")
            
            for area, analysis in market_intel.items():
                confidence = analysis.get('confidence', 0)
                source_count = analysis.get('source_count', 0)
                print(f"   📊 {area.title()}: {confidence:.1%} confidence ({source_count} sources)")
            
            if overall_assessment:
                print(f"   🎯 Overall Market Outlook: Available")
            else:
                print(f"   🎯 Overall Market Outlook: Basic assessment")
            
            research_results['market_intelligence'] = market_intelligence
            research_results['pipeline_status']['market_intelligence'] = 'success'
        else:
            print(f"❌ Market intelligence error: {market_intelligence['error']}")
            research_results['pipeline_status']['market_intelligence'] = f"failed: {market_intelligence['error']}"
        
    except Exception as e:
        print(f"❌ Market intelligence error: {e}")
        research_results['pipeline_status']['market_intelligence'] = f'failed: {e}'
    
    # ==================== STEP 6: COMBINED ASSESSMENT ====================
    print(f"\n🎯 STEP 6: Generating Combined Investment Assessment")
    print("-" * 50)
    
    try:
        combined_assessment = generate_combined_assessment(
            ticker, 
            research_results.get('financial_analysis', {}),
            research_results.get('market_intelligence', {})
        )
        
        research_results['combined_assessment'] = combined_assessment
        
        print("✅ Combined assessment generated")
        print(f"   📊 Investment Recommendation: {combined_assessment.get('investment_recommendation', 'N/A')}")
        print(f"   🎯 Confidence Level: {combined_assessment.get('overall_confidence', 'N/A')}")
        print(f"   ⚖️ Analysis Quality: {combined_assessment.get('analysis_quality', 'N/A')}")
        
        research_results['pipeline_status']['combined_assessment'] = 'success'
        
    except Exception as e:
        print(f"❌ Combined assessment error: {e}")
        research_results['pipeline_status']['combined_assessment'] = f'failed: {e}'
    
    # ==================== STEP 7: SAVE COMPLETE REPORT ====================
    print(f"\n💾 STEP 7: Saving Complete Investment Research Report")
    print("-" * 50)
    
    try:
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/{ticker}_complete_research_{timestamp}.json"
        
        # Save comprehensive report
        with open(report_filename, 'w') as f:
            json.dump(research_results, f, indent=2, default=str)
        
        print(f"✅ Complete research report saved: {report_filename}")
        
        # Generate executive summary
        generate_executive_summary(ticker, research_results)
        
        research_results['pipeline_status']['report_generation'] = 'success'
        research_results['report_filename'] = report_filename
        
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        research_results['pipeline_status']['report_generation'] = f'failed: {e}'
    
    # ==================== FINAL RESULTS ====================
    print(f"\n🎉 COMPLETE INVESTMENT RESEARCH FINISHED!")
    print("=" * 70)
    
    # Pipeline status summary
    success_count = sum(1 for status in research_results['pipeline_status'].values() if status == 'success')
    total_steps = len(research_results['pipeline_status'])
    
    print(f"📊 Pipeline Success Rate: {success_count}/{total_steps} steps completed successfully")
    print(f"🎯 Target Company: {research_results.get('company_name', ticker)}")
    print(f"📅 Analysis Date: {research_results['research_timestamp']}")
    
    # Show pipeline status
    print(f"\n📋 Pipeline Status:")
    for step, status in research_results['pipeline_status'].items():
        status_icon = "✅" if status == 'success' else "⚠️" if 'partial' in status else "❌"
        print(f"   {status_icon} {step.replace('_', ' ').title()}: {status}")
    
    return research_results


def generate_combined_assessment(ticker: str, financial_analysis: Dict, market_intelligence: Dict) -> Dict[str, Any]:
    """
    Generate combined investment assessment from both analyses
    """
    combined = {
        'ticker': ticker,
        'assessment_timestamp': datetime.now().isoformat(),
        'data_sources_used': []
    }
    
    # Extract key metrics from financial analysis
    financial_rating = 'unknown'
    financial_confidence = 0
    
    if financial_analysis:
        overall_assessment = financial_analysis.get('overall_assessment', {})
        financial_rating = overall_assessment.get('overall_rating', 'unknown')
        
        # Check if LLM analysis is available
        if 'llm_insights' in financial_analysis:
            financial_confidence = 0.8
            combined['data_sources_used'].append('financial_llm_analysis')
        else:
            financial_confidence = 0.6
            combined['data_sources_used'].append('financial_basic_analysis')
    
    # Extract key metrics from market intelligence
    market_outlook = 'unknown'
    market_confidence = 0
    
    if market_intelligence:
        overall_assessment = market_intelligence.get('overall_assessment', {})
        market_outlook = overall_assessment.get('overall_outlook', 'unknown')
        
        # Calculate average confidence from focus areas
        market_intel = market_intelligence.get('market_intelligence', {})
        if market_intel:
            confidences = [area.get('confidence', 0) for area in market_intel.values()]
            market_confidence = sum(confidences) / len(confidences) if confidences else 0
            combined['data_sources_used'].append('market_intelligence_rag')
    
    # Combine assessments
    positive_indicators = []
    negative_indicators = []
    
    # Financial indicators
    if 'positive' in financial_rating.lower():
        positive_indicators.append('Strong financial metrics')
    elif 'mixed' in financial_rating.lower():
        positive_indicators.append('Stable financial position')
    else:
        negative_indicators.append('Financial metrics need attention')
    
    # Market indicators
    if 'positive' in market_outlook.lower():
        positive_indicators.append('Positive market sentiment')
    elif 'neutral' in market_outlook.lower():
        positive_indicators.append('Neutral market conditions')
    else:
        negative_indicators.append('Market headwinds present')
    
    # Generate overall recommendation
    if len(positive_indicators) > len(negative_indicators):
        if len(positive_indicators) >= 2:
            investment_recommendation = "BUY - Strong fundamentals with positive market support"
        else:
            investment_recommendation = "HOLD - Positive indicators with some concerns"
    elif len(negative_indicators) > len(positive_indicators):
        investment_recommendation = "HOLD - Requires further analysis due to concerns"
    else:
        investment_recommendation = "HOLD - Mixed signals require careful monitoring"
    
    # Calculate overall confidence
    overall_confidence = (financial_confidence + market_confidence) / 2
    
    # Determine analysis quality
    if len(combined['data_sources_used']) >= 2 and overall_confidence > 0.7:
        analysis_quality = "High - Comprehensive multi-source analysis"
    elif overall_confidence > 0.5:
        analysis_quality = "Medium - Good coverage with some limitations"
    else:
        analysis_quality = "Basic - Limited data sources available"
    
    combined.update({
        'investment_recommendation': investment_recommendation,
        'overall_confidence': f"{overall_confidence:.1%}",
        'analysis_quality': analysis_quality,
        'positive_factors': positive_indicators,
        'risk_factors': negative_indicators,
        'financial_rating': financial_rating,
        'market_outlook': market_outlook[:100] + "..." if len(market_outlook) > 100 else market_outlook
    })
    
    return combined


def generate_executive_summary(ticker: str, research_results: Dict):
    """Generate and display executive summary"""
    
    print(f"\n📋 EXECUTIVE SUMMARY - {ticker}")
    print("=" * 50)
    
    # Company info
    financial_analysis = research_results.get('financial_analysis', {})
    overview = financial_analysis.get('company_overview', {})
    
    if overview:
        print(f"Company: {overview.get('company_name', ticker)}")
        print(f"Sector: {overview.get('sector', 'N/A')}")
        print(f"Current Price: ${overview.get('current_price', 0):,.2f}")
        print(f"Market Cap: ${overview.get('market_cap', 0):,.0f}")
    
    # Combined assessment
    combined = research_results.get('combined_assessment', {})
    if combined:
        print(f"\nInvestment Recommendation: {combined.get('investment_recommendation', 'N/A')}")
        print(f"Analysis Confidence: {combined.get('overall_confidence', 'N/A')}")
        print(f"Analysis Quality: {combined.get('analysis_quality', 'N/A')}")
        
        positive_factors = combined.get('positive_factors', [])
        risk_factors = combined.get('risk_factors', [])
        
        if positive_factors:
            print(f"\nPositive Factors:")
            for factor in positive_factors:
                print(f"  ✅ {factor}")
        
        if risk_factors:
            print(f"\nRisk Factors:")
            for factor in risk_factors:
                print(f"  ⚠️ {factor}")
    
    # Data sources
    data_sources = combined.get('data_sources_used', [])
    if data_sources:
        print(f"\nData Sources: {', '.join(data_sources)}")
    
    print(f"\nReport saved with complete details for further analysis.")


if __name__ == "__main__":
    # Run complete investment research
    from dotenv import load_dotenv
    load_dotenv()
    
    results = run_complete_investment_research("TSKA", "Tesla")
    
    if results:
        print(f"\n🚀 Phase 2 Complete! Your AI Investment Research Agent is working!")
    else:
        print(f"\n❌ Integration test failed. Check errors above.")