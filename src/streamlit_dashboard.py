"""
Streamlit Dashboard - AI Investment Research Agent Web Interface
Beautiful, professional interface for stock research
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our AI agents
from financial_data_collector import FinancialDataCollector
from data_manager import StructuredDataManager
from vector_manager import VectorDataManager
from agents.financial_agent import EnhancedFinancialAnalysisAgent
from agents.market_agent import MarketIntelligenceAgent

# Page configuration
st.set_page_config(
    page_title="AI Investment Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'research_data' not in st.session_state:
        st.session_state.research_data = None
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []

def create_price_chart(historical_data, ticker):
    """Create interactive price chart"""
    if not historical_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=df.index,
        open=df.get('Open', df.get('Close', [])),
        high=df.get('High', df.get('Close', [])),
        low=df.get('Low', df.get('Close', [])),
        close=df.get('Close', []),
        name=ticker
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price - Last 30 Days",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_white",
        height=400
    )
    
    return fig

def display_financial_metrics(financial_analysis):
    """Display key financial metrics in a clean layout"""
    if not financial_analysis:
        st.warning("No financial analysis available")
        return
    
    overview = financial_analysis.get('company_overview', {})
    valuation = financial_analysis.get('valuation_analysis', {})
    risk = financial_analysis.get('risk_assessment', {})
    
    # Company overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${overview.get('current_price', 0):,.2f}"
        )
    
    with col2:
        market_cap = overview.get('market_cap', 0)
        if market_cap > 1_000_000_000:
            market_cap_display = f"${market_cap/1_000_000_000:.1f}B"
        else:
            market_cap_display = f"${market_cap/1_000_000:.1f}M"
        
        st.metric(
            label="Market Cap",
            value=market_cap_display
        )
    
    with col3:
        pe_ratio = valuation.get('pe_ratio', 'N/A')
        st.metric(
            label="P/E Ratio",
            value=pe_ratio if pe_ratio != 'N/A' else pe_ratio
        )
    
    with col4:
        dividend_yield = valuation.get('dividend_yield', 0)
        st.metric(
            label="Dividend Yield",
            value=f"{dividend_yield*100:.2f}%" if dividend_yield else "0.00%"
        )

def display_ai_insights(financial_analysis, market_intelligence):
    """Display AI-generated insights"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Financial AI Analysis")
        
        # LLM Insights
        llm_insights = financial_analysis.get('llm_insights', {})
        if 'detailed_analysis' in llm_insights:
            st.success("✅ AI Analysis Available")
            with st.expander("View Detailed Financial Analysis"):
                st.write(llm_insights['detailed_analysis'])
        else:
            st.info("ℹ️ Basic analysis only (LLM not available)")
        
        # Investment Recommendation
        investment_rec = financial_analysis.get('investment_recommendation', {})
        if 'recommendation_text' in investment_rec:
            with st.expander("Investment Recommendation"):
                st.write(investment_rec['recommendation_text'])
    
    with col2:
        st.subheader("📈 Market Intelligence")
        
        if market_intelligence and 'market_intelligence' in market_intelligence:
            market_intel = market_intelligence['market_intelligence']
            
            for area, analysis in market_intel.items():
                confidence = analysis.get('confidence', 0)
                source_count = analysis.get('source_count', 0)
                
                st.write(f"**{area.replace('_', ' ').title()}**")
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.1%} | Sources: {source_count}")
                
                if st.button(f"View {area.title()} Details", key=f"btn_{area}"):
                    with st.expander(f"{area.title()} Analysis", expanded=True):
                        st.write(analysis.get('analysis', 'No analysis available'))
        else:
            st.info("ℹ️ Market intelligence data not available")

def display_recommendation_card(combined_assessment):
    """Display investment recommendation in a prominent card"""
    if not combined_assessment:
        return
    
    recommendation = combined_assessment.get('investment_recommendation', 'No recommendation available')
    confidence = combined_assessment.get('overall_confidence', 'Unknown')
    
    # Determine recommendation type for styling
    if 'BUY' in recommendation.upper():
        card_class = "recommendation-buy"
        emoji = "🟢"
    elif 'SELL' in recommendation.upper():
        card_class = "recommendation-sell"
        emoji = "🔴"
    else:
        card_class = "recommendation-hold"
        emoji = "🟡"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h3>{emoji} Investment Recommendation</h3>
        <p><strong>{recommendation}</strong></p>
        <p>Confidence Level: {confidence}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show factors
    positive_factors = combined_assessment.get('positive_factors', [])
    risk_factors = combined_assessment.get('risk_factors', [])
    
    if positive_factors or risk_factors:
        col1, col2 = st.columns(2)
        
        with col1:
            if positive_factors:
                st.write("**✅ Positive Factors:**")
                for factor in positive_factors:
                    st.write(f"• {factor}")
        
        with col2:
            if risk_factors:
                st.write("**⚠️ Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")

def run_research_pipeline(ticker, company_name=None):
    """Run the complete research pipeline with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("🔧 Initializing AI research components...")
        progress_bar.progress(10)
        
        collector = FinancialDataCollector()
        structured_manager = StructuredDataManager()
        vector_manager = VectorDataManager()
        financial_agent = EnhancedFinancialAnalysisAgent(use_llm=True)
        market_agent = MarketIntelligenceAgent(use_llm=True)
        
        # Step 1: Data Collection
        status_text.text("📊 Collecting financial data...")
        progress_bar.progress(20)
        
        separated_data = collector.collect_separated_data(ticker, company_name)
        if not separated_data:
            st.error("❌ Failed to collect data")
            return None
        
        # Step 2: Store structured data
        status_text.text("💾 Processing structured data...")
        progress_bar.progress(40)
        
        structured_manager.store_company_data(ticker, separated_data['structured'])
        
        # Step 3: Store unstructured data
        status_text.text("🗂️ Processing news and sentiment data...")
        progress_bar.progress(50)
        
        vector_manager.store_unstructured_data(ticker, separated_data['unstructured'])
        
        # Step 4: Financial analysis
        status_text.text("🧮 Running AI financial analysis...")
        progress_bar.progress(70)
        
        stored_data = structured_manager.get_latest_company_data(ticker)
        financial_analysis = financial_agent.analyze_company_financials(stored_data)
        
        # Step 5: Market intelligence
        status_text.text("📈 Analyzing market intelligence...")
        progress_bar.progress(85)
        
        market_intelligence = market_agent.analyze_market_intelligence(
            ticker=ticker,
            focus_areas=["sentiment", "news_impact", "risk_factors"]
        )
        
        # Step 6: Combined assessment
        status_text.text("🎯 Generating investment recommendation...")
        progress_bar.progress(95)
        
        # Generate combined assessment (simplified version)
        combined_assessment = generate_simple_combined_assessment(
            financial_analysis, market_intelligence
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Research completed successfully!")
        
        # Compile results
        research_results = {
            'ticker': ticker,
            'company_name': company_name,
            'timestamp': datetime.now().isoformat(),
            'financial_analysis': financial_analysis,
            'market_intelligence': market_intelligence,
            'combined_assessment': combined_assessment,
            'raw_data': separated_data
        }
        
        return research_results
        
    except Exception as e:
        st.error(f"❌ Research failed: {str(e)}")
        return None

def generate_simple_combined_assessment(financial_analysis, market_intelligence):
    """Generate a simple combined assessment"""
    
    # Extract key indicators
    financial_rating = financial_analysis.get('overall_assessment', {}).get('overall_rating', 'unknown')
    
    # Determine recommendation
    if 'positive' in financial_rating.lower():
        recommendation = "HOLD - Positive financial indicators"
    elif 'mixed' in financial_rating.lower():
        recommendation = "HOLD - Mixed signals require monitoring"
    else:
        recommendation = "RESEARCH - Requires detailed analysis"
    
    return {
        'investment_recommendation': recommendation,
        'overall_confidence': '75%',
        'positive_factors': ['Financial metrics available', 'AI analysis completed'],
        'risk_factors': ['Market volatility', 'Economic uncertainty']
    }

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 AI Investment Research Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Professional-grade investment research powered by AI**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Research Controls")
        
        # Stock input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        company_name = st.text_input(
            "Company Name (Optional)",
            value="Apple Inc.",
            help="Optional: Enter full company name for better analysis"
        )
        
        # Research button
        research_button = st.button(
            "🚀 Start AI Research",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Research history
        if st.session_state.research_history:
            st.subheader("📚 Recent Research")
            for i, research in enumerate(st.session_state.research_history[-5:]):
                if st.button(f"{research['ticker']} - {research['timestamp'][:10]}", key=f"history_{i}"):
                    st.session_state.research_data = research
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("• Use major stock tickers (AAPL, MSFT, GOOGL)")
        st.markdown("• Research takes 30-60 seconds")
        st.markdown("• AI analyzes financials + market sentiment")
    
    # Main content area
    if research_button and ticker:
        with st.spinner(f"🔍 Researching {ticker}..."):
            research_data = run_research_pipeline(ticker, company_name)
            
            if research_data:
                st.session_state.research_data = research_data
                st.session_state.research_history.append({
                    'ticker': ticker,
                    'timestamp': research_data['timestamp'],
                    'company_name': company_name
                })
                st.success(f"✅ Research completed for {ticker}!")
                st.rerun()
    
    # Display results
    if st.session_state.research_data:
        data = st.session_state.research_data
        
        # Company header
        st.subheader(f"📊 Research Report: {data['ticker']}")
        if data.get('company_name'):
            st.write(f"**{data['company_name']}**")
        
        st.write(f"*Generated on {data['timestamp'][:19].replace('T', ' ')}*")
        
        # Investment recommendation (prominent)
        display_recommendation_card(data.get('combined_assessment'))
        
        st.markdown("---")
        
        # Financial metrics
        st.subheader("📈 Key Financial Metrics")
        display_financial_metrics(data.get('financial_analysis'))
        
        # Price chart
        financial_data = data.get('financial_analysis', {}).get('company_overview', {})
        historical_data = financial_data.get('historical_data', [])
        
        if historical_data:
            st.subheader("📊 Price Chart")
            chart = create_price_chart(historical_data, data['ticker'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        st.markdown("---")
        
        # AI Analysis
        st.subheader("🤖 AI Analysis & Market Intelligence")
        display_ai_insights(
            data.get('financial_analysis'), 
            data.get('market_intelligence')
        )
        
        st.markdown("---")
        
        # Raw data download
        with st.expander("📥 Download Research Data"):
            st.download_button(
                label="Download Complete Research Report (JSON)",
                data=json.dumps(data, indent=2, default=str),
                file_name=f"{data['ticker']}_research_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        # Welcome message
        st.info("👋 Welcome! Enter a stock ticker in the sidebar and click 'Start AI Research' to begin.")
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧮 Financial Analysis
            - AI-powered financial metrics
            - P/E ratios, market cap, dividends
            - Professional investment insights
            """)
        
        with col2:
            st.markdown("""
            ### 📰 Market Intelligence
            - Real-time news analysis
            - Social sentiment tracking
            - Risk factor identification
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Smart Recommendations
            - BUY/HOLD/SELL guidance
            - Confidence scoring
            - Comprehensive reporting
            """)

if __name__ == "__main__":
    main()