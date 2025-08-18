"""
Market Intelligence Agent - Analyzes unstructured data using RAG
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Any
import logging
from vector_manager import VectorDataManager

logger = logging.getLogger(__name__)

# LLM Integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

class MarketIntelligenceAgent:
    """
    AI Agent that analyzes market sentiment and news using RAG
    Combines vector search with LLM analysis for market intelligence
    """
    
    def __init__(self, use_llm: bool = True):
        self.agent_name = "Market Intelligence Agent"
        self.use_llm = use_llm and OPENAI_AVAILABLE
        
        # Initialize vector database manager
        self.vector_manager = VectorDataManager()
        
        # Initialize LLM client
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv('GITHUB_AI_API_KEY')
                    )
                logger.info("Market Intelligence Agent initialized with LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False
        
        if not self.use_llm:
            logger.info("Market Intelligence Agent running without LLM")
    
    def analyze_market_intelligence(self, ticker: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market intelligence for a ticker using RAG
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            focus_areas: Specific areas to analyze (e.g., ["earnings", "competition", "sentiment"])
            
        Returns:
            Dictionary with market intelligence analysis
        """
        try:
            logger.info(f"Starting market intelligence analysis for {ticker}")
            
            if focus_areas is None:
                focus_areas = ["sentiment", "news_impact", "market_trends", "risk_factors"]
            
            analysis_results = {
                'ticker': ticker,
                'analysis_timestamp': '2024-01-15T10:30:00Z', #TODO Update the timestamp
                'focus_areas': focus_areas,
                'market_intelligence': {}
            }
            
            # Analyze each focus area
            for area in focus_areas:
                logger.info(f"Analyzing {area} for {ticker}")
                area_analysis = self._analyze_focus_area(ticker, area)
                analysis_results['market_intelligence'][area] = area_analysis
            
            # Generate overall market assessment
            if self.use_llm:
                overall_assessment = self._generate_overall_market_assessment(ticker, analysis_results)
                analysis_results['overall_assessment'] = overall_assessment
            else:
                analysis_results['overall_assessment'] = self._generate_basic_assessment(analysis_results)
            
            logger.info(f"Market intelligence analysis completed for {ticker}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in market intelligence analysis for {ticker}: {e}")
            return {'error': str(e)}
    
    def _analyze_focus_area(self, ticker: str, focus_area: str) -> Dict[str, Any]:
        """Analyze a specific focus area using RAG"""
        
        # Define search queries for each focus area
        search_queries = {
            'sentiment': f"{ticker} investor sentiment bullish bearish opinion",
            'news_impact': f"{ticker} breaking news market reaction price movement",
            'market_trends': f"{ticker} industry trends sector rotation market conditions",
            'risk_factors': f"{ticker} risks challenges concerns threats competition",
            'earnings': f"{ticker} earnings results financial performance revenue",
            'competition': f"{ticker} competitors market share competitive position",
            'growth': f"{ticker} growth prospects expansion opportunities future"
        }
        
        query = search_queries.get(focus_area, f"{ticker} {focus_area}")
        
        # Search vector database for relevant documents
        relevant_docs = self.vector_manager.search_documents(
            query=query,
            ticker=ticker,
            limit=5
        )
        
        if not relevant_docs:
            return {
                'analysis': f"No relevant documents found for {focus_area}",
                'confidence': 0,
                'source_count': 0
            }
        
        # Analyze with LLM if available
        if self.use_llm and relevant_docs:
            llm_analysis = self._analyze_documents_with_llm(ticker, focus_area, relevant_docs)
            return llm_analysis
        else:
            # Basic analysis without LLM
            return self._basic_document_analysis(focus_area, relevant_docs)
    
    def _analyze_documents_with_llm(self, ticker: str, focus_area: str, documents: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze retrieved documents"""
        try:
            # Prepare document content for LLM
            doc_content = ""
            source_info = []
            
            for i, doc in enumerate(documents[:3], 1):  # Use top 3 most relevant //TODO WHy only 3, can't we introduce all?
                content = doc['document'][:500]  # Limit content length
                source = doc['metadata'].get('source', 'Unknown')
                doc_type = doc['metadata'].get('type', 'unknown')
                relevance = doc.get('relevance', 'Unknown')
                
                doc_content += f"\nDocument {i} [{doc_type.upper()}] (Relevance: {relevance}):\n{content}\n"
                source_info.append({
                    'type': doc_type,
                    'source': source,
                    'relevance': relevance
                })
            
            # Create focused prompt for the specific area
            prompt = f"""
Analyze the following documents about {ticker} focusing on {focus_area}:

{doc_content}

Provide analysis for {focus_area}:
1. Key Insights (2-3 main points)
2. Market Implications (how this affects the stock)
3. Confidence Level (High/Medium/Low based on source quality)
4. Risk Assessment (potential concerns)

Keep response concise and actionable. Focus specifically on {focus_area}.
"""

            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a market intelligence analyst specializing in {focus_area} analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'analysis': analysis_text,
                'confidence': self._calculate_confidence(documents),
                'source_count': len(documents),
                'sources_used': source_info,
                'model_used': 'gpt-4o'
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis for {focus_area}: {e}")
            return self._basic_document_analysis(focus_area, documents)
    
    def _basic_document_analysis(self, focus_area: str, documents: List[Dict]) -> Dict[str, Any]:
        """Basic analysis without LLM"""
        
        # Count positive/negative indicators
        positive_words = ['strong', 'beat', 'exceed', 'growth', 'positive', 'bullish', 'up', 'gain']
        negative_words = ['weak', 'miss', 'decline', 'negative', 'bearish', 'down', 'loss', 'concern']
        
        positive_count = 0
        negative_count = 0
        total_docs = len(documents)
        
        key_themes = []
        
        for doc in documents:
            content = doc['document'].lower()
            title = doc['metadata'].get('title', '').lower()
            
            # Count sentiment indicators
            doc_positive = sum(1 for word in positive_words if word in content or word in title)
            doc_negative = sum(1 for word in negative_words if word in content or word in title)
            
            if doc_positive > doc_negative:
                positive_count += 1
            elif doc_negative > doc_positive:
                negative_count += 1
            
            # Extract key themes from titles
            if doc['metadata'].get('title'):
                key_themes.append(doc['metadata']['title'][:50])
        
        # Determine overall sentiment for this focus area
        if positive_count > negative_count:
            sentiment = 'Positive'
        elif negative_count > positive_count:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        analysis_text = f"""
{focus_area.title()} Analysis:
- Overall Sentiment: {sentiment}
- Documents Analyzed: {total_docs}
- Positive Indicators: {positive_count}
- Negative Indicators: {negative_count}
- Key Themes: {', '.join(key_themes[:2])}
"""
        
        return {
            'analysis': analysis_text,
            'confidence': min(0.7, total_docs / 5),  # Max 70% confidence for basic analysis
            'source_count': total_docs,
            'sentiment': sentiment.lower()
        }
    
    def _calculate_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence based on document quality and quantity"""
        if not documents:
            return 0.0
        
        # Base confidence on number of documents
        doc_count_score = min(1.0, len(documents) / 5)
        
        # Adjust for source quality
        quality_score = 0
        for doc in documents:
            relevance = doc.get('relevance', 'Low')
            if relevance == 'Very High':
                quality_score += 0.3
            elif relevance == 'High':
                quality_score += 0.2
            elif relevance == 'Medium':
                quality_score += 0.1
        
        quality_score = min(1.0, quality_score)
        
        # Combined confidence
        confidence = (doc_count_score * 0.6) + (quality_score * 0.4)
        return round(confidence, 3)
    
    def _generate_overall_market_assessment(self, ticker: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market assessment using LLM"""
        try:
            # Summarize all focus area results
            summary_content = f"Market Intelligence Summary for {ticker}:\n\n"
            
            for area, analysis in analysis_results['market_intelligence'].items():
                summary_content += f"{area.upper()}:\n{analysis.get('analysis', 'No analysis available')}\n\n"
            
            prompt = f"""
Based on the market intelligence analysis below, provide an overall market assessment for {ticker}:

{summary_content}

Provide:
1. Overall Market Outlook (Positive/Negative/Neutral)
2. Key Market Drivers (top 2-3 factors affecting the stock)
3. Main Risks (primary concerns for investors)
4. Investment Implications (what this means for potential investors)
5. Confidence Level (High/Medium/Low)

Keep response concise and actionable for investment decisions.
"""

            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior market analyst providing overall market assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            return {
                'overall_outlook': response.choices[0].message.content,
                'analysis_method': 'llm_comprehensive',
                'confidence': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return self._generate_basic_assessment(analysis_results)
    
    def _generate_basic_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic assessment without LLM"""
        
        # Count positive/negative areas
        positive_areas = 0
        negative_areas = 0
        
        for area, analysis in analysis_results['market_intelligence'].items():
            sentiment = analysis.get('sentiment', 'neutral')
            if sentiment == 'positive':
                positive_areas += 1
            elif sentiment == 'negative':
                negative_areas += 1
        
        if positive_areas > negative_areas:
            outlook = 'Positive market indicators outweigh negative factors'
        elif negative_areas > positive_areas:
            outlook = 'Negative market factors require attention'
        else:
            outlook = 'Mixed market signals require careful analysis'
        
        return {
            'overall_outlook': outlook,
            'analysis_method': 'basic_sentiment_aggregation',
            'confidence': 'medium'
        }


# Test the Market Intelligence Agent
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🧪 Testing Market Intelligence Agent...")
    
    # Initialize agent
    agent = MarketIntelligenceAgent(use_llm=True)
    
    # Run analysis
    analysis = agent.analyze_market_intelligence(
        ticker="AAPL",
        focus_areas=["sentiment", "news_impact", "risk_factors"]
    )
    
    # Display results
    print(f"\n📊 Market Intelligence Analysis for {analysis.get('ticker', 'Unknown')}")
    print("=" * 60)
    
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
    else:
        market_intel = analysis.get('market_intelligence', {})
        
        for area, area_analysis in market_intel.items():
            print(f"\n🔍 {area.upper()} ANALYSIS:")
            print(f"Confidence: {area_analysis.get('confidence', 0):.1%}")
            print(f"Sources: {area_analysis.get('source_count', 0)} documents")
            
            analysis_text = area_analysis.get('analysis', 'No analysis available')
            print(f"Analysis: {analysis_text[:200]}...")
        
        # Overall assessment
        overall = analysis.get('overall_assessment', {})
        if overall:
            print(f"\n🎯 OVERALL MARKET OUTLOOK:")
            print(overall.get('overall_outlook', 'No overall assessment available')[:300] + "...")
    
    print("\n🎉 Market Intelligence Agent test completed!")