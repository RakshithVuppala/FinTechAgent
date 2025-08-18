"""
Enhanced Financial Analysis Agent - Analyzes structured financial data with LLM insights
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

# LLM Integration
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")


class EnhancedFinancialAnalysisAgent:
    """
    AI Agent that analyzes structured financial data with LLM insights
    Combines quantitative analysis with qualitative LLM interpretation
    """

    def __init__(self, use_llm: bool = True):
        self.agent_name = "Enhanced Financial Analysis Agent"
        self.use_llm = use_llm and OPENAI_AVAILABLE

        # Initialize LLM client
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv("GITHUB_AI_API_KEY"),
                )
                logger.info("LLM client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.use_llm = False

        if not self.use_llm:
            logger.info("Running in basic mode without LLM")

    def analyze_company_financials(
        self, structured_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze structured financial data with LLM enhancement
        """
        try:
            ticker = structured_data.get("ticker", "Unknown")
            logger.info(f"Starting enhanced financial analysis for {ticker}")

            # Extract financial metrics
            financial_metrics = structured_data.get("financial_metrics", {})

            # Step 1: Basic quantitative analysis
            basic_analysis = {
                "company_overview": self._analyze_company_overview(financial_metrics),
                "valuation_analysis": self._analyze_valuation(financial_metrics),
                "risk_assessment": self._assess_risk(financial_metrics),
                "performance_metrics": self._calculate_performance_metrics(
                    financial_metrics
                ),
                "market_position": self._analyze_market_position(financial_metrics),
            }

            # Step 2: LLM enhancement (if available)
            if self.use_llm:
                llm_insights = self._generate_llm_insights(
                    ticker, basic_analysis, financial_metrics
                )
                basic_analysis["llm_insights"] = llm_insights
                basic_analysis["investment_recommendation"] = (
                    self._generate_investment_recommendation(
                        ticker, basic_analysis, financial_metrics
                    )
                )
            else:
                basic_analysis["llm_insights"] = {
                    "note": "LLM not available - basic analysis only"
                }
                basic_analysis["investment_recommendation"] = (
                    self._generate_basic_recommendation(basic_analysis)
                )

            # Step 3: Generate overall assessment
            basic_analysis["overall_assessment"] = self._generate_overall_assessment(
                basic_analysis
            )

            logger.info(f"Enhanced financial analysis completed for {ticker}")
            return basic_analysis

        except Exception as e:
            logger.error(f"Error in enhanced financial analysis: {e}")
            return {"error": str(e)}

    def _generate_llm_insights(
        self, ticker: str, analysis: Dict[str, Any], raw_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to generate intelligent insights from the financial data"""
        try:
            # Prepare data for LLM
            financial_summary = self._prepare_financial_summary_for_llm(
                ticker, analysis, raw_metrics
            )

            # LLM Prompt for financial analysis
            prompt = f"""
You are a professional financial analyst. Analyze the following financial data for {ticker} and provide insights:

COMPANY DATA:
{financial_summary}

Please provide:
1. Key Financial Strengths (2-3 points)
2. Key Financial Concerns (2-3 points)  
3. Valuation Assessment (fair/overvalued/undervalued and why)
4. Risk Factors (main risks to consider)
5. Investment Thesis (2-3 sentence summary)

Keep your response concise and professional. Focus on actionable insights for investment decisions.
"""

            # Make LLM call
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",  # GPT-4o model for enhanced analysis
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst providing investment research insights.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            llm_analysis = response.choices[0].message.content

            # Parse LLM response into structured format
            return {
                "detailed_analysis": llm_analysis,
                "analysis_timestamp": "2024-01-01",  # You'd use real timestamp
                "model_used": "gpt-4o",
                "analysis_type": "financial_fundamentals",
            }

        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return {"error": f"LLM analysis failed: {str(e)}"}

    def _generate_investment_recommendation(
        self, ticker: str, analysis: Dict[str, Any], raw_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment recommendation using LLM"""
        try:
            # Prepare recommendation prompt
            company_name = raw_metrics.get("company_name", ticker)
            current_price = raw_metrics.get("current_price", 0)
            pe_ratio = raw_metrics.get("pe_ratio", 0)
            market_cap = raw_metrics.get("market_cap", 0)

            prompt = f"""
As a financial analyst, provide an investment recommendation for {company_name} ({ticker}):

Current Price: ${current_price}
P/E Ratio: {pe_ratio}
Market Cap: ${market_cap:,.0f}
Sector: {raw_metrics.get("sector", "N/A")}

Based on the financial metrics, provide:
1. Investment Recommendation: BUY/HOLD/SELL
2. Price Target (12-month): Specific price with justification
3. Investment Timeframe: Short/Medium/Long term
4. Key Catalysts: What could drive the stock higher
5. Key Risks: What could hurt the investment

Be specific and actionable. Limit to 300 words.
"""

            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior investment analyst providing actionable investment recommendations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )

            recommendation = response.choices[0].message.content

            return {
                "recommendation_text": recommendation,
                "generated_at": "2024-01-01",  # Real timestamp
                "confidence_level": "High",  # Could be calculated
                "model_used": "gpt-4o",
            }

        except Exception as e:
            logger.error(f"Error generating investment recommendation: {e}")
            return {"error": f"Recommendation generation failed: {str(e)}"}

    def _prepare_financial_summary_for_llm(
        self, ticker: str, analysis: Dict[str, Any], raw_metrics: Dict[str, Any]
    ) -> str:
        """Prepare a clean summary of financial data for LLM processing"""

        company_overview = analysis.get("company_overview", {})
        valuation = analysis.get("valuation_analysis", {})
        risk = analysis.get("risk_assessment", {})

        summary = f"""
Company: {company_overview.get("company_name", "N/A")} ({ticker})
Sector: {company_overview.get("sector", "N/A")}
Industry: {company_overview.get("industry", "N/A")}

VALUATION METRICS:
- Current Price: ${company_overview.get("current_price", 0):,.2f}
- Market Cap: ${company_overview.get("market_cap", 0):,.0f} ({company_overview.get("market_cap_category", "N/A")})
- P/E Ratio: {valuation.get("pe_ratio", "N/A")}
- Dividend Yield: {valuation.get("dividend_yield", 0) * 100:.2f}%

RISK METRICS:
- Beta: {risk.get("beta", "N/A")} ({risk.get("beta_assessment", "N/A")})
- 52-Week Range: ${raw_metrics.get("fifty_two_week_low", 0):.2f} - ${raw_metrics.get("fifty_two_week_high", 0):.2f}
- Current Position: {risk.get("price_assessment", "N/A")}

ASSESSMENTS:
- P/E Assessment: {valuation.get("pe_assessment", "N/A")}
- Dividend Assessment: {valuation.get("dividend_assessment", "N/A")}
- Market Position: {analysis.get("market_position", {}).get("relative_size", "N/A")}
"""
        return summary

    # Keep all the existing basic analysis methods
    def _analyze_company_overview(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic company information"""
        return {
            "company_name": metrics.get("company_name", "N/A"),
            "sector": metrics.get("sector", "N/A"),
            "industry": metrics.get("industry", "N/A"),
            "current_price": metrics.get("current_price", 0),
            "market_cap": metrics.get("market_cap", 0),
            "market_cap_category": self._categorize_market_cap(
                metrics.get("market_cap", 0)
            ),
        }

    def _analyze_valuation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics"""
        pe_ratio = metrics.get("pe_ratio", 0)

        if pe_ratio == 0 or pe_ratio is None:
            pe_assessment = "No P/E ratio available"
        elif pe_ratio < 15:
            pe_assessment = "Low P/E - Potentially undervalued"
        elif pe_ratio < 25:
            pe_assessment = "Moderate P/E - Reasonably valued"
        elif pe_ratio < 40:
            pe_assessment = "High P/E - Growth expectations"
        else:
            pe_assessment = "Very high P/E - High growth or overvaluation risk"

        return {
            "pe_ratio": pe_ratio,
            "pe_assessment": pe_assessment,
            "dividend_yield": metrics.get("dividend_yield", 0),
            "dividend_assessment": self._assess_dividend(
                metrics.get("dividend_yield", 0)
            ),
        }

    def _assess_risk(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors"""
        beta = metrics.get("beta", 1.0)

        if beta < 0.8:
            beta_assessment = "Low volatility - More stable than market"
        elif beta < 1.2:
            beta_assessment = "Moderate volatility - Similar to market"
        else:
            beta_assessment = "High volatility - More volatile than market"

        current_price = metrics.get("current_price", 0)
        week_52_high = metrics.get("fifty_two_week_high", 0)
        week_52_low = metrics.get("fifty_two_week_low", 0)

        if week_52_high and week_52_low and current_price:
            price_position = (current_price - week_52_low) / (
                week_52_high - week_52_low
            )
            if price_position > 0.8:
                price_assessment = "Near 52-week high - Momentum or overvaluation risk"
            elif price_position < 0.2:
                price_assessment = (
                    "Near 52-week low - Potential value or fundamental issues"
                )
            else:
                price_assessment = "Mid-range positioning"
        else:
            price_assessment = "Insufficient price data"

        return {
            "beta": beta,
            "beta_assessment": beta_assessment,
            "price_position_52week": price_position
            if "price_position" in locals()
            else 0,
            "price_assessment": price_assessment,
        }

    def _calculate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional performance metrics"""
        return {
            "has_historical_data": bool(metrics.get("historical_data")),
            "data_points_available": len(metrics.get("historical_data", [])),
            "news_articles_count": len(metrics.get("news", [])),
        }

    def _analyze_market_position(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market position"""
        market_cap = metrics.get("market_cap", 0)
        sector = metrics.get("sector", "Unknown")

        return {
            "market_cap_category": self._categorize_market_cap(market_cap),
            "sector": sector,
            "relative_size": self._assess_relative_size(market_cap, sector),
        }

    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize company by market cap"""
        if market_cap == 0:
            return "Unknown"
        elif market_cap < 2_000_000_000:
            return "Small Cap"
        elif market_cap < 10_000_000_000:
            return "Mid Cap"
        elif market_cap < 200_000_000_000:
            return "Large Cap"
        else:
            return "Mega Cap"

    def _assess_dividend(self, dividend_yield: float) -> str:
        """Assess dividend yield"""
        if dividend_yield == 0:
            return "No dividend - Growth focus"
        elif dividend_yield < 0.02:
            return "Low dividend yield"
        elif dividend_yield < 0.04:
            return "Moderate dividend yield"
        elif dividend_yield < 0.06:
            return "High dividend yield"
        else:
            return "Very high dividend yield - Verify sustainability"

    def _assess_relative_size(self, market_cap: float, sector: str) -> str:
        """Assess relative size within sector"""
        cap_category = self._categorize_market_cap(market_cap)

        if cap_category == "Mega Cap":
            return f"Dominant player in {sector}"
        elif cap_category == "Large Cap":
            return f"Major player in {sector}"
        elif cap_category == "Mid Cap":
            return f"Established player in {sector}"
        else:
            return f"Smaller player in {sector}"

    def _generate_basic_recommendation(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate basic recommendation without LLM"""
        overall_assessment = analysis.get("overall_assessment", {})
        rating = overall_assessment.get("overall_rating", "Mixed indicators")

        if "Positive" in rating:
            recommendation = "HOLD - Positive indicators suggest stable investment"
        elif "Mixed" in rating:
            recommendation = "HOLD - Mixed signals require further analysis"
        else:
            recommendation = "RESEARCH - Requires detailed analysis before investment"

        return {
            "recommendation_text": recommendation,
            "generated_at": "2024-01-01",
            "confidence_level": "Medium",
            "model_used": "basic_rules",
        }

    def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall investment assessment"""
        company_overview = analysis.get("company_overview", {})
        valuation = analysis.get("valuation_analysis", {})
        risk = analysis.get("risk_assessment", {})

        score = 0
        factors = []

        # PE Ratio scoring
        pe_ratio = valuation.get("pe_ratio", 0)
        if 15 <= pe_ratio <= 25:
            score += 1
            factors.append("Reasonable valuation")
        elif pe_ratio > 0:
            factors.append("Check valuation carefully")

        # Beta scoring
        beta = risk.get("beta", 1.0)
        if 0.8 <= beta <= 1.2:
            score += 1
            factors.append("Moderate risk profile")

        # Market cap scoring
        market_cap = company_overview.get("market_cap", 0)
        if market_cap > 10_000_000_000:
            score += 1
            factors.append("Established company size")

        # Overall assessment
        if score >= 2:
            overall_rating = "Positive indicators"
        elif score == 1:
            overall_rating = "Mixed indicators"
        else:
            overall_rating = "Requires careful analysis"

        return {
            "overall_rating": overall_rating,
            "score": score,
            "max_score": 3,
            "key_factors": factors,
        }


# Test the enhanced agent
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 Testing Enhanced Financial Analysis Agent...")

    # Test data
    test_structured_data = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "financial_metrics": {
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "current_price": 150.00,
            "market_cap": 2500000000000,
            "pe_ratio": 25.5,
            "dividend_yield": 0.005,
            "beta": 1.2,
            "fifty_two_week_high": 180.00,
            "fifty_two_week_low": 120.00,
            "historical_data": [{"date": "2024-01-01", "price": 145}] * 30,
            "news": [{"title": "Apple earnings beat expectations"}] * 5,
        },
    }

    # Test with LLM
    agent = EnhancedFinancialAnalysisAgent(use_llm=True)
    analysis = agent.analyze_company_financials(test_structured_data)

    # Display results
    print(
        f"✅ Analysis completed for: {analysis.get('company_overview', {}).get('company_name', 'N/A')}"
    )
    print(
        f"📊 Overall rating: {analysis.get('overall_assessment', {}).get('overall_rating', 'N/A')}"
    )

    # Show LLM insights if available
    llm_insights = analysis.get("llm_insights", {})
    if "detailed_analysis" in llm_insights:
        print(f"\n🤖 LLM Analysis Preview:")
        print(llm_insights["detailed_analysis"][:200] + "...")
    else:
        print(f"\n⚠️ LLM Status: {llm_insights.get('note', 'Not available')}")

    # Show investment recommendation
    investment_rec = analysis.get("investment_recommendation", {})
    if "recommendation_text" in investment_rec:
        print(f"\n💡 Investment Recommendation Preview:")
        print(investment_rec["recommendation_text"][:150] + "...")

    print("\n🎉 Enhanced Financial Analysis Agent test completed!")
