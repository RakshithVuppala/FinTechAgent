import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import traceback

# Import system modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import your existing components
from financial_data_collector import FinancialDataCollector
from data_manager import StructuredDataManager
from vector_manager import VectorDataManager
from agents.financial_agent import EnhancedFinancialAnalysisAgent
from agents.market_agent import MarketIntelligenceAgent


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class TaskResult:
    task_name: str
    status: TaskStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = datetime.now()


class InvestmentResearchOrchestrator:
    """
    Master Orchestrator Agent for Investment Research

    Coordinates all agents and data flows to provide comprehensive
    investment analysis with intelligent caching, error handling,
    and parallel processing.
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_duration_minutes: int = 60,
        max_retries: int = 3,
        parallel_execution: bool = True,
        use_llm: bool = True,
    ):
        # Initialize all components
        self.data_collector = FinancialDataCollector()
        self.structured_manager = StructuredDataManager()
        self.vector_manager = VectorDataManager()
        self.financial_agent = EnhancedFinancialAnalysisAgent(use_llm=use_llm)
        self.market_agent = MarketIntelligenceAgent(use_llm=use_llm)

        # Configuration
        self.use_cache = use_cache
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.max_retries = max_retries
        self.parallel_execution = parallel_execution

        # Internal state
        self.task_results: Dict[str, TaskResult] = {}
        self.execution_history: List[Dict] = []
        self.cache: Dict[str, Dict] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Progress tracking
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback function for real-time progress updates"""
        self.progress_callback = callback

    def _update_progress(self, message: str, percentage: float):
        """Update progress and call callback if set"""
        self.logger.info(f"Progress: {percentage:.1f}% - {message}")
        if self.progress_callback:
            self.progress_callback(message, percentage)

    def _is_cache_valid(self, ticker: str, task_name: str) -> bool:
        """Check if cached data is still valid"""
        if not self.use_cache:
            return False

        cache_key = f"{ticker}_{task_name}"
        if cache_key not in self.cache:
            return False

        cached_time = self.cache[cache_key].get("timestamp")
        if not cached_time:
            return False

        return datetime.now() - cached_time < self.cache_duration

    def _get_from_cache(self, ticker: str, task_name: str) -> Optional[Any]:
        """Retrieve data from cache if valid"""
        if self._is_cache_valid(ticker, task_name):
            cache_key = f"{ticker}_{task_name}"
            return self.cache[cache_key]["data"]
        return None

    def _store_in_cache(self, ticker: str, task_name: str, data: Any):
        """Store data in cache with timestamp"""
        if self.use_cache:
            cache_key = f"{ticker}_{task_name}"
            self.cache[cache_key] = {"data": data, "timestamp": datetime.now()}

    async def _execute_task_with_retry(
        self, task_name: str, task_func: Callable, *args, **kwargs
    ) -> TaskResult:
        """Execute a task with retry logic and error handling"""
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Executing {task_name} (attempt {attempt + 1})")
                
                # Check if function is async or sync
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    result = task_func(*args, **kwargs)

                execution_time = time.time() - start_time
                task_result = TaskResult(
                    task_name=task_name,
                    status=TaskStatus.COMPLETED,
                    data=result,
                    execution_time=execution_time,
                )

                self.task_results[task_name] = task_result
                return task_result

            except Exception as e:
                self.logger.error(
                    f"Task {task_name} failed on attempt {attempt + 1}: {str(e)}"
                )

                if attempt == self.max_retries - 1:  # Last attempt
                    execution_time = time.time() - start_time
                    task_result = TaskResult(
                        task_name=task_name,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        execution_time=execution_time,
                    )
                    self.task_results[task_name] = task_result
                    return task_result

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2**attempt)

    def _collect_company_data(self, ticker: str, company_name: str) -> Dict:
        """Task: Collect all company data"""
        # Check cache first
        cached_data = self._get_from_cache(ticker, "data_collection")
        if cached_data:
            self.logger.info(f"Using cached data for {ticker}")
            return cached_data

        # Collect fresh data
        data = self.data_collector.collect_separated_data(ticker, company_name)
        
        if not data:
            raise Exception(f"Failed to collect data for {ticker}")

        # Store in cache
        self._store_in_cache(ticker, "data_collection", data)

        return data

    def _store_data(self, ticker: str, data: Dict) -> bool:
        """Task: Store data in appropriate storage systems"""
        if not data or not isinstance(data, dict):
            raise Exception(f"Invalid data format for {ticker}")
            
        # Store structured data
        structured_data = data.get("structured", {})
        if structured_data:
            structured_stored = self.structured_manager.store_company_data(
                ticker, structured_data
            )
        else:
            structured_stored = True  # No structured data to store

        # Store unstructured data
        unstructured_data = data.get("unstructured", {})
        if unstructured_data:
            vector_stored = self.vector_manager.store_unstructured_data(
                ticker, unstructured_data
            )
        else:
            vector_stored = True  # No unstructured data to store

        return structured_stored and vector_stored

    def _analyze_financials(self, ticker: str) -> Dict:
        """Task: Run financial analysis"""
        # Check cache
        cached_analysis = self._get_from_cache(ticker, "financial_analysis")
        if cached_analysis:
            return cached_analysis

        # Get data from storage
        structured_data = self.structured_manager.get_latest_company_data(ticker)
        if not structured_data:
            raise Exception(f"No structured data found for {ticker}")

        # Run analysis
        analysis = self.financial_agent.analyze_company_financials(structured_data)
        
        if not analysis or "error" in analysis:
            raise Exception(f"Financial analysis failed for {ticker}: {analysis.get('error', 'Unknown error')}")

        # Cache result
        self._store_in_cache(ticker, "financial_analysis", analysis)

        return analysis

    def _analyze_market_intelligence(self, ticker: str) -> Dict:
        """Task: Run market intelligence analysis"""
        # Check cache
        cached_intelligence = self._get_from_cache(ticker, "market_intelligence")
        if cached_intelligence:
            return cached_intelligence

        # Run analysis
        focus_areas = [
            "sentiment",
            "news_impact",
            "risk_factors"
        ]
        intelligence = self.market_agent.analyze_market_intelligence(
            ticker, focus_areas
        )
        
        if not intelligence or "error" in intelligence:
            raise Exception(f"Market intelligence analysis failed for {ticker}: {intelligence.get('error', 'Unknown error')}")

        # Cache result
        self._store_in_cache(ticker, "market_intelligence", intelligence)

        return intelligence

    def _calculate_financial_score(self, financial_analysis: Dict) -> float:
        """Calculate a financial score from 0-100 based on available metrics"""
        score = 60  # Start with neutral base score
        
        # Check valuation metrics
        valuation = financial_analysis.get("valuation_analysis", {})
        pe_ratio = valuation.get("pe_ratio", 0)
        
        if pe_ratio and pe_ratio > 0:
            if 10 <= pe_ratio <= 20:  # Good P/E range
                score += 15
            elif 20 < pe_ratio <= 30:  # Acceptable P/E
                score += 5
            elif pe_ratio > 40:  # High P/E, might be overvalued
                score -= 10
        
        # Check risk metrics
        risk_assessment = financial_analysis.get("risk_assessment", {})
        beta = risk_assessment.get("beta", 1.0)
        
        if beta and 0.8 <= beta <= 1.2:  # Moderate risk
            score += 10
        elif beta > 1.5:  # High risk
            score -= 15
        
        # Check market cap (stability indicator)
        company_overview = financial_analysis.get("company_overview", {})
        market_cap = company_overview.get("market_cap", 0)
        
        if market_cap > 100_000_000_000:  # Large cap (>100B)
            score += 10
        elif market_cap > 10_000_000_000:  # Mid cap
            score += 5
        
        # Check overall assessment
        overall_assessment = financial_analysis.get("overall_assessment", {})
        assessment_score = overall_assessment.get("score", 0)
        if assessment_score >= 2:  # Good score from the financial agent
            score += 10
        
        # Ensure score is within bounds
        return max(0, min(100, score))

    def _generate_combined_assessment(
        self, ticker: str, financial_analysis: Dict, market_intelligence: Dict
    ) -> Dict:
        """Task: Generate final combined investment assessment"""

        # Extract key metrics from financial analysis
        financial_assessment = financial_analysis.get("overall_assessment", {})
        
        # Calculate financial score based on available metrics
        financial_score = self._calculate_financial_score(financial_analysis)
        
        # Get market sentiment from market intelligence
        market_assessment = market_intelligence.get("overall_assessment", {})
        market_sentiment = market_assessment.get("market_sentiment", "neutral")

        # Combine scores (weighted average)
        financial_weight = 0.6
        market_weight = 0.4

        # Convert market sentiment to numeric score
        sentiment_scores = {
            "very_positive": 85,
            "positive": 75,
            "neutral": 60,
            "negative": 45,
            "very_negative": 30,
        }
        market_score = sentiment_scores.get(market_sentiment, 60)

        combined_score = (financial_score * financial_weight) + (
            market_score * market_weight
        )

        # Generate recommendation
        if combined_score >= 80:
            recommendation = "STRONG BUY"
            confidence = "High"
        elif combined_score >= 70:
            recommendation = "BUY"
            confidence = "Medium-High"
        elif combined_score >= 60:
            recommendation = "HOLD"
            confidence = "Medium"
        elif combined_score >= 50:
            recommendation = "WEAK HOLD"
            confidence = "Medium-Low"
        else:
            recommendation = "SELL"
            confidence = "High"

        return {
            "ticker": ticker,
            "overall_score": round(combined_score, 1),
            "recommendation": recommendation,
            "confidence": confidence,
            "financial_score": financial_score,
            "market_score": market_score,
            "analysis_timestamp": datetime.now().isoformat(),
            "key_strengths": self._extract_strengths(
                financial_analysis, market_intelligence
            ),
            "key_risks": self._extract_risks(financial_analysis, market_intelligence),
            "price_target": self._calculate_price_target(
                financial_analysis, market_intelligence
            ),
        }

    def _extract_strengths(
        self, financial_analysis: Dict, market_intelligence: Dict
    ) -> List[str]:
        """Extract key investment strengths"""
        strengths = []

        # From financial analysis
        valuation = financial_analysis.get("valuation_analysis", {})
        pe_ratio = valuation.get("pe_ratio", 0)
        if pe_ratio and 15 <= pe_ratio <= 25:
            strengths.append("Reasonable valuation metrics")

        risk_assessment = financial_analysis.get("risk_assessment", {})
        beta = risk_assessment.get("beta", 1.0)
        if beta and 0.8 <= beta <= 1.2:
            strengths.append("Moderate risk profile")

        # From market intelligence
        sentiment = market_intelligence.get("overall_assessment", {}).get(
            "market_sentiment", "neutral"
        )
        if sentiment in ["positive", "very_positive"]:
            strengths.append("Positive market sentiment")

        return strengths[:3]  # Top 3 strengths

    def _extract_risks(
        self, financial_analysis: Dict, market_intelligence: Dict
    ) -> List[str]:
        """Extract key investment risks"""
        risks = []

        # From financial analysis
        risk_assessment = financial_analysis.get("risk_assessment", {})
        beta = risk_assessment.get("beta", 1.0)
        if beta and beta > 1.5:
            risks.append("High volatility risk")
            
        # Check price position
        price_position = risk_assessment.get("price_position_52week", 0.5)
        if price_position > 0.9:
            risks.append("Near 52-week high - potential overvaluation")
        elif price_position < 0.1:
            risks.append("Near 52-week low - potential fundamental issues")

        # From market intelligence
        sentiment = market_intelligence.get("overall_assessment", {}).get(
            "market_sentiment", "neutral"
        )
        if sentiment in ["negative", "very_negative"]:
            risks.append("Negative market sentiment")

        return risks[:3]  # Top 3 risks

    def _calculate_price_target(
        self, financial_analysis: Dict, market_intelligence: Dict
    ) -> Dict:
        """Calculate 12-month price target"""
        current_price = financial_analysis.get("company_overview", {}).get(
            "current_price", 0
        )

        if current_price == 0:
            return {"target_price": None, "upside_potential": None}

        # Simple price target calculation (can be enhanced)
        financial_assessment = financial_analysis.get("overall_assessment", {})
        financial_score = financial_assessment.get("score", 60)
        
        market_assessment = market_intelligence.get("overall_assessment", {})
        market_sentiment = market_assessment.get("market_sentiment", "neutral")

        # Base multiplier on scores
        multiplier = 1.0 + ((financial_score - 60) / 100)  # Base growth expectation

        # Adjust for market sentiment
        sentiment_adjustments = {
            "very_positive": 1.15,
            "positive": 1.08,
            "neutral": 1.0,
            "negative": 0.92,
            "very_negative": 0.85,
        }
        multiplier *= sentiment_adjustments.get(market_sentiment, 1.0)

        target_price = round(current_price * multiplier, 2)
        upside_potential = round(
            ((target_price - current_price) / current_price) * 100, 1
        )

        return {
            "target_price": target_price,
            "current_price": current_price,
            "upside_potential": f"{upside_potential}%",
            "time_horizon": "12 months",
        }

    async def research_company(
        self, ticker: str, company_name: str, skip_cache: bool = False
    ) -> Dict:
        """
        Main orchestration method - Complete investment research pipeline

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            company_name: Full company name (e.g., 'Apple Inc.')
            skip_cache: Force fresh analysis ignoring cache

        Returns:
            Comprehensive investment research report
        """

        if skip_cache:
            self.use_cache = False

        research_start_time = time.time()

        try:
            self._update_progress(f"Starting research for {ticker} ({company_name})", 0)

            # Task 1: Data Collection (0-30%)
            self._update_progress(
                "Collecting financial data from multiple sources...", 5
            )
            data_task = await self._execute_task_with_retry(
                "data_collection", self._collect_company_data, ticker, company_name
            )

            if data_task.status != TaskStatus.COMPLETED:
                raise Exception(f"Data collection failed: {data_task.error}")

            self._update_progress("Data collection completed", 30)

            # Task 2: Data Storage (30-40%)
            self._update_progress("Storing data in knowledge base...", 35)
            storage_task = await self._execute_task_with_retry(
                "data_storage", self._store_data, ticker, data_task.data
            )

            self._update_progress("Data storage completed", 40)

            if self.parallel_execution:
                # Tasks 3 & 4: Parallel Analysis (40-90%)
                self._update_progress(
                    "Running AI analysis (financial + market intelligence)...", 45
                )

                # Create async wrappers for the sync methods and run in parallel
                async def run_financial():
                    return await self._execute_task_with_retry(
                        "financial_analysis", self._analyze_financials, ticker
                    )
                
                async def run_market():
                    return await self._execute_task_with_retry(
                        "market_intelligence", self._analyze_market_intelligence, ticker
                    )

                financial_task, market_task = await asyncio.gather(
                    run_financial(),
                    run_market(),
                    return_exceptions=True,
                )
            else:
                # Sequential execution
                self._update_progress("Analyzing financial metrics...", 45)
                financial_task = await self._execute_task_with_retry(
                    "financial_analysis", self._analyze_financials, ticker
                )

                self._update_progress("Analyzing market intelligence...", 70)
                market_task = await self._execute_task_with_retry(
                    "market_intelligence", self._analyze_market_intelligence, ticker
                )

            self._update_progress("AI analysis completed", 90)

            # Handle exceptions from parallel execution
            if isinstance(financial_task, Exception):
                raise financial_task
            if isinstance(market_task, Exception):
                raise market_task

            # Validate analysis results
            if financial_task.status != TaskStatus.COMPLETED:
                raise Exception(f"Financial analysis failed: {financial_task.error}")

            if market_task.status != TaskStatus.COMPLETED:
                raise Exception(f"Market intelligence failed: {market_task.error}")

            # Task 5: Generate Combined Assessment (90-100%)
            self._update_progress("Generating investment recommendation...", 95)

            combined_assessment = self._generate_combined_assessment(
                ticker, financial_task.data, market_task.data
            )

            # Calculate total execution time
            total_execution_time = time.time() - research_start_time

            # Compile final results
            final_results = {
                "research_metadata": {
                    "ticker": ticker,
                    "company_name": company_name,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "total_execution_time": round(total_execution_time, 2),
                    "cache_used": any(
                        result.status == TaskStatus.CACHED
                        for result in self.task_results.values()
                    ),
                    "orchestrator_version": "1.0",
                },
                "financial_analysis": financial_task.data,
                "market_intelligence": market_task.data,
                "combined_assessment": combined_assessment,
                "execution_summary": {
                    "tasks_completed": len(
                        [
                            t
                            for t in self.task_results.values()
                            if t.status == TaskStatus.COMPLETED
                        ]
                    ),
                    "tasks_failed": len(
                        [
                            t
                            for t in self.task_results.values()
                            if t.status == TaskStatus.FAILED
                        ]
                    ),
                    "task_details": {
                        name: {
                            "status": result.status.value,
                            "execution_time": result.execution_time,
                        }
                        for name, result in self.task_results.items()
                    },
                },
            }

            # Store execution history
            self.execution_history.append(
                {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": total_execution_time,
                    "success": True,
                }
            )

            self._update_progress("Investment research completed successfully!", 100)

            return final_results

        except Exception as e:
            self.logger.error(f"Research pipeline failed for {ticker}: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Store failed execution
            self.execution_history.append(
                {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": time.time() - research_start_time,
                    "success": False,
                    "error": str(e),
                }
            )

            raise e

    def get_system_health(self) -> Dict:
        """Get orchestrator system health and performance metrics"""
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e["success"]])

        avg_execution_time = 0
        if successful_executions > 0:
            avg_execution_time = (
                sum(
                    [
                        e["execution_time"]
                        for e in self.execution_history
                        if e["success"]
                    ]
                )
                / successful_executions
            )

        return {
            "system_status": "healthy"
            if total_executions == 0 or (successful_executions / total_executions) > 0.8
            else "degraded",
            "total_executions": total_executions,
            "success_rate": round((successful_executions / total_executions) * 100, 1)
            if total_executions > 0
            else 0,
            "average_execution_time": round(avg_execution_time, 2),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "active_cache_entries": len(self.cache),
            "last_execution": self.execution_history[-1]
            if self.execution_history
            else None,
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent executions"""
        if not hasattr(self, "cache_hits"):
            return 0.0

        total_cache_checks = getattr(self, "total_cache_checks", 0)
        cache_hits = getattr(self, "cache_hits", 0)

        return (
            round((cache_hits / total_cache_checks) * 100, 1)
            if total_cache_checks > 0
            else 0.0
        )

    def clear_cache(self, ticker: str = None):
        """Clear cache for specific ticker or all cache"""
        if ticker:
            keys_to_remove = [
                key for key in self.cache.keys() if key.startswith(ticker)
            ]
            for key in keys_to_remove:
                del self.cache[key]
            self.logger.info(f"Cleared cache for {ticker}")
        else:
            self.cache.clear()
            self.logger.info("Cleared all cache")

    def export_results(self, results: Dict, format: str = "json") -> str:
        """Export research results to file"""
        ticker = results["research_metadata"]["ticker"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            filename = f"research_{ticker}_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)

        return filename


# Example usage and testing
async def main():
    """Example usage of the Orchestrator Agent"""

    # Initialize orchestrator
    orchestrator = InvestmentResearchOrchestrator(
        use_cache=True, cache_duration_minutes=30, parallel_execution=True
    )

    # Set up progress callback (optional)
    def progress_callback(message: str, percentage: float):
        print(f"[{percentage:5.1f}%] {message}")

    orchestrator.set_progress_callback(progress_callback)

    try:
        # Research a company
        print("🚀 Starting Investment Research for Apple Inc...")
        results = await orchestrator.research_company("AAPL", "Apple Inc.")

        # Display results
        print("\n" + "=" * 80)
        print("📊 INVESTMENT RESEARCH COMPLETED")
        print("=" * 80)

        # Key metrics
        assessment = results["combined_assessment"]
        print(f"🎯 Recommendation: {assessment['recommendation']}")
        print(f"📈 Overall Score: {assessment['overall_score']}/100")
        print(f"💰 Price Target: ${assessment['price_target']['target_price']}")
        print(f"📊 Upside Potential: {assessment['price_target']['upside_potential']}")
        print(
            f"⏱️  Execution Time: {results['research_metadata']['total_execution_time']} seconds"
        )

        # Export results
        filename = orchestrator.export_results(results)
        print(f"💾 Results exported to: {filename}")

        # System health
        health = orchestrator.get_system_health()
        print(f"\n🏥 System Health: {health['system_status'].upper()}")
        print(f"✅ Success Rate: {health['success_rate']}%")

    except Exception as e:
        print(f"❌ Research failed: {str(e)}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
