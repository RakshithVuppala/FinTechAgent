"""
Example usage of the Investment Research Orchestrator
Shows how to integrate the orchestrator into your applications
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.orchestrator_agent import InvestmentResearchOrchestrator

async def research_multiple_stocks():
    """Example: Research multiple stocks sequentially"""
    
    stocks = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc.")
    ]
    
    # Initialize orchestrator with caching enabled
    orchestrator = InvestmentResearchOrchestrator(
        use_cache=True,
        cache_duration_minutes=60,
        parallel_execution=True,
        use_llm=True  # Use LLM if available
    )
    
    results = {}
    
    for ticker, company_name in stocks:
        print(f"\n{'='*60}")
        print(f"Researching {ticker} - {company_name}")
        print('='*60)
        
        try:
            # Research the company
            result = await orchestrator.research_company(ticker, company_name)
            
            # Extract key information
            combined_assessment = result.get("combined_assessment", {})
            metadata = result.get("research_metadata", {})
            
            print(f"✅ {ticker} Analysis Complete")
            print(f"   Recommendation: {combined_assessment.get('recommendation', 'N/A')}")
            print(f"   Score: {combined_assessment.get('overall_score', 'N/A')}/100")
            print(f"   Execution Time: {metadata.get('total_execution_time', 0):.2f}s")
            
            results[ticker] = result
            
        except Exception as e:
            print(f"❌ Failed to research {ticker}: {str(e)}")
            results[ticker] = {"error": str(e)}
    
    # Summary report
    print(f"\n{'='*60}")
    print("PORTFOLIO ANALYSIS SUMMARY")
    print('='*60)
    
    recommendations = {}
    for ticker, result in results.items():
        if "error" not in result:
            combined = result.get("combined_assessment", {})
            rec = combined.get("recommendation", "UNKNOWN")
            score = combined.get("overall_score", 0)
            recommendations[ticker] = {"recommendation": rec, "score": score}
            print(f"{ticker:6} | {rec:12} | Score: {score:5.1f}")
        else:
            print(f"{ticker:6} | {'ERROR':12} | {result['error']}")
    
    # System health
    health = orchestrator.get_system_health()
    print(f"\nSystem Performance:")
    print(f"  Success Rate: {health['success_rate']}%")
    print(f"  Avg Execution Time: {health['average_execution_time']:.2f}s")
    print(f"  Cache Hit Rate: {health['cache_hit_rate']}%")
    
    return results

async def research_with_custom_callback():
    """Example: Research with custom progress tracking"""
    
    # Custom progress callback with more detailed logging
    def detailed_progress(message: str, percentage: float):
        print(f"[{percentage:6.1f}%] {message}")
        
        # You could also:
        # - Update a progress bar in a GUI
        # - Send progress updates to a web frontend
        # - Log to a file
        # - Send notifications
    
    orchestrator = InvestmentResearchOrchestrator(
        use_cache=False,  # Fresh analysis
        parallel_execution=True
    )
    
    orchestrator.set_progress_callback(detailed_progress)
    
    print("Research with detailed progress tracking:")
    result = await orchestrator.research_company("TSLA", "Tesla Inc.")
    
    # Export results
    filename = orchestrator.export_results(result)
    print(f"\n📁 Results exported to: {filename}")
    
    return result

def sync_wrapper_example():
    """Example: Wrapper for using orchestrator in sync code"""
    
    def research_company_sync(ticker: str, company_name: str):
        """Synchronous wrapper for the async orchestrator"""
        orchestrator = InvestmentResearchOrchestrator()
        
        # Run the async function in a new event loop
        return asyncio.run(orchestrator.research_company(ticker, company_name))
    
    # Use it like a regular synchronous function
    print("Testing synchronous wrapper:")
    try:
        result = research_company_sync("NVDA", "NVIDIA Corporation")
        combined = result.get("combined_assessment", {})
        print(f"NVDA Recommendation: {combined.get('recommendation', 'N/A')}")
        print(f"NVDA Score: {combined.get('overall_score', 'N/A')}")
        return result
    except Exception as e:
        print(f"Sync wrapper failed: {e}")
        return None

async def main():
    """Main function demonstrating different usage patterns"""
    
    print("Investment Research Orchestrator - Usage Examples")
    print("=" * 60)
    
    # Example 1: Multiple stocks
    print("\n🔍 Example 1: Researching multiple stocks")
    await research_multiple_stocks()
    
    # Example 2: Custom progress tracking
    print("\n🔍 Example 2: Custom progress tracking")
    await research_with_custom_callback()
    
    # Example 3: Synchronous wrapper
    print("\n🔍 Example 3: Synchronous wrapper")
    sync_wrapper_example()
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())