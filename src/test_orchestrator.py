"""
Test script for the Orchestrator Agent
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.orchestrator_agent import InvestmentResearchOrchestrator

async def test_orchestrator():
    """Test the orchestrator with a simple example"""
    
    print("Testing Investment Research Orchestrator...")
    
    # Initialize orchestrator with basic settings
    orchestrator = InvestmentResearchOrchestrator(
        use_cache=True,
        cache_duration_minutes=30,
        parallel_execution=False,  # Start with sequential for easier debugging
        use_llm=False  # Start without LLM for basic testing
    )
    
    # Set up progress callback
    def progress_callback(message: str, percentage: float):
        print(f"[{percentage:5.1f}%] {message}")
    
    orchestrator.set_progress_callback(progress_callback)
    
    try:
        # Test system health before execution
        health = orchestrator.get_system_health()
        print(f"Initial System Health: {health['system_status'].upper()}")
        
        # Test with a simple ticker
        print("\nStarting test research for Apple Inc...")
        results = await orchestrator.research_company("AAPL", "Apple Inc.")
        
        # Display basic results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        metadata = results.get("research_metadata", {})
        print(f"Ticker: {metadata.get('ticker', 'N/A')}")
        print(f"Execution Time: {metadata.get('total_execution_time', 0):.2f} seconds")
        print(f"Cache Used: {metadata.get('cache_used', False)}")
        
        # Check if we have analysis results
        financial_analysis = results.get("financial_analysis", {})
        market_intelligence = results.get("market_intelligence", {})
        combined_assessment = results.get("combined_assessment", {})
        
        if financial_analysis:
            print(f"Financial Analysis: Available")
            overall_assessment = financial_analysis.get("overall_assessment", {})
            print(f"   - Rating: {overall_assessment.get('overall_rating', 'N/A')}")
        else:
            print(f"Financial Analysis: Missing")
            
        if market_intelligence:
            print(f"Market Intelligence: Available")
        else:
            print(f"Market Intelligence: Missing")
            
        if combined_assessment:
            print(f"Combined Assessment: Available")
            print(f"   - Recommendation: {combined_assessment.get('recommendation', 'N/A')}")
            print(f"   - Overall Score: {combined_assessment.get('overall_score', 'N/A')}")
        else:
            print(f"Combined Assessment: Missing")
        
        # Test system health after execution
        health_after = orchestrator.get_system_health()
        print(f"\nFinal System Health: {health_after['system_status'].upper()}")
        print(f"Success Rate: {health_after['success_rate']}%")
        print(f"Total Executions: {health_after['total_executions']}")
        
        print("\nOrchestrator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Still check system health to see what happened
        health_after = orchestrator.get_system_health()
        print(f"System Health after error: {health_after['system_status'].upper()}")
        if health_after['total_executions'] > 0:
            print(f"Success Rate: {health_after['success_rate']}%")
        
        return False

if __name__ == "__main__":
    print("Running Orchestrator Agent Test")
    print("=" * 50)
    
    # Run the test
    success = asyncio.run(test_orchestrator())
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed - check the error messages above")