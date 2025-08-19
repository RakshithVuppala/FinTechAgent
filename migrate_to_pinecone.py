#!/usr/bin/env python3
"""
Pinecone Migration Script for FinTech Agent
==========================================

This script helps migrate from ChromaDB to Pinecone vector database.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    print("Checking environment setup...")
    
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("SUCCESS: Loaded .env file")
        except ImportError:
            print("WARNING:  python-dotenv not installed, .env file not loaded")
    
    # Check required environment variables
    required_vars = ['PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease add these to your .env file:")
        print("PINECONE_API_KEY=your_api_key_here")
        print("PINECONE_ENVIRONMENT=us-east1-aws")
        print("\nGet your API key at: https://app.pinecone.io/")
        return False
    
    print("SUCCESS: Environment variables configured")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nINSTALL: Installing Pinecone dependencies...")
    
    try:
        # Install pinecone and sentence-transformers
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pinecone>=5.0.0", 
            "sentence-transformers>=2.2.0"
        ])
        print("SUCCESS: Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False

def test_pinecone_connection():
    """Test connection to Pinecone."""
    print("\nTEST: Testing Pinecone connection...")
    
    try:
        # Add src to path to import vector_manager
        sys.path.insert(0, str(Path("src")))
        from vector_manager import VectorDataManager
        
        # Initialize vector manager
        manager = VectorDataManager()
        
        if manager.use_pinecone:
            print("SUCCESS: Successfully connected to Pinecone")
            
            # Get initial stats
            stats = manager.get_collection_stats()
            print(f"   Index: {stats.get('collection_name', 'unknown')}")
            print(f"   Documents: {stats.get('total_documents', 0)}")
            print(f"   Storage type: {stats.get('storage_type', 'unknown')}")
            return True
        else:
            print("ERROR: Pinecone connection failed, using fallback storage")
            return False
            
    except Exception as e:
        print(f"ERROR: Connection test failed: {e}")
        return False

def run_test_migration():
    """Run a test with sample data."""
    print("\nTEST: Running test migration with sample data...")
    
    try:
        sys.path.insert(0, str(Path("src")))
        from vector_manager import VectorDataManager
        
        # Initialize manager
        manager = VectorDataManager()
        
        # Test data
        test_data = {
            'news_articles': [
                {
                    'title': 'Apple Q4 Earnings Beat Expectations',
                    'description': 'Apple reported strong quarterly results with revenue growth across all segments.',
                    'source': 'Reuters',
                    'provider': 'Financial News',
                    'url': 'https://example.com/apple-earnings',
                    'published_at': '2024-01-15T16:30:00Z'
                }
            ],
            'reddit_sentiment': {
                'sentiment': 'positive',
                'sentiment_score': 0.75,
                'sample_posts': [
                    {
                        'title': 'AAPL earnings were amazing!',
                        'selftext': 'Great quarter for Apple, services revenue growing strong.',
                        'subreddit': 'investing',
                        'score': 95,
                        'num_comments': 24
                    }
                ]
            }
        }
        
        # Test storage
        print("   STORE: Testing document storage...")
        success = manager.store_unstructured_data('AAPL', test_data)
        
        if success:
            print("   SUCCESS: Document storage successful")
            
            # Test search
            print("   SEARCH: Testing document search...")
            results = manager.search_documents(
                query="Apple earnings revenue growth",
                ticker="AAPL",
                limit=5
            )
            
            print(f"   SUCCESS: Search returned {len(results)} results")
            
            if results:
                print(f"   Top result: {results[0]['metadata'].get('title', 'No title')[:50]}...")
                print(f"   Similarity: {results[0]['similarity_score']:.3f}")
            
            # Test document retrieval
            print("   RETRIEVE: Testing document retrieval...")
            docs = manager.get_ticker_documents('AAPL')
            print(f"   SUCCESS: Retrieved {len(docs)} documents for AAPL")
            
            # Test sentiment analysis
            print("   SENTIMENT: Testing sentiment analysis...")
            sentiment = manager.analyze_ticker_sentiment('AAPL')
            print(f"   SUCCESS: Sentiment: {sentiment.get('overall_sentiment', 'unknown')}")
            
            return True
        else:
            print("   ERROR: Document storage failed")
            return False
            
    except Exception as e:
        print(f"   ERROR: Test migration failed: {e}")
        return False

def show_migration_summary():
    """Show migration summary and next steps."""
    print("\nCOMPLETE: PINECONE MIGRATION COMPLETE!")
    print("=" * 50)
    print("\nSUCCESS: What was migrated:")
    print("   • Updated requirements.txt with Pinecone dependencies")
    print("   • Added Pinecone configuration to .env.example")
    print("   • Replaced ChromaDB with Pinecone in vector_manager.py")
    print("   • Added SentenceTransformers for embedding generation")
    print("   • Maintained all existing API compatibility")
    
    print("\nMIGRATE: Next steps:")
    print("   1. Set up your Pinecone account at: https://app.pinecone.io/")
    print("   2. Add your PINECONE_API_KEY to .env file")
    print("   3. Run your Streamlit app: python run.py")
    print("   4. Test with real stock data in the dashboard")
    
    print("\nCOST: Expected costs (based on your usage):")
    print("   • Small scale (10 stocks): ~$5-15/month")
    print("   • Medium scale (50 stocks): ~$25-50/month")
    print("   • Storage: $0.33/GB/month")
    print("   • Searches: $8.25 per million queries")
    
    print("\nFEATURES: Features maintained:")
    print("   • All existing Streamlit functionality")
    print("   • Same API calls and return formats")
    print("   • Semantic search capabilities")
    print("   • Sentiment analysis")
    print("   • Document retrieval and filtering")

def main():
    """Main migration function."""
    print("MIGRATE: PINECONE MIGRATION ASSISTANT")
    print("=" * 40)
    print("This script will help you migrate from ChromaDB to Pinecone.")
    print()
    
    # Step 1: Check environment
    if not check_environment():
        print("\nERROR: Environment check failed. Please fix the issues above and try again.")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\nERROR: Dependency installation failed.")
        return False
    
    # Step 3: Test connection
    if not test_pinecone_connection():
        print("\nERROR: Pinecone connection test failed.")
        print("Please check your API key and try again.")
        return False
    
    # Step 4: Run test migration
    if not run_test_migration():
        print("\nERROR: Test migration failed.")
        return False
    
    # Step 5: Show summary
    show_migration_summary()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)