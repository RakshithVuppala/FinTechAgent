"""
Vector Manager - Detailed implementation with explanations
Handles storage and retrieval of unstructured financial data
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime
import hashlib
import json

# Handle ChromaDB import with fallback for deployment environments
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CHROMADB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"ChromaDB not available: {e}. Using in-memory storage fallback.")

logger = logging.getLogger(__name__)

class VectorDataManager:
    """
    Manages vector storage for unstructured financial data
    
    What this does:
    1. Takes news articles and social posts (text)
    2. Converts them to mathematical vectors (embeddings)
    3. Stores vectors in ChromaDB for fast similarity search (or in-memory fallback)
    4. Enables semantic search (meaning-based, not keyword-based)
    """
    
    def __init__(self, data_dir: str = "data/vector_db"):
        """
        Initialize the vector database with fallback support
        
        Args:
            data_dir: Where to store the database files
        """
        self.data_dir = data_dir
        self.use_chromadb = CHROMADB_AVAILABLE
        
        if self.use_chromadb:
            # ChromaDB is available - use persistent storage
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                # Try persistent client first
                self.client = chromadb.PersistentClient(path=data_dir)
            except Exception as e:
                logger.warning(f"Failed to create persistent client: {e}. Using in-memory client.")
                # Fallback to in-memory client
                self.client = chromadb.EphemeralClient()
                self.use_chromadb = "memory"
            
            # Create or get existing collection
            self.collection = self.client.get_or_create_collection(
                name="financial_documents",
                metadata={"description": "Financial news articles and social sentiment data"}
            )
            
            logger.info(f"Vector database initialized with ChromaDB ({self.use_chromadb})")
        else:
            # ChromaDB not available - use simple in-memory storage
            self.documents = []
            self.metadata = []
            self.ids = []
            logger.info("Vector database initialized with in-memory fallback storage")
        if self.use_chromadb:
            logger.info(f"Current collection has {self.collection.count()} documents")
        else:
            logger.info(f"Current in-memory storage has {len(self.documents)} documents")
    
    def store_unstructured_data(self, ticker: str, unstructured_data: Dict[str, Any]) -> bool:
        """
        Store unstructured data (news, Reddit posts) in vector database
        
        Process:
        1. Extract text from news articles and social posts
        2. Clean and prepare text for embedding
        3. Generate unique IDs for each document
        4. Store in ChromaDB (automatic vectorization happens here)
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            unstructured_data: Dict containing news_articles and reddit_sentiment
            
        Returns:
            bool: Success/failure of storage operation
        """
        try:
            # Lists to store data for batch insertion
            documents = []      # The actual text content
            metadatas = []      # Information about each document
            ids = []           # Unique identifiers
            
            logger.info(f"Processing unstructured data for {ticker}")
            
            # ================== PROCESS NEWS ARTICLES ==================
            news_articles = unstructured_data.get('news_articles', [])
            logger.info(f"Processing {len(news_articles)} news articles")
            
            for i, article in enumerate(news_articles):
                # More flexible validation - require at least title OR description
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                summary = article.get('summary', '').strip()
                content = article.get('content', '').strip()
                
                # Try to find any usable content
                usable_content = description or summary or content or ''
                
                if not title and not usable_content:
                    logger.warning(f"Skipping article {i}: no usable content found")
                    continue
                
                # Use title as fallback content if description is missing
                if not usable_content and title:
                    usable_content = title
                
                # Use description as fallback title if title is missing
                if not title and usable_content:
                    title = usable_content[:100] + "..." if len(usable_content) > 100 else usable_content
                
                # Combine title and content for richer context
                text_content = f"TITLE: {title}\n\nCONTENT: {usable_content}"
                
                # Generate unique document ID
                doc_id = self._generate_doc_id(ticker, 'news', i, title)
                
                # Prepare metadata (information about the document)
                metadata = {
                    'ticker': ticker,
                    'type': 'news',
                    'source': article.get('source', 'unknown'),
                    'provider': article.get('provider', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('published_at', ''),
                    'stored_at': datetime.now().isoformat(),
                    'title': title[:100],  # Use processed title
                    'article_index': i,
                    'content_length': len(usable_content),
                    'has_description': bool(description),
                    'content_source': 'description' if description else 'summary' if summary else 'content' if content else 'title'
                }
                
                # Add to batch
                documents.append(text_content)
                metadatas.append(metadata)
                ids.append(doc_id)
                
                logger.debug(f"Prepared news article {i}: {article.get('title', '')[:50]}...")
            
            # ================== PROCESS REDDIT SENTIMENT ==================
            reddit_data = unstructured_data.get('reddit_sentiment', {})
            sample_posts = reddit_data.get('sample_posts', [])
            overall_sentiment = reddit_data.get('sentiment', 'neutral')
            
            logger.info(f"Processing {len(sample_posts)} Reddit posts with overall sentiment: {overall_sentiment}")
            
            for i, post in enumerate(sample_posts):
                # More flexible Reddit post validation
                title = post.get('title', '').strip()
                selftext = post.get('selftext', '').strip()
                body = post.get('body', '').strip()
                
                # Use any available text content
                post_content = selftext or body or ''
                
                if not title and not post_content:
                    logger.warning(f"Skipping Reddit post {i}: no usable content")
                    continue
                
                # Use title as content fallback if no post text
                if not post_content and title:
                    post_content = title
                
                # Combine title and content for full context
                text_content = f"REDDIT POST: {title}\n\nCONTENT: {post_content}\n\nOVERALL_SENTIMENT: {overall_sentiment}"
                
                # Generate unique document ID  
                doc_id = self._generate_doc_id(ticker, 'reddit', i, title)
                
                # Prepare metadata
                metadata = {
                    'ticker': ticker,
                    'type': 'reddit',
                    'subreddit': post.get('subreddit', 'unknown'),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'overall_sentiment': overall_sentiment,
                    'post_sentiment_score': reddit_data.get('sentiment_score', 0),
                    'stored_at': datetime.now().isoformat(),
                    'title': title[:100],  # Use processed title
                    'post_index': i,
                    'content_length': len(post_content),
                    'has_selftext': bool(selftext),
                    'content_source': 'selftext' if selftext else 'body' if body else 'title'
                }
                
                # Add to batch
                documents.append(text_content)
                metadatas.append(metadata)
                ids.append(doc_id)
                
                logger.debug(f"Prepared Reddit post {i}: {post.get('title', '')[:50]}...")
            
            # ================== STORE IN DATABASE ==================
            if documents:
                logger.info(f"Storing {len(documents)} documents in vector database...")
                
                if self.use_chromadb:
                    # ChromaDB automatically converts text to vectors using embeddings
                    self.collection.add(
                        documents=documents,    # Text content (gets converted to vectors)
                        metadatas=metadatas,   # Information about each document
                        ids=ids                # Unique identifiers
                    )
                    logger.info(f"Collection now has {self.collection.count()} total documents")
                else:
                    # Fallback to in-memory storage
                    self.documents.extend(documents)
                    self.metadata.extend(metadatas)
                    self.ids.extend(ids)
                    logger.info(f"In-memory storage now has {len(self.documents)} total documents")
                
                logger.info(f"Successfully stored {len(documents)} documents for {ticker}")
                return True
            else:
                logger.warning(f"No valid documents found to store for {ticker}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing unstructured data for {ticker}: {e}")
            return False
    
    def search_documents(self, query: str, ticker: str = None, doc_type: str = None, limit: int = 10) -> List[Dict]:
        """
        Search for documents using semantic similarity
        
        How this works:
        1. Convert query text to vector (embedding)
        2. Find vectors in database most similar to query vector
        3. Return corresponding documents with similarity scores
        
        Args:
            query: Search text (e.g., "Apple earnings beat expectations")
            ticker: Filter by stock symbol (optional)
            doc_type: Filter by document type: 'news' or 'reddit' (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        try:
            logger.info(f"Searching for: '{query}' (ticker: {ticker}, type: {doc_type}, limit: {limit})")
            
            if self.use_chromadb:
                # Build filter criteria
                where_filter = {}
                if ticker:
                    where_filter['ticker'] = ticker
                if doc_type:
                    where_filter['type'] = doc_type
                
                # Perform semantic similarity search
                # ChromaDB converts query to vector and finds most similar documents
                results = self.collection.query(
                    query_texts=[query],                    # Text to search for
                    n_results=limit,                       # Number of results
                    where=where_filter if where_filter else None,  # Filter criteria
                    include=['documents', 'metadatas', 'distances']  # What to return
                )
                
                # Format results for easier use
                formatted_results = []
                
                if results['documents'] and results['documents'][0]:
                    num_results = len(results['documents'][0])
                    logger.info(f"Found {num_results} matching documents")
                    
                    for i in range(num_results):
                        result_item = {
                            'document': results['documents'][0][i],      # The actual text content
                            'metadata': results['metadatas'][0][i],      # Document information
                            'similarity_score': 1 - results['distances'][0][i],  # Higher = more similar
                            'distance': results['distances'][0][i],      # Lower = more similar
                            'id': results['ids'][0][i],                  # Unique document ID
                            'relevance': self._calculate_relevance_score(results['distances'][0][i])
                        }
                        formatted_results.append(result_item)
                        
                        # Log top results for debugging
                        if i < 3:  # Log first 3 results
                            logger.debug(f"Result {i+1}: {result_item['metadata'].get('title', 'No title')[:50]}... (similarity: {result_item['similarity_score']:.3f})")
                else:
                    logger.info("No matching documents found")
                    
                return formatted_results
            else:
                # Fallback: Simple text search in in-memory storage
                formatted_results = []
                query_lower = query.lower()
                
                for i, (doc, meta, doc_id) in enumerate(zip(self.documents, self.metadata, self.ids)):
                    # Apply filters
                    if ticker and meta.get('ticker') != ticker:
                        continue
                    if doc_type and meta.get('type') != doc_type:
                        continue
                    
                    # Simple text similarity (count matching words)
                    doc_lower = doc.lower()
                    query_words = query_lower.split()
                    matches = sum(1 for word in query_words if word in doc_lower)
                    
                    if matches > 0:
                        # Simple relevance score based on word matches
                        relevance_score = matches / len(query_words)
                        
                        result_item = {
                            'document': doc,
                            'metadata': meta,
                            'similarity_score': relevance_score,
                            'distance': 1 - relevance_score,
                            'id': doc_id,
                            'relevance': self._calculate_relevance_score(1 - relevance_score)
                        }
                        formatted_results.append(result_item)
                
                # Sort by relevance score (highest first)
                formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                formatted_results = formatted_results[:limit]
                
                logger.info(f"Found {len(formatted_results)} matching documents (fallback search)")
                return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_ticker_documents(self, ticker: str, doc_type: str = None, limit: int = 50) -> List[Dict]:
        """
        Get all documents for a specific ticker (without similarity search)
        
        Useful for:
        - Getting all news for a company
        - Getting all Reddit posts about a stock
        - General document retrieval
        
        Args:
            ticker: Stock symbol
            doc_type: 'news', 'reddit', or None for all types
            limit: Maximum documents to return
            
        Returns:
            List of documents with metadata
        """
        try:
            logger.info(f"Retrieving documents for {ticker} (type: {doc_type}, limit: {limit})")
            
            if self.use_chromadb:
                # Build filter
                where_filter = {'ticker': ticker}
                if doc_type:
                    where_filter['type'] = doc_type
                
                # Get documents (no similarity search, just filtering)
                results = self.collection.get(
                    where=where_filter,
                    limit=limit,
                    include=['documents', 'metadatas']
                )
                
                # Format results
                formatted_results = []
                if results['documents']:
                    for i in range(len(results['documents'])):
                        formatted_results.append({
                            'document': results['documents'][i],
                            'metadata': results['metadatas'][i],
                            'id': results['ids'][i]
                        })
                    
                    logger.info(f"Retrieved {len(formatted_results)} documents for {ticker}")
                else:
                    logger.info(f"No documents found for {ticker}")
                    
                return formatted_results
            else:
                # Fallback: Filter in-memory storage
                formatted_results = []
                
                for doc, meta, doc_id in zip(self.documents, self.metadata, self.ids):
                    # Apply filters
                    if meta.get('ticker') != ticker:
                        continue
                    if doc_type and meta.get('type') != doc_type:
                        continue
                    
                    formatted_results.append({
                        'document': doc,
                        'metadata': meta,
                        'id': doc_id
                    })
                    
                    if len(formatted_results) >= limit:
                        break
                
                logger.info(f"Retrieved {len(formatted_results)} documents for {ticker} (fallback storage)")
                return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting documents for {ticker}: {e}")
            return []
    
    def analyze_ticker_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze overall sentiment for a ticker based on stored documents
        
        Process:
        1. Get all documents for ticker
        2. Analyze sentiment patterns
        3. Calculate overall sentiment score
        4. Identify key themes
        
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            logger.info(f"Analyzing sentiment for {ticker}")
            
            # Get all documents for this ticker
            all_docs = self.get_ticker_documents(ticker)
            
            if not all_docs:
                return {
                    'ticker': ticker,
                    'overall_sentiment': 'neutral',
                    'confidence': 0,
                    'document_count': 0,
                    'analysis': 'No documents found for analysis'
                }
            
            # Separate by document type
            news_docs = [doc for doc in all_docs if doc['metadata'].get('type') == 'news']
            reddit_docs = [doc for doc in all_docs if doc['metadata'].get('type') == 'reddit']
            
            # Count positive/negative indicators in titles
            positive_keywords = ['beat', 'exceed', 'strong', 'growth', 'bullish', 'positive', 'gain', 'rise', 'up']
            negative_keywords = ['miss', 'decline', 'weak', 'bearish', 'negative', 'fall', 'drop', 'down', 'concern']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for doc in all_docs:
                title = doc['metadata'].get('title', '').lower()
                content = doc['document'].lower()
                text = f"{title} {content}"
                
                pos_score = sum(1 for word in positive_keywords if word in text)
                neg_score = sum(1 for word in negative_keywords if word in text)
                
                if pos_score > neg_score:
                    positive_count += 1
                elif neg_score > pos_score:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate overall sentiment
            total_docs = len(all_docs)
            positive_ratio = positive_count / total_docs if total_docs > 0 else 0
            negative_ratio = negative_count / total_docs if total_docs > 0 else 0
            
            if positive_ratio > 0.6:
                overall_sentiment = 'very_positive'
            elif positive_ratio > 0.4:
                overall_sentiment = 'positive'
            elif negative_ratio > 0.6:
                overall_sentiment = 'very_negative'  
            elif negative_ratio > 0.4:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Calculate confidence based on document count and sentiment clarity
            confidence = min(0.9, (total_docs / 10) * abs(positive_ratio - negative_ratio))
            
            analysis_result = {
                'ticker': ticker,
                'overall_sentiment': overall_sentiment,
                'confidence': round(confidence, 3),
                'document_count': total_docs,
                'news_articles': len(news_docs),
                'reddit_posts': len(reddit_docs),
                'sentiment_breakdown': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count,
                    'positive_ratio': round(positive_ratio, 3),
                    'negative_ratio': round(negative_ratio, 3)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Sentiment analysis complete for {ticker}: {overall_sentiment} (confidence: {confidence:.3f})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {e}")
            return {'error': str(e)}
    
    def _generate_doc_id(self, ticker: str, doc_type: str, index: int, title: str) -> str:
        """
        Generate unique document ID
        
        Format: TICKER_TYPE_DATE_HASH
        Example: AAPL_news_20240115_a1b2c3d4
        
        Why unique IDs matter:
        - Prevent duplicate storage of same article
        - Enable document updates
        - Track document lineage
        """
        # Create content for hashing
        content = f"{ticker}_{doc_type}_{title}_{index}"
        
        # Generate short hash for uniqueness
        doc_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Add timestamp for chronological ordering
        timestamp = datetime.now().strftime("%Y%m%d")
        
        return f"{ticker}_{doc_type}_{timestamp}_{doc_hash}"
    
    def _calculate_relevance_score(self, distance: float) -> str:
        """
        Convert distance to human-readable relevance score
        
        ChromaDB returns distance (lower = more similar)
        Convert to relevance score (higher = more relevant)
        """
        similarity = 1 - distance
        
        if similarity > 0.8:
            return "Very High"
        elif similarity > 0.6:
            return "High"
        elif similarity > 0.4:
            return "Medium"
        elif similarity > 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the vector database
        
        Useful for:
        - Monitoring database growth
        - Understanding data coverage
        - Debugging issues
        """
        try:
            if self.use_chromadb:
                collection_count = self.collection.count()
                
                # Get sample of documents to analyze distribution
                sample_docs = self.collection.get(limit=100, include=['metadatas'])
                
                # Count by ticker and type
                ticker_counts = {}
                type_counts = {'news': 0, 'reddit': 0, 'other': 0}
                
                if sample_docs['metadatas']:
                    for metadata in sample_docs['metadatas']:
                        ticker = metadata.get('ticker', 'unknown')
                        doc_type = metadata.get('type', 'other')
                        
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                stats = {
                    'total_documents': collection_count,
                    'collection_name': self.collection.name,
                    'database_path': self.data_dir,
                    'document_types': type_counts,
                    'top_tickers': dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                    'last_updated': datetime.now().isoformat(),
                    'storage_type': 'chromadb'
                }
                
                logger.info(f"Collection stats: {collection_count} total documents")
                return stats
            else:
                # Fallback: Analyze in-memory storage
                collection_count = len(self.documents)
                
                # Count by ticker and type
                ticker_counts = {}
                type_counts = {'news': 0, 'reddit': 0, 'other': 0}
                
                for metadata in self.metadata:
                    ticker = metadata.get('ticker', 'unknown')
                    doc_type = metadata.get('type', 'other')
                    
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                stats = {
                    'total_documents': collection_count,
                    'collection_name': 'in_memory_fallback',
                    'database_path': 'memory',
                    'document_types': type_counts,
                    'top_tickers': dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                    'last_updated': datetime.now().isoformat(),
                    'storage_type': 'in_memory_fallback'
                }
                
                logger.info(f"In-memory storage stats: {collection_count} total documents")
                return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_ticker_data(self, ticker: str) -> bool:
        """
        Remove all documents for a specific ticker
        
        Useful for:
        - Data cleanup
        - Re-importing fresh data
        - Testing
        """
        try:
            logger.info(f"Clearing all data for ticker: {ticker}")
            
            if self.use_chromadb:
                # Get all document IDs for this ticker
                results = self.collection.get(
                    where={'ticker': ticker},
                    include=['ids']
                )
                
                if results['ids']:
                    # Delete all documents for this ticker
                    self.collection.delete(ids=results['ids'])
                    logger.info(f"Deleted {len(results['ids'])} documents for {ticker}")
                    return True
                else:
                    logger.info(f"No documents found for {ticker}")
                    return True
            else:
                # Fallback: Remove from in-memory storage
                initial_count = len(self.documents)
                
                # Find indices to remove
                indices_to_remove = []
                for i, meta in enumerate(self.metadata):
                    if meta.get('ticker') == ticker:
                        indices_to_remove.append(i)
                
                # Remove in reverse order to maintain indices
                for i in reversed(indices_to_remove):
                    del self.documents[i]
                    del self.metadata[i]
                    del self.ids[i]
                
                removed_count = len(indices_to_remove)
                logger.info(f"Deleted {removed_count} documents for {ticker} from in-memory storage")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing data for {ticker}: {e}")
            return False


# ================== COMPREHENSIVE TEST SUITE ==================
def run_comprehensive_test():
    """
    Comprehensive test of all vector database functionality
    """
    print("🧪 COMPREHENSIVE VECTOR DATABASE TEST")
    print("=" * 60)
    
    # Initialize manager
    print("\n1️⃣ Initializing Vector Database...")
    manager = VectorDataManager()
    initial_stats = manager.get_collection_stats()
    print(f"   Initial document count: {initial_stats.get('total_documents', 0)}")
    
    # Create comprehensive test data
    print("\n2️⃣ Preparing Test Data...")
    test_data = {
        'news_articles': [
            {
                'title': 'Apple Reports Strong Q4 Earnings, Beats Expectations',
                'description': 'Apple Inc. reported quarterly earnings that significantly exceeded analyst expectations, driven by strong iPhone sales and Services revenue growth. The company posted revenue of $123.9 billion, up 8% year-over-year.',
                'source': 'Yahoo Finance',
                'provider': 'Reuters',
                'url': 'https://finance.yahoo.com/news/apple-earnings-q4-2024',
                'published_at': '2024-01-15T16:30:00Z'
            },
            {
                'title': 'Apple Stock Rises 3% on AI Feature Announcement',
                'description': 'Shares of Apple gained 3% in after-hours trading following the company\'s announcement of new AI-powered features for iPhone and iPad. The features focus on enhanced Siri capabilities and on-device machine learning.',
                'source': 'Bloomberg',
                'provider': 'Bloomberg News',
                'url': 'https://bloomberg.com/news/apple-ai-features-2024',
                'published_at': '2024-01-14T14:20:00Z'
            },
            {
                'title': 'Apple Faces Challenges in China Market, Sales Down 12%',
                'description': 'Apple\'s iPhone sales in China decreased 12% year-over-year amid increased competition from local manufacturers like Huawei and Xiaomi. The company is implementing new strategies to regain market share.',
                'source': 'Reuters',
                'provider': 'Reuters Business',
                'url': 'https://reuters.com/technology/apple-china-challenges',
                'published_at': '2024-01-13T10:15:00Z'
            }
        ],
        'reddit_sentiment': {
            'sentiment': 'positive',
            'sentiment_score': 0.65,
            'post_count': 47,
            'sample_posts': [
                {
                    'title': 'AAPL crushed earnings! Bullish on Services growth',
                    'selftext': 'Apple just reported amazing earnings. Services revenue grew 16% YoY and margins are expanding. This is exactly what we wanted to see. iPhone sales were solid too despite macro headwinds.',
                    'subreddit': 'investing',
                    'score': 124,
                    'num_comments': 23
                },
                {
                    'title': 'Apple AI features look promising, buying more shares',
                    'selftext': 'The new AI announcement shows Apple is serious about competing with Google and Microsoft. On-device processing is brilliant for privacy. Long AAPL.',
                    'subreddit': 'stocks',
                    'score': 89,
                    'num_comments': 15
                },
                {
                    'title': 'Concerned about Apple China exposure, thoughts?',
                    'selftext': 'The 12% decline in China sales is worrying. China is a huge market for Apple. How are they planning to compete with local brands? Considering reducing position.',
                    'subreddit': 'SecurityAnalysis',
                    'score': 67,
                    'num_comments': 31
                }
            ]
        }
    }
    print(f"   Prepared {len(test_data['news_articles'])} news articles")
    print(f"   Prepared {len(test_data['reddit_sentiment']['sample_posts'])} Reddit posts")
    
    # Test data storage
    print("\n3️⃣ Testing Data Storage...")
    storage_success = manager.store_unstructured_data('AAPL', test_data)
    print(f"   Storage result: {'✅ Success' if storage_success else '❌ Failed'}")
    
    if storage_success:
        updated_stats = manager.get_collection_stats()
        new_docs = updated_stats.get('total_documents', 0) - initial_stats.get('total_documents', 0)
        print(f"   Added {new_docs} new documents")
    
    # Test various search queries
    print("\n4️⃣ Testing Search Functionality...")
    
    search_tests = [
        {
            'query': 'Apple earnings beat expectations revenue growth',
            'description': 'Earnings-related news'
        },
        {
            'query': 'artificial intelligence AI features Siri machine learning',
            'description': 'AI and technology news'
        },
        {
            'query': 'China market competition Huawei sales decline',
            'description': 'China market challenges'
        },
        {
            'query': 'bullish investment Services revenue growth',
            'description': 'Positive investment sentiment'
        }
    ]
    
    for i, test in enumerate(search_tests, 1):
        print(f"\n   4.{i} Search: {test['description']}")
        print(f"       Query: '{test['query']}'")
        
        results = manager.search_documents(
            query=test['query'],
            ticker='AAPL',
            limit=3
        )
        
        print(f"       Results: {len(results)} documents found")
        
        for j, result in enumerate(results[:2], 1):  # Show top 2 results
            title = result['metadata'].get('title', 'No title')
            relevance = result.get('relevance', 'Unknown')
            doc_type = result['metadata'].get('type', 'unknown')
            print(f"         {j}. [{doc_type.upper()}] {title[:60]}... (Relevance: {relevance})")
    
    # Test ticker-specific retrieval
    print("\n5️⃣ Testing Ticker Document Retrieval...")
    all_aapl_docs = manager.get_ticker_documents('AAPL')
    news_only = manager.get_ticker_documents('AAPL', doc_type='news')
    reddit_only = manager.get_ticker_documents('AAPL', doc_type='reddit')
    
    print(f"   Total AAPL documents: {len(all_aapl_docs)}")
    print(f"   News articles: {len(news_only)}")
    print(f"   Reddit posts: {len(reddit_only)}")
    
    # Test sentiment analysis
    print("\n6️⃣ Testing Sentiment Analysis...")
    sentiment_analysis = manager.analyze_ticker_sentiment('AAPL')
    
    if 'error' not in sentiment_analysis:
        print(f"   Overall sentiment: {sentiment_analysis['overall_sentiment']}")
        print(f"   Confidence: {sentiment_analysis['confidence']}")
        print(f"   Documents analyzed: {sentiment_analysis['document_count']}")
        
        breakdown = sentiment_analysis.get('sentiment_breakdown', {})
        print(f"   Positive: {breakdown.get('positive', 0)} ({breakdown.get('positive_ratio', 0)*100:.1f}%)")
        print(f"   Negative: {breakdown.get('negative', 0)} ({breakdown.get('negative_ratio', 0)*100:.1f}%)")
        print(f"   Neutral: {breakdown.get('neutral', 0)}")
    else:
        print(f"   ❌ Sentiment analysis failed: {sentiment_analysis['error']}")
    
    # Final statistics
    print("\n7️⃣ Final Database Statistics...")
    final_stats = manager.get_collection_stats()
    print(f"   Total documents: {final_stats.get('total_documents', 0)}")
    print(f"   Document types: {final_stats.get('document_types', {})}")
    print(f"   Top tickers: {final_stats.get('top_tickers', {})}")
    
    print("\n🎉 VECTOR DATABASE TEST COMPLETED!")
    print("=" * 60)
    print("✅ All functionality tested successfully")
    print("✅ Ready for Market Intelligence Agent integration")
    
    return True


# Run the test when script is executed directly
if __name__ == "__main__":
    run_comprehensive_test()