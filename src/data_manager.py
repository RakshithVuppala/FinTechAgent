"""
Data Manager - Handles storage and retrieval of structured financial data
Supports both file-based storage and in-memory fallback for deployment
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StructuredDataManager:
    """
    Storage manager for structured financial data
    Automatically falls back to in-memory storage if file system is read-only
    """

    def __init__(self, data_dir: str = "data/structured"):
        self.data_dir = data_dir
        self.use_file_storage = True
        self.memory_storage = {}  # Fallback in-memory storage
        
        # Try to create directory and test write access
        try:
            os.makedirs(data_dir, exist_ok=True)
            
            # Test write access with a temporary file
            test_file = os.path.join(data_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            logger.info(f"File storage initialized at {data_dir}")
            
        except (OSError, PermissionError) as e:
            logger.warning(f"File storage not available ({e}), using in-memory storage")
            self.use_file_storage = False

    def store_company_data(self, ticker: str, structured_data: Dict[str, Any]) -> bool:
        """Store structured data for a company"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Add storage metadata
            storage_data = {
                "ticker": ticker,
                "stored_at": datetime.now().isoformat(),
                "data": structured_data,
            }

            if self.use_file_storage:
                # File-based storage
                filename = f"{ticker}_{timestamp}.json"
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, "w") as f:
                    json.dump(storage_data, f, indent=2, default=str)
                
                logger.info(f"Stored structured data for {ticker} at {filepath}")
            else:
                # In-memory storage
                storage_key = f"{ticker}_{timestamp}"
                self.memory_storage[storage_key] = storage_data
                
                logger.info(f"Stored structured data for {ticker} in memory (key: {storage_key})")
            
            return True

        except Exception as e:
            logger.error(f"Error storing data for {ticker}: {e}")
            return False

    def get_latest_company_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the most recent structured data for a company"""
        try:
            if self.use_file_storage:
                # File-based retrieval
                ticker_files = [
                    f
                    for f in os.listdir(self.data_dir)
                    if f.startswith(f"{ticker}_") and f.endswith(".json")
                ]

                if not ticker_files:
                    logger.warning(f"No file data found for ticker {ticker}")
                    return None

                # Get the most recent file
                latest_file = sorted(ticker_files)[-1]
                filepath = os.path.join(self.data_dir, latest_file)

                with open(filepath, "r") as f:
                    stored_data = json.load(f)

                logger.info(f"Retrieved latest data for {ticker} from {latest_file}")
                return stored_data["data"]
            
            else:
                # In-memory retrieval
                ticker_keys = [
                    key for key in self.memory_storage.keys() 
                    if key.startswith(f"{ticker}_")
                ]
                
                if not ticker_keys:
                    logger.warning(f"No memory data found for ticker {ticker}")
                    return None
                
                # Get the most recent entry (keys are timestamped)
                latest_key = sorted(ticker_keys)[-1]
                stored_data = self.memory_storage[latest_key]
                
                logger.info(f"Retrieved latest data for {ticker} from memory (key: {latest_key})")
                return stored_data["data"]

        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {e}")
            return None

    def list_stored_companies(self) -> list:
        """Get list of all companies with stored data"""
        try:
            tickers = set()
            
            if self.use_file_storage:
                # File-based listing
                files = os.listdir(self.data_dir)
                for file in files:
                    if file.endswith(".json"):
                        ticker = file.split("_")[0]
                        tickers.add(ticker)
            else:
                # In-memory listing
                for key in self.memory_storage.keys():
                    ticker = key.split("_")[0]
                    tickers.add(ticker)

            return sorted(list(tickers))

        except Exception as e:
            logger.error(f"Error listing companies: {e}")
            return []
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about current storage method and data"""
        if self.use_file_storage:
            try:
                files = os.listdir(self.data_dir)
                file_count = len([f for f in files if f.endswith('.json')])
                return {
                    "storage_type": "file_system",
                    "location": self.data_dir,
                    "entry_count": file_count,
                    "status": "active"
                }
            except Exception as e:
                return {
                    "storage_type": "file_system",
                    "location": self.data_dir,
                    "entry_count": 0,
                    "status": f"error: {e}"
                }
        else:
            return {
                "storage_type": "in_memory",
                "location": "memory",
                "entry_count": len(self.memory_storage),
                "status": "active"
            }


if __name__ == "__main__":
    # Test the data manager
    print("🧪 Testing Structured Data Manager...")

    # Create test data (simulating what we'd get from data collector)
    test_data = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "timestamp": datetime.now().isoformat(),
        "financial_metrics": {
            "current_price": 150.00,
            "market_cap": 2500000000000,
            "pe_ratio": 25.5,
        },
    }

    # Initialize manager
    manager = StructuredDataManager()

    # Store data
    success = manager.store_company_data("AAPL", test_data)
    print(f"✅ Data storage: {'Success' if success else 'Failed'}")

    # Retrieve data
    retrieved_data = manager.get_latest_company_data("AAPL")
    print(f"✅ Data retrieval: {'Success' if retrieved_data else 'Failed'}")

    # List companies
    companies = manager.list_stored_companies()
    print(f"✅ Stored companies: {companies}")

    print("🎉 Data Manager test completed!")
