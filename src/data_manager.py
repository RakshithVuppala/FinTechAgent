"""
Data Manager - Handles storage and retrieval of structured financial data
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StructuredDataManager:
    """
    Simple storage manager for structured financial data
    Uses JSON files for now (can upgrade to database later)
    """

    def __init__(self, data_dir: str = "data/structured"):
        self.data_dir = data_dir
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def store_company_data(self, ticker: str, structured_data: Dict[str, Any]) -> bool:
        """Store structured data for a company"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)

            # Add storage metadata
            storage_data = {
                "ticker": ticker,
                "stored_at": datetime.now().isoformat(),
                "data": structured_data,
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(storage_data, f, indent=2, default=str)

            logger.info(f"Stored structured data for {ticker} at {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error storing data for {ticker}: {e}")
            return False

    def get_latest_company_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the most recent structured data for a company"""
        try:
            # Find all files for this ticker
            ticker_files = [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith(f"{ticker}_") and f.endswith(".json")
            ]

            if not ticker_files:
                logger.warning(f"No data found for ticker {ticker}")
                return None

            # Get the most recent file
            latest_file = sorted(ticker_files)[-1]
            filepath = os.path.join(self.data_dir, latest_file)

            # Load and return data
            with open(filepath, "r") as f:
                stored_data = json.load(f)

            logger.info(f"Retrieved latest data for {ticker} from {latest_file}")
            return stored_data["data"]

        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {e}")
            return None

    def list_stored_companies(self) -> list:
        """Get list of all companies with stored data"""
        try:
            files = os.listdir(self.data_dir)
            tickers = set()

            for file in files:
                if file.endswith(".json"):
                    ticker = file.split("_")[0]
                    tickers.add(ticker)

            return sorted(list(tickers))

        except Exception as e:
            logger.error(f"Error listing companies: {e}")
            return []


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
