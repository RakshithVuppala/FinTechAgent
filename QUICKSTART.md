# 🚀 Quick Start Guide

Get up and running with the AI-Powered Financial Research Agent in under 5 minutes!

## 📦 One-Command Setup

```bash
git clone https://github.com/your-username/FinTechAgent.git
cd FinTechAgent
python run.py
```

That's it! The application will:
- ✅ Check Python compatibility  
- ✅ Install dependencies automatically
- ✅ Set up environment files
- ✅ Launch the web interface

## 🌐 Access the Dashboard

Once started, open your browser to:
**http://localhost:8501**

## 🎯 First Analysis

1. **Enter a ticker**: Try `AAPL`, `MSFT`, or `GOOGL`
2. **Choose method**: Select "Orchestrator Agent" (recommended)
3. **Start research**: Click "🚀 Start AI Research"
4. **View results**: Get comprehensive analysis in 30-60 seconds

## 🔧 Optional: Add API Keys

Edit `.env` file for enhanced features:

```bash
# GitHub AI (Free tier available)
GITHUB_AI_API_KEY=your_key_here

# Reddit API (For sentiment analysis)  
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

## 🐳 Docker Option

For production deployment:

```bash
./scripts/deploy.sh
```

## 📋 What You Get

- **Real-time validation** of stock tickers
- **Comprehensive analysis** from multiple data sources
- **AI-powered insights** with investment recommendations  
- **Interactive charts** and performance metrics
- **Export capabilities** for reports
- **Professional interface** with system monitoring

## 🆘 Need Help?

- **Common issues**: Check [DEPLOYMENT.md](DEPLOYMENT.md)
- **Detailed docs**: See [README.md](README.md)  
- **Bug reports**: [GitHub Issues](https://github.com/your-username/FinTechAgent/issues)

## 🎉 Example Results

```json
{
  "ticker": "AAPL",
  "recommendation": "BUY", 
  "score": 85.2,
  "execution_time": "8.2s",
  "key_strengths": ["Strong financials", "Positive sentiment"]
}
```

---

**Ready to analyze? Let's go! 🚀**