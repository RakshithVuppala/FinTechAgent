# Deployment Guide

This document provides comprehensive deployment instructions for the AI-Powered Financial Research Agent.

## 🚀 Quick Deployment Options

### Option 1: Quick Start (Recommended for Development)

```bash
# Clone and run in one command
git clone https://github.com/your-username/FinTechAgent.git
cd FinTechAgent
python run.py
```

### Option 2: Docker Deployment (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/your-username/FinTechAgent.git
cd FinTechAgent

# Deploy with Docker
./scripts/deploy.sh
```

### Option 3: Manual Setup

```bash
# Clone the repository
git clone https://github.com/your-username/FinTechAgent.git
cd FinTechAgent

# Set up environment
python scripts/setup.py

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run the application
streamlit run src/streamlit_dashboard.py
```

## 📋 Pre-Deployment Checklist

### System Requirements

- [ ] Python 3.9 or higher
- [ ] 4GB RAM minimum (8GB recommended)
- [ ] 2GB disk space
- [ ] Internet connection for data fetching

### Optional Requirements (for enhanced features)

- [ ] Docker & Docker Compose (for containerized deployment)
- [ ] GitHub AI API key (for enhanced LLM features)
- [ ] Reddit API credentials (for sentiment analysis)
- [ ] Alpha Vantage API key (for additional financial data)

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `.env.example` and configure:

```bash
# Required for enhanced AI features (optional)
GITHUB_AI_API_KEY=your_github_ai_key_here

# Optional API keys for additional features
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Application settings
CACHE_DURATION_MINUTES=30
STREAMLIT_SERVER_PORT=8501
```

### API Key Setup

#### GitHub AI API (Recommended)
1. Visit [GitHub Models](https://docs.github.com/en/github-models)
2. Generate an API key
3. Add to `.env` file: `GITHUB_AI_API_KEY=your_key`

#### Reddit API (Optional)
1. Visit [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create a new application
3. Add credentials to `.env` file

#### Alpha Vantage API (Optional)
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Get free API key
3. Add to `.env` file: `ALPHA_VANTAGE_API_KEY=your_key`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker directly

```bash
# Build image
docker build -t fintech-agent .

# Run container
docker run -d \
  --name fintech-agent \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  fintech-agent
```

## ☁️ Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - t2.medium or larger
   - Open port 8501 in security group

2. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker ubuntu
   ```

3. **Deploy Application**
   ```bash
   git clone https://github.com/your-username/FinTechAgent.git
   cd FinTechAgent
   ./scripts/deploy.sh
   ```

### Google Cloud Platform

1. **Create Compute Engine Instance**
   - Choose e2-standard-2 or larger
   - Allow HTTP/HTTPS traffic

2. **Deploy with Cloud Shell**
   ```bash
   git clone https://github.com/your-username/FinTechAgent.git
   cd FinTechAgent
   ./scripts/deploy.sh
   ```

### Azure Container Instances

```bash
# Create resource group
az group create --name fintech-agent --location eastus

# Deploy container
az container create \
  --resource-group fintech-agent \
  --name fintech-agent \
  --image your-registry/fintech-agent:latest \
  --ports 8501 \
  --dns-name-label fintech-agent
```

### Heroku

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Add buildpack and deploy**
   ```bash
   heroku buildpacks:set heroku/python
   git push heroku main
   ```

3. **Set environment variables**
   ```bash
   heroku config:set GITHUB_AI_API_KEY=your_key
   ```

## 🔍 Health Monitoring

### Health Check Endpoints

- Application health: `http://localhost:8501/_stcore/health`
- Custom health check: `http://localhost:8501/health` (if implemented)

### Monitoring Commands

```bash
# Check container status
docker ps

# View application logs
docker logs fintech-agent

# Monitor resource usage
docker stats fintech-agent

# Check system health
curl http://localhost:8501/_stcore/health
```

## 🔒 Security Considerations

### Production Security

- [ ] Use HTTPS in production
- [ ] Set up firewall rules
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Monitor access logs

### API Key Security

- [ ] Never commit API keys to version control
- [ ] Use environment variables only
- [ ] Rotate keys regularly
- [ ] Monitor API usage

## 🚨 Troubleshooting

### Common Issues

**Issue: Application won't start**
```bash
# Check logs
docker logs fintech-agent

# Verify dependencies
python -c "import streamlit; print('OK')"
```

**Issue: Ticker validation errors**
```bash
# Test network connectivity
curl -I https://query1.finance.yahoo.com/v1/finance/search?q=AAPL
```

**Issue: Out of memory errors**
```bash
# Check memory usage
docker stats fintech-agent

# Increase memory limit
docker run --memory=4g fintech-agent
```

### Performance Optimization

1. **Enable Caching**
   ```bash
   export ENABLE_CACHE=true
   export CACHE_DURATION_MINUTES=60
   ```

2. **Optimize Memory Usage**
   - Use Docker memory limits
   - Monitor vector database size
   - Clear old cache entries

3. **Database Optimization**
   - Regular vector database cleanup
   - Monitor disk usage in `/data` directory

## 📊 Production Monitoring

### Metrics to Monitor

- Application response time
- Memory usage
- API call volume
- Error rates
- Cache hit rates

### Monitoring Setup

```bash
# Set up log aggregation
docker run -d \
  --name log-aggregator \
  -v /var/log:/var/log \
  logstash:latest

# Monitor with Prometheus (optional)
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  prom/prometheus
```

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./scripts/deploy.sh restart
```

### Database Maintenance

```bash
# Clean old data (optional)
docker exec fintech-agent rm -rf /app/data/vector_db/*
docker exec fintech-agent rm -rf /app/data/structured/*

# Restart to reinitialize
docker-compose restart
```

### Backup Strategy

```bash
# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Backup configuration
cp .env .env.backup
```

## 📞 Support

For deployment issues:

1. Check the [GitHub Issues](https://github.com/your-username/FinTechAgent/issues)
2. Review the logs: `docker logs fintech-agent`
3. Test basic functionality: `python -c "import streamlit"`
4. Contact support with specific error messages

## 🎯 Production Best Practices

- Use Docker for consistent deployments
- Set up monitoring and alerting
- Regular backups of configuration
- Use environment-specific `.env` files
- Implement proper logging
- Set up SSL/TLS certificates
- Monitor resource usage
- Plan for scaling

---

**Happy Deploying! 🚀**