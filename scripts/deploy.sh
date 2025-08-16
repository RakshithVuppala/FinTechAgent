#!/bin/bash

# AI-Powered Financial Research Agent Deployment Script
# ====================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="fintech-agent"
DOCKER_IMAGE="fintech-agent:latest"
CONTAINER_NAME="fintech-agent"
PORT=8501

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_warning ".env file not found. Creating from template..."
        cp .env.example .env
        log_warning "Please edit .env file with your API keys before running the application."
    fi
    
    # Create necessary directories
    mkdir -p data/{raw,interim,processed,structured,vector_db} logs models
    
    log_success "Environment setup completed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE .
    
    log_success "Docker image built successfully"
}

# Deploy application
deploy_app() {
    log_info "Deploying application..."
    
    # Stop existing container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        log_info "Stopping existing container..."
        docker stop $CONTAINER_NAME
    fi
    
    # Remove existing container
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        log_info "Removing existing container..."
        docker rm $CONTAINER_NAME
    fi
    
    # Start new container
    docker-compose up -d
    
    log_success "Application deployed successfully"
}

# Check application health
check_health() {
    log_info "Checking application health..."
    
    # Wait for application to start
    sleep 10
    
    # Check if container is running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        log_success "Container is running"
        
        # Check if application is responding
        if curl -f http://localhost:$PORT/_stcore/health &> /dev/null; then
            log_success "Application is healthy and responding"
            log_info "Access the application at: http://localhost:$PORT"
        else
            log_warning "Application is starting up. Please wait a moment and try again."
        fi
    else
        log_error "Container is not running. Check logs with: docker logs $CONTAINER_NAME"
        exit 1
    fi
}

# Show logs
show_logs() {
    log_info "Showing application logs..."
    docker-compose logs -f
}

# Main deployment function
deploy() {
    echo "=================================="
    echo "  FinTech Agent Deployment Script"
    echo "=================================="
    echo
    
    check_prerequisites
    setup_environment
    build_image
    deploy_app
    check_health
    
    echo
    log_success "Deployment completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Edit .env file with your API keys (optional for basic functionality)"
    echo "2. Access the application at: http://localhost:$PORT"
    echo "3. View logs with: docker-compose logs -f"
    echo "4. Stop the application with: docker-compose down"
    echo
}

# Script options
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        log_info "Stopping application..."
        docker-compose down
        log_success "Application stopped"
        ;;
    "restart")
        log_info "Restarting application..."
        docker-compose restart
        log_success "Application restarted"
        ;;
    "build")
        build_image
        ;;
    "health")
        check_health
        ;;
    *)
        echo "Usage: $0 {deploy|logs|stop|restart|build|health}"
        echo
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  logs    - Show application logs"
        echo "  stop    - Stop the application"
        echo "  restart - Restart the application"
        echo "  build   - Build Docker image only"
        echo "  health  - Check application health"
        exit 1
        ;;
esac