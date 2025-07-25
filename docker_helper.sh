#!/bin/bash

# Plant Damage Estimation Docker Helper Script
# This script provides easy commands to work with the dockerized version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Check if NVIDIA Docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not available. GPU acceleration will not work."
        print_warning "Please install nvidia-docker2 for GPU support."
    else
        print_status "NVIDIA Docker runtime detected. GPU acceleration available."
    fi
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    mkdir -p inference_input inference_output
    mkdir -p saved/runs
    
    if [ ! -f "oracle.pth" ]; then
        print_warning "oracle.pth not found. Please download it from the Google Drive link in README."
        print_warning "The dummy dataset should work for testing, but you'll need the oracle for PxCL training."
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
}

# Show usage
show_usage() {
    echo "Plant Damage Estimation Docker Helper"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       - Setup directories and check requirements"
    echo "  build       - Build Docker image"
    echo "  train       - Run training"
    echo "  inference   - Run inference (requires model in saved/ directory)"
    echo "  tensorboard - Start TensorBoard server"
    echo "  shell       - Open interactive shell in container"
    echo "  clean       - Clean up Docker images and containers"
    echo "  logs        - Show container logs"
    echo ""
    echo "Examples:"
    echo "  $0 setup && $0 build && $0 train"
    echo "  $0 inference"
    echo "  $0 tensorboard"
}

# Main script logic
case "${1:-}" in
    "setup")
        check_docker
        check_nvidia_docker
        setup_directories
        print_status "Setup complete!"
        ;;
    "build")
        check_docker
        build_image
        ;;
    "train")
        check_docker
        print_status "Starting training..."
        docker-compose up plant-damage-estimation
        ;;
    "inference")
        check_docker
        if [ ! "$(ls -A inference_input 2>/dev/null)" ]; then
            print_error "inference_input directory is empty. Please add images for inference."
            exit 1
        fi
        print_status "Starting inference..."
        docker-compose --profile inference up plant-damage-inference
        ;;
    "tensorboard")
        check_docker
        print_status "Starting TensorBoard on http://localhost:6006"
        docker-compose --profile tensorboard up tensorboard
        ;;
    "shell")
        check_docker
        print_status "Opening interactive shell..."
        docker-compose run --rm plant-damage-estimation bash
        ;;
    "clean")
        check_docker
        print_status "Cleaning up Docker resources..."
        docker-compose down --rmi all --volumes --remove-orphans
        docker system prune -f
        ;;
    "logs")
        check_docker
        docker-compose logs -f
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
