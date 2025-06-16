#!/bin/bash

# 🔬 RAG Benchmark Runner
# Script untuk menjalankan benchmark lengkap Simple API vs Multi API

echo "🚀 RAG Benchmark Runner for Thesis Research"
echo "=============================================="

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Step 1: Check if Docker is running
echo "🔍 Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi
echo "✅ Docker is running"

# Step 2: Check current running containers
echo "🔍 Checking for existing containers..."
if check_port 8080; then
    echo "⚠️ Port 8080 is already in use (Multi API)"
    read -p "Do you want to stop existing containers? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping existing containers..."
        docker-compose down
        docker-compose -f docker-compose-simple.yml down
    else
        echo "❌ Cannot proceed with ports in use"
        exit 1
    fi
fi

# Step 3: Start Multi API (Enhanced Multi-Step RAG)
echo "🚀 Starting Multi API (Port 8080)..."
docker-compose up -d

if ! wait_for_service "http://localhost:8080" "Multi API"; then
    echo "❌ Failed to start Multi API"
    exit 1
fi

# Step 4: Start Simple API
echo "🚀 Starting Simple API (Port 8081)..."
docker-compose -f docker-compose-simple.yml up -d

if ! wait_for_service "http://localhost:8081" "Simple API"; then
    echo "❌ Failed to start Simple API"
    exit 1
fi

# Step 5: Check API endpoints
echo "🔍 Testing API endpoints..."

# Test Multi API
echo "Testing Multi API..."
response=$(curl -s -w "%{http_code}" -H "X-API-Key: your_secure_api_key_here" -H "Content-Type: application/json" -d '{"query":"Test query","embedding_model":"small"}' http://localhost:8080/api/chat)
http_code=${response: -3}
if [ "$http_code" = "200" ]; then
    echo "✅ Multi API is responding correctly"
else
    echo "⚠️ Multi API returned HTTP $http_code"
fi

# Test Simple API  
echo "Testing Simple API..."
response=$(curl -s -w "%{http_code}" -H "X-API-Key: your_secure_api_key_here" -H "Content-Type: application/json" -d '{"query":"Test query","embedding_model":"small"}' http://localhost:8081/api/chat)
http_code=${response: -3}
if [ "$http_code" = "200" ]; then
    echo "✅ Simple API is responding correctly"
else
    echo "⚠️ Simple API returned HTTP $http_code"
fi

# Step 6: Install Python dependencies for benchmark
echo "📦 Installing Python dependencies..."
pip install requests pandas pathlib > /dev/null 2>&1

# Step 7: Run benchmark
echo "🔬 Starting RAG Benchmark..."
echo "⏱️ This will take approximately 15-20 minutes"
echo "📊 Testing 12 questions × 4 combinations = 48 total tests"
echo ""

read -p "Do you want to proceed with the benchmark? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running benchmark..."
    python benchmarks/complete_benchmark.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 BENCHMARK COMPLETED SUCCESSFULLY!"
        echo "📁 Results saved in: benchmark_results/"
        echo ""
        echo "📋 Generated files:"
        ls -la benchmark_results/ | grep "$(date +%Y%m%d)"
        echo ""
        echo "📄 Key files for thesis:"
        echo "  - comprehensive_report_*.md  (Detailed analysis)"
        echo "  - latex_table_*.tex         (LaTeX table for thesis)" 
        echo "  - benchmark_data_*.csv      (Raw data for analysis)"
        echo "  - benchmark_summary_*.json  (Summary statistics)"
    else
        echo "❌ Benchmark failed"
    fi
else
    echo "⏹️ Benchmark cancelled"
fi

echo ""
echo "🧹 Cleanup: To stop all services, run:"
echo "  docker-compose down"
echo "  docker-compose -f docker-compose-simple.yml down" 