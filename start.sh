#!/bin/bash

# Quantum-Classical ML Simulation Platform Startup Script

echo "🚀 Starting Quantum-Classical ML Simulation Platform..."

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "🐳 Docker detected. Starting with Docker Compose..."
    docker-compose up --build
else
    echo "📦 Docker not found. Starting manually..."
    
    # Start backend
    echo "🔧 Starting backend server..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Start backend server in background
    echo "Starting FastAPI server..."
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    cd ..
    
    # Start frontend
    echo "🎨 Starting frontend server..."
    
    # Install Node.js dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    fi
    
    # Start frontend server
    echo "Starting React development server..."
    npm start &
    FRONTEND_PID=$!
    
    echo "✅ Application started successfully!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    
    # Wait for user input to stop
    echo ""
    echo "Press Ctrl+C to stop all servers..."
    
    # Function to cleanup on exit
    cleanup() {
        echo "🛑 Stopping servers..."
        kill $BACKEND_PID 2>/dev/null
        kill $FRONTEND_PID 2>/dev/null
        exit 0
    }
    
    # Set trap to cleanup on script exit
    trap cleanup SIGINT SIGTERM
    
    # Wait indefinitely
    wait
fi