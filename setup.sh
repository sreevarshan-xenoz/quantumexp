#!/bin/bash

# Quantum ML Simulation Setup Script
echo "🚀 Setting up Quantum ML Simulation..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Node.js and Python 3 are installed"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install --legacy-peer-deps

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start both frontend and backend:"
echo "  npm start"
echo ""
echo "To start only frontend:"
echo "  npm run start:frontend"
echo ""
echo "To start only backend:"
echo "  npm run start:backend"
echo ""
echo "Services will be available at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
