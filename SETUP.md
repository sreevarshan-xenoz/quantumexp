# Setup Guide - Quantum-Classical ML Simulation Platform

This guide will help you set up and run the Quantum-Classical ML Simulation Platform on your local machine.

## 🎯 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/sreevarshan-xenoz/quantumexp.git
cd quantum-ml-simulation

# Run the automated setup script
./start.sh
```

### Option 2: Docker Setup
```bash
# Make sure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Start the application
docker-compose up --build
```

### Option 3: Manual Setup
Follow the detailed instructions below for manual setup.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space

### Software Dependencies

#### Frontend Requirements
- **Node.js**: Version 16 or higher
- **npm**: Version 8 or higher (comes with Node.js)

```bash
# Check Node.js version
node --version

# Check npm version
npm --version
```

#### Backend Requirements
- **Python**: Version 3.8 or higher
- **pip**: Python package manager

```bash
# Check Python version
python3 --version

# Check pip version
pip --version
```

## 🔧 Manual Installation

### 1. Frontend Setup

```bash
# Navigate to project root
cd quantum-ml-simulation

# Install Node.js dependencies
npm install

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000`

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`
API documentation will be available at `http://localhost:8000/docs`

## 🧪 Quantum Computing Setup (Optional)

For full quantum computing capabilities, install Qiskit:

```bash
# Install Qiskit and quantum ML packages
pip install qiskit qiskit-machine-learning qiskit-algorithms

# Verify installation
python -c "import qiskit; print(qiskit.__version__)"
```

**Note**: The application will work without Qiskit installed, using mock quantum models for demonstration.

## 🐳 Docker Setup (Alternative)

### Prerequisites
- Docker Desktop or Docker Engine
- Docker Compose

### Installation

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Docker Services
- **Frontend**: React development server on port 3000
- **Backend**: FastAPI server on port 8000

## 🔍 Verification

### 1. Check Frontend
Open `http://localhost:3000` in your browser. You should see:
- Modern, responsive interface
- Dark/light mode toggle
- Interactive parameter controls

### 2. Check Backend
Open `http://localhost:8000/docs` in your browser. You should see:
- FastAPI interactive documentation
- Available API endpoints
- Health check endpoint

### 3. Test API Connection
```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","qiskit_available":true,"xgboost_available":true}
```

## 🛠️ Troubleshooting

### Common Issues

#### Frontend Issues

**Issue**: `npm install` fails
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Issue**: Port 3000 already in use
```bash
# Find process using port 3000
lsof -ti:3000

# Kill the process
kill -9 $(lsof -ti:3000)

# Or use a different port
PORT=3001 npm start
```

#### Backend Issues

**Issue**: Python dependencies fail to install
```bash
# Update pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

**Issue**: Qiskit installation fails
```bash
# Install without quantum dependencies
pip install fastapi uvicorn scikit-learn matplotlib xgboost

# The app will run with mock quantum models
```

**Issue**: Port 8000 already in use
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use a different port
uvicorn main:app --reload --port 8001
```

#### Docker Issues

**Issue**: Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

**Issue**: Permission denied
```bash
# On Linux, add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in
```

### Performance Optimization

#### For Large Datasets
- Reduce sample size for faster processing
- Use fewer quantum circuit repetitions
- Consider using classical models only for initial testing

#### For Quantum Simulations
- Start with small datasets (< 1000 samples)
- Use 2-3 repetitions for feature maps
- Monitor memory usage during quantum training

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Frontend configuration
REACT_APP_API_URL=http://localhost:8000

# Backend configuration (optional)
PYTHONPATH=/app
LOG_LEVEL=INFO
```

### API Configuration

The frontend automatically detects the backend URL. If running on different ports:

```javascript
// src/api/simulation.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';
```

## 📊 Usage Examples

### Basic Simulation
1. Open `http://localhost:3000`
2. Select "Circles" dataset
3. Set noise level to 0.2
4. Choose "Logistic" for classical model
5. Choose "VQC" for quantum model
6. Click "Run Simulation"

### Advanced Configuration
1. Try different feature maps (ZZ, Z, Pauli)
2. Experiment with various classical models
3. Adjust sample sizes and noise levels
4. Compare hybrid model performance

## 🚀 Production Deployment

### Frontend Deployment (Vercel)
```bash
# Build for production
npm run build

# Deploy to Vercel
npx vercel --prod
```

### Backend Deployment (Heroku)
```bash
# Create Procfile
echo "web: uvicorn main:app --host=0.0.0.0 --port=\$PORT" > Procfile

# Deploy to Heroku
git add .
git commit -m "Deploy to production"
git push heroku main
```

## 📞 Support

If you encounter issues:

1. **Check the logs**: Look at browser console and terminal output
2. **Verify prerequisites**: Ensure all dependencies are installed
3. **Test components separately**: Try frontend and backend independently
4. **Check network connectivity**: Ensure ports are not blocked
5. **Review documentation**: Check API docs at `/docs` endpoint

## 🎉 Success!

Once everything is running, you should have:
- ✅ Frontend at `http://localhost:3000`
- ✅ Backend API at `http://localhost:8000`
- ✅ Interactive quantum-classical ML simulations
- ✅ Real-time visualizations and comparisons

Happy experimenting with quantum machine learning! 🚀