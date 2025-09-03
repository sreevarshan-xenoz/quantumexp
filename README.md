# Quantum-Classical ML Simulation Platform

A modern, interactive web application for comparing quantum and classical machine learning algorithms through comprehensive simulations and visualizations.

## 🚀 Features

### Quantum Hardware Integration ⚛️ **NEW!**
- **Real Quantum Computers**: Connect to IBM Quantum, IonQ, and Rigetti hardware
- **Error Mitigation Suite**: Advanced noise reduction and quantum error correction
- **Cost Estimation**: Built-in cost tracking and estimation for quantum hardware usage
- **Hardware Monitoring**: Real-time queue times, execution metrics, and device status

### Interactive Simulation Environment
- **Real-time Parameter Control**: Adjust dataset parameters, noise levels, and sample sizes with smooth sliders
- **Model Selection**: Choose from multiple quantum (VQC, QSVC) and classical (Logistic Regression, Random Forest, SVM, XGBoost) models
- **Quantum Feature Maps**: Support for ZZ, Z, and Pauli feature maps with customizable parameters

### Advanced Visualizations
- **Dataset Preview**: Live visualization of generated datasets with D3.js
- **Quantum Circuit Diagrams**: Interactive quantum circuit visualization
- **Decision Boundaries**: Real-time decision boundary plots for all models
- **Performance Metrics**: Comprehensive comparison charts and metrics

### Modern UI/UX
- **Dark/Light Mode**: Seamless theme switching with system preference detection
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Smooth Animations**: Micro-interactions and transitions for enhanced user experience
- **Progress Tracking**: Real-time simulation progress with detailed stage indicators

## 🛠️ Technology Stack

### Frontend
- **React 18**: Modern React with hooks and functional components
- **Tailwind CSS**: Utility-first CSS framework for rapid UI development
- **D3.js**: Data visualization library for interactive charts and plots
- **React Router**: Client-side routing for single-page application

### Quantum Computing (Backend Integration Ready)
- **Qiskit**: IBM's quantum computing framework
- **Qiskit Machine Learning**: Quantum machine learning algorithms
- **Variational Algorithms**: VQC and quantum kernel methods
- **Feature Maps**: Multiple quantum feature encoding strategies

### Classical ML
- **Scikit-learn**: Comprehensive classical ML algorithms
- **XGBoost**: Gradient boosting framework
- **NumPy/Pandas**: Data manipulation and numerical computing

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd quantumexp

# Run the setup script
./setup.sh

# Start both frontend and backend
npm start
```

### Option 2: Manual Setup
```bash
# Install Node.js dependencies
npm install --legacy-peer-deps

# Set up Python backend
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Start both services
npm start
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Available Scripts
```bash
npm start                    # Start both frontend and backend
npm run start:frontend       # Start only the React frontend
npm run start:backend        # Start only the FastAPI backend
npm run setup               # Install all dependencies
npm run dev                 # Alias for npm start
npm run build               # Build the React app for production
npm run test                # Run React tests
npm run lint                # Lint JavaScript/React code
npm run format              # Format code with Prettier
```

## 📦 Installation

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager
- Python 3.8+ (for quantum hardware integration)

### Quick Start
For basic simulation (no quantum hardware):
```bash
# Clone and start the platform
git clone <repository-url>
cd quantum-ml-platform
./start.sh
```

### Quantum Hardware Setup
For real quantum computer access, see our comprehensive setup guide:
📖 **[Quantum Hardware Setup Guide](QUANTUM_HARDWARE_SETUP.md)**

Includes setup for:
- IBM Quantum computers
- IonQ (via AWS Braket)
- Rigetti (via AWS Braket)
- Error mitigation configuration

### Frontend Setup
```bash
# Clone the repository
git clone <repository-url>
cd quantum-ml-simulation

# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:3000`

### Backend Setup (Optional - for full functionality)
```bash
# Install Python dependencies
pip install fastapi uvicorn scikit-learn qiskit qiskit-machine-learning qiskit-algorithms xgboost matplotlib

# Run the backend server
uvicorn main:app --reload
```

## 🎯 Usage Guide

### 1. Configure Dataset
- Select dataset type (Circles, Moons, or Blobs)
- Adjust noise level using the interactive slider
- Set sample size for the simulation

### 2. Choose Models
- **Quantum Models**: Select VQC or QSVC
- **Classical Models**: Choose from Logistic Regression, Random Forest, SVM, or XGBoost
- **Feature Maps**: Pick ZZ, Z, or Pauli feature maps for quantum models

### 3. Run Simulation
- Click "Run Simulation" to start the process
- Monitor real-time progress with detailed stage indicators
- View live updates of the simulation status

### 4. Analyze Results
- Compare accuracy, precision, recall, and F1 scores
- Examine training time differences
- View decision boundary visualizations
- Export results for further analysis

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
REACT_APP_API_URL=http://localhost:8000
```

### Tailwind Configuration
The project uses a custom Tailwind configuration with:
- Dark mode support
- Custom animations and transitions
- Extended color palette
- Responsive breakpoints

## 📊 Supported Algorithms

### Quantum Models
- **VQC (Variational Quantum Classifier)**: Parameterized quantum circuits for classification
- **QSVC (Quantum Support Vector Classifier)**: Quantum kernel-based classification

### Classical Models
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble method with decision trees
- **Support Vector Machine**: Kernel-based classification
- **XGBoost**: Gradient boosting framework

### Feature Maps
- **ZZ Feature Map**: Entangling feature map with ZZ interactions
- **Z Feature Map**: Non-entangling Z rotations
- **Pauli Feature Map**: Pauli X, Y, Z operations

## 🎨 UI Components

### Interactive Elements
- **Parameter Sliders**: Smooth, responsive sliders with real-time value display
- **Model Selectors**: Visual model selection with descriptions
- **Progress Indicators**: Animated progress bars with stage tracking
- **Theme Toggle**: Seamless dark/light mode switching

### Visualizations
- **Dataset Visualizer**: D3.js-powered scatter plots with animations
- **Quantum Circuit Visualizer**: Canvas-based quantum circuit diagrams
- **Results Dashboard**: Comprehensive performance comparison charts

## 🚀 Deployment

### Frontend Deployment
The application can be deployed to various platforms:

#### Vercel
```bash
npm run build
# Deploy to Vercel
```

#### Netlify
```bash
npm run build
# Deploy build folder to Netlify
```

#### GitHub Pages
```bash
npm install --save-dev gh-pages
npm run build
npm run deploy
```

### Backend Deployment
For production deployment of the Python backend:

#### Heroku
```bash
# Create Procfile
echo "web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}" > Procfile

# Deploy to Heroku
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

### Test Structure
- Component tests using React Testing Library
- Integration tests for API calls
- Visual regression tests for UI components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow React best practices and hooks patterns
- Use Tailwind CSS for styling
- Maintain responsive design principles
- Write comprehensive tests for new features
- Document complex algorithms and components

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qiskit Team**: For the excellent quantum computing framework
- **React Community**: For the robust frontend ecosystem
- **Tailwind CSS**: For the utility-first CSS framework
- **D3.js Community**: For powerful data visualization tools

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review existing discussions

---

**Built with ❤️ for the quantum computing and machine learning community**