# Quantum Hardware Integration Setup Guide

This guide will help you set up real quantum hardware integration for your Quantum-Classical ML platform.

## 🎯 Overview

Your platform now supports connecting to real quantum computers from major providers:
- **IBM Quantum** - Access to IBM's quantum processors
- **IonQ** (via AWS Braket) - Trapped ion quantum computers
- **Rigetti** (via AWS Braket) - Superconducting quantum processors

## 🔧 Prerequisites

### 1. Install Additional Dependencies

```bash
# Navigate to backend directory
cd backend

# Install quantum hardware dependencies
pip install qiskit-ibm-provider==0.7.0
pip install amazon-braket-sdk==1.73.0
pip install boto3==1.34.0
```

### 2. Set Up Provider Accounts

#### IBM Quantum Setup

1. **Create IBM Quantum Account**
   - Visit [IBM Quantum](https://quantum-computing.ibm.com/)
   - Sign up for a free account
   - Navigate to "My Account" → "API Token"
   - Copy your API token

2. **Configure IBM Quantum Credentials**
   ```bash
   # Option 1: Environment variable
   export IBMQ_TOKEN="your_ibm_quantum_token_here"
   
   # Option 2: Save to Qiskit (recommended)
   python -c "from qiskit import IBMQ; IBMQ.save_account('your_token_here')"
   ```

#### AWS Braket Setup (for IonQ and Rigetti)

1. **Create AWS Account**
   - Visit [AWS Console](https://aws.amazon.com/)
   - Sign up for an AWS account
   - Enable AWS Braket service

2. **Configure AWS Credentials**
   ```bash
   # Install AWS CLI
   pip install awscli
   
   # Configure credentials
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Enter your preferred region (e.g., us-east-1)
   # Enter output format (json)
   
   # Or set environment variables
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

3. **Enable Braket Service**
   - Go to AWS Braket console
   - Accept terms and conditions
   - Note: Quantum hardware access requires payment

## 💰 Cost Considerations

### IBM Quantum
- **Free Tier**: Limited access to simulators and some quantum processors
- **Premium**: Pay-per-use pricing for advanced quantum systems
- **Typical Cost**: ~$0.00015 per shot on quantum hardware

### AWS Braket (IonQ/Rigetti)
- **IonQ**: ~$0.01 per shot
- **Rigetti**: ~$0.00035 per shot
- **Additional AWS charges**: Data transfer, storage

### Cost Estimation
The platform includes built-in cost estimation:
```python
# Example API call
POST /quantum_hardware/estimate_cost
{
  "provider": "ibm",
  "shots": 1000,
  "num_qubits": 2
}
```

## 🚀 Getting Started

### 1. Start the Platform
```bash
# Start the complete platform
./start.sh

# Or manually
cd backend && python main.py &
cd .. && npm start
```

### 2. Access Quantum Hardware Manager
1. Navigate to the **Advanced** page
2. Scroll to the **Quantum Hardware Manager** section
3. Check provider connection status
4. Connect to your preferred provider

### 3. Test Connection
```bash
# Test IBM Quantum connection
curl -X POST "http://localhost:8000/quantum_hardware/connect?provider=ibm"

# Check hardware status
curl "http://localhost:8000/quantum_hardware/status"
```

## 🛡️ Error Mitigation Features

### Available Techniques
1. **Measurement Error Mitigation**
   - Corrects readout errors using calibration matrices
   - Automatically characterizes device noise

2. **Zero-Noise Extrapolation**
   - Extrapolates results to zero-noise limit
   - Supports linear, exponential, and polynomial extrapolation

3. **Dynamical Decoupling**
   - Inserts pulse sequences to mitigate decoherence
   - Supports X, XY4, and CPMG sequences

### Usage Example
```python
# Apply error mitigation
POST /quantum_error_mitigation/apply
{
  "counts": {"00": 480, "01": 120, "10": 150, "11": 250},
  "mitigation_techniques": ["measurement_error", "zero_noise_extrapolation"],
  "calibration_data": {
    "calibration_matrix": [[0.95, 0.05], [0.03, 0.97]]
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### 1. IBM Quantum Connection Failed
```bash
# Check token validity
python -c "from qiskit import IBMQ; IBMQ.load_account(); print('Token valid')"

# Re-save token if needed
python -c "from qiskit import IBMQ; IBMQ.save_account('your_token', overwrite=True)"
```

#### 2. AWS Braket Access Denied
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check Braket permissions
aws braket get-device --device-arn "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
```

#### 3. High Costs
- Use simulators for development and testing
- Estimate costs before running on hardware
- Start with small shot counts (100-1000)
- Use the built-in cost estimation feature

### Debug Mode
Enable debug logging in the backend:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Monitoring and Analytics

### Hardware Execution Metrics
- **Queue Time**: Time waiting in quantum processor queue
- **Execution Time**: Actual quantum circuit execution time
- **Success Rate**: Percentage of successful job completions
- **Cost Tracking**: Real-time cost monitoring

### Error Mitigation Effectiveness
- **Entropy Reduction**: Measure of noise reduction
- **Fidelity Improvement**: Quantum state fidelity enhancement
- **Confidence Intervals**: Statistical confidence in results

## 🔮 Next Steps

### Advanced Features Coming Soon
1. **Real Quantum Hardware Integration** ✅ **COMPLETED**
2. **Quantum Error Mitigation Suite** ✅ **COMPLETED**
3. **Advanced Quantum Algorithms** (QAOA, VQE, QNN)
4. **Quantum-Classical Hybrid Optimization**
5. **Quantum Federated Learning**

### Research Applications
- **Quantum Machine Learning Research**
- **NISQ Algorithm Development**
- **Quantum Advantage Studies**
- **Error Mitigation Research**

## 📚 Additional Resources

### Documentation
- [IBM Qiskit Documentation](https://qiskit.org/documentation/)
- [AWS Braket Documentation](https://docs.aws.amazon.com/braket/)
- [Quantum Error Mitigation Guide](https://qiskit.org/textbook/ch-quantum-hardware/error-mitigation.html)

### Tutorials
- [Getting Started with IBM Quantum](https://qiskit.org/textbook/ch-prerequisites/setting-the-environment.html)
- [AWS Braket Getting Started](https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started.html)

### Community
- [Qiskit Slack](https://qiskit.slack.com/)
- [AWS Quantum Computing Blog](https://aws.amazon.com/blogs/quantum-computing/)

## 🎉 Congratulations!

You now have access to real quantum hardware! Your platform can:
- Connect to multiple quantum providers
- Execute circuits on real quantum processors
- Apply advanced error mitigation techniques
- Estimate and track costs
- Monitor hardware performance

Start with simulators, then gradually move to quantum hardware as you become more comfortable with the platform.

**Happy Quantum Computing!** 🚀⚛️