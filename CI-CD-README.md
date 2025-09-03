# CI/CD Pipeline Documentation

This document describes the comprehensive CI/CD pipeline setup for the Quantum ML Simulation project.

## 🚀 Pipeline Overview

The CI/CD pipeline consists of multiple workflows that ensure code quality, security, and automated deployment:

### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)
- **Triggers**: Push to main/develop, Pull requests, Releases
- **Features**:
  - Multi-version testing (Python 3.9-3.11, Node.js 18-20)
  - Code quality checks (linting, formatting, type checking)
  - Security scanning with Trivy
  - Docker image building and pushing
  - Integration testing
  - Automated deployment to staging/production

### 2. Code Quality Pipeline (`.github/workflows/code-quality.yml`)
- **Triggers**: Pull requests, Push to main/develop
- **Features**:
  - Python code formatting (Black, isort)
  - Python linting (Flake8)
  - Python type checking (MyPy)
  - Python security scanning (Bandit)
  - JavaScript/React linting and formatting
  - Auto-formatting on pull requests

### 3. Docker Pipeline (`.github/workflows/docker.yml`)
- **Triggers**: Push to main/develop, Tags, Pull requests
- **Features**:
  - Multi-platform Docker image building
  - Container registry management
  - Security scanning of Docker images
  - Automated deployment with Docker images

### 4. Dependency Management (`.github/workflows/dependency-update.yml`)
- **Triggers**: Weekly schedule, Manual dispatch
- **Features**:
  - Automated dependency updates
  - Security vulnerability scanning
  - Pull request creation for updates

## 🛠️ Setup Instructions

### Prerequisites
1. GitHub repository with Actions enabled
2. GitHub Container Registry access
3. Environment secrets configured (if needed)

### Required Secrets
Configure these secrets in your GitHub repository settings:

```bash
# For container registry (automatically provided)
GITHUB_TOKEN

# For deployment (if using external services)
DEPLOY_TOKEN
KUBECONFIG
```

### Environment Setup
1. **Staging Environment**: Configure in GitHub repository settings
2. **Production Environment**: Configure with required approvals

## 📋 Pipeline Stages

### 1. Code Quality & Testing
```yaml
- Code formatting and linting
- Type checking
- Security scanning
- Unit tests with coverage
- Integration tests
```

### 2. Build & Package
```yaml
- Docker image building
- Multi-platform support
- Container registry push
- Image security scanning
```

### 3. Deploy
```yaml
- Staging deployment (develop branch)
- Production deployment (main branch)
- Health checks
- Rollback capabilities
```

## 🔧 Local Development

### Running Quality Checks Locally

#### Python
```bash
cd backend
pip install black flake8 isort mypy bandit
black .                    # Format code
flake8 .                   # Lint code
isort .                    # Sort imports
mypy .                     # Type check
bandit -r .                # Security scan
```

#### JavaScript/React
```bash
npm install
npm run lint               # Lint code
npm run format             # Format code
npm run format:check       # Check formatting
npm run test:coverage      # Run tests with coverage
```

### Docker Development
```bash
# Build and run locally
docker-compose up --build

# Run specific services
docker-compose up backend frontend
```

## 📊 Monitoring & Metrics

### Code Coverage
- Python: Uploaded to Codecov
- JavaScript: Uploaded to Codecov
- Minimum coverage thresholds can be configured

### Security Scanning
- Trivy for container vulnerabilities
- Bandit for Python security issues
- npm audit for JavaScript dependencies
- Results uploaded to GitHub Security tab

### Performance Monitoring
- Load testing with k6 (configurable)
- Performance regression detection
- Resource usage monitoring

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check dependency versions
   - Verify Docker build context
   - Review error logs in Actions

2. **Test Failures**
   - Update test cases for new features
   - Check environment variables
   - Verify test data

3. **Deployment Issues**
   - Check environment configuration
   - Verify secrets and permissions
   - Review deployment logs

### Debug Commands
```bash
# Check pipeline status
gh run list

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

## 🔄 Workflow Customization

### Adding New Environments
1. Create environment in GitHub settings
2. Add deployment job in workflow
3. Configure environment-specific variables

### Custom Quality Gates
1. Modify workflow conditions
2. Add custom checks
3. Configure approval requirements

### Notification Integration
- Slack notifications
- Discord webhooks
- Email alerts
- Custom webhook endpoints

## 📈 Best Practices

1. **Branch Protection**: Enable branch protection rules
2. **Required Checks**: Make CI checks required for merging
3. **Review Requirements**: Require code reviews
4. **Environment Approvals**: Use environment protection rules
5. **Secret Management**: Use GitHub Secrets for sensitive data
6. **Artifact Retention**: Configure appropriate retention policies

## 🔗 Related Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Security Best Practices](https://docs.github.com/en/code-security)
- [Deployment Strategies](https://docs.github.com/en/actions/deployment)

## 📞 Support

For issues with the CI/CD pipeline:
1. Check the Actions tab in GitHub
2. Review workflow logs
3. Create an issue with detailed error information
4. Contact the development team

---

**Last Updated**: $(date)
**Pipeline Version**: 1.0.0
