# GitHub Actions CI/CD Pipeline

This directory contains the GitHub Actions workflows for the MLOps Heart Disease Prediction project.

## Workflows Overview

### 1. CI Pipeline (`ci.yml`)

**Triggers**: Push to main/develop, Pull Requests to main
**Purpose**: Continuous Integration with linting and testing

**Jobs**:

- **Lint**: Code quality checks with flake8, black, and isort
- **Test**: Comprehensive test suite on Python 3.11 and 3.12

**Artifacts Generated**:

- Test results and logs
- Model evaluation results
- Data processing outputs

### 2. Model Training Pipeline (`model-training.yml`)

**Triggers**: Manual dispatch, Weekly schedule (Monday 2 AM UTC)
**Purpose**: Complete model training and MLflow tracking

**Jobs**:

- **Train-and-Track**: Full pipeline execution
  - Data acquisition and EDA
  - Feature engineering and model training
  - MLflow experiment tracking
  - Model packaging for deployment
  - Pipeline validation

**Artifacts Generated**:

- Trained models
- Deployment packages
- Training logs and reports
- Environment snapshots

### 3. PR Validation (`pr-validation.yml`)

**Triggers**: Pull Request events
**Purpose**: Quick validation for pull requests

**Jobs**:

- **Quick-Validation**: Fast checks for PR approval
- **Security-Scan**: Security and code quality scanning

## Required Secrets

Configure these secrets in your GitHub repository:
`Settings → Secrets and variables → Actions`

| Secret Name           | Description               | Example Value                                            |
| --------------------- | ------------------------- | -------------------------------------------------------- |
| `MLFLOW_TRACKING_URI` | Railway MLflow server URL | `https://mlflow-tracking-production-53fb.up.railway.app` |

## Workflow Features

### Code Quality

- **Linting**: flake8 for Python code standards
- **Formatting**: black for consistent code formatting
- **Import Sorting**: isort for organized imports
- **Security Scanning**: GitHub Super Linter integration

### Testing Strategy

- **Multi-Python Support**: Tests on Python 3.11 and 3.12
- **Comprehensive Coverage**: All 4 tasks validated
- **MLflow Integration**: Railway server connectivity testing
- **Artifact Preservation**: Test results stored for 30 days

### Performance Optimizations

- **Dependency Caching**: pip cache for faster builds
- **Parallel Execution**: Matrix strategy for multiple Python versions
- **Conditional Steps**: Skip unnecessary steps on failure
- **Smart Triggers**: Different workflows for different events

## Usage Instructions

### Running CI Pipeline

The CI pipeline runs automatically on:

- Push to `main` or `develop` branches
- Pull requests to `main` branch

### Manual Model Training

To trigger model training manually:

1. Go to `Actions` tab in GitHub
2. Select `Model Training Pipeline`
3. Click `Run workflow`
4. Configure options:
   - **Retrain Models**: Force retrain all models
   - **Experiment Name**: Custom experiment name

### Monitoring Workflows

- **Status Badges**: Add to README for build status
- **Notifications**: Configure in repository settings
- **Logs**: View detailed logs in Actions tab
- **Artifacts**: Download generated files from workflow runs

## Troubleshooting

### Common Issues

**MLflow Connection Failures**:

- Verify `MLFLOW_TRACKING_URI` secret is set correctly
- Check Railway server status
- Ensure network connectivity

**Test Failures**:

- Check Python version compatibility
- Verify all dependencies are installed
- Review test logs for specific errors

**Linting Errors**:

- Run `black src/ tests/` locally to fix formatting
- Run `isort src/ tests/` to fix import sorting
- Run `flake8 src/ tests/` to check code quality

**Artifact Issues**:

- Ensure required directories exist
- Check file permissions
- Verify artifact paths in workflow

### Local Development

To run the same checks locally:

```bash
# Install development dependencies
pip install black isort flake8

# Run code formatting
black src/ tests/

# Sort imports
isort src/ tests/

# Check code quality
flake8 src/ tests/

# Run tests in CI mode
python run_tests.py --ci
```

## Workflow Customization

### Adding New Workflows

1. Create new `.yml` file in `.github/workflows/`
2. Define triggers, jobs, and steps
3. Add required secrets if needed
4. Test with a pull request

### Modifying Existing Workflows

1. Edit the appropriate `.yml` file
2. Test changes in a feature branch
3. Monitor workflow runs for issues
4. Update documentation as needed

### Environment Variables

Available in all workflows:

- `CI=true`: Indicates CI environment
- `PYTHONPATH`: Set to workspace root
- `MLFLOW_TRACKING_URI`: Railway MLflow server
- `GITHUB_*`: Standard GitHub Actions variables

## Best Practices

### Security

- Never commit secrets to repository
- Use GitHub Secrets for sensitive data
- Regularly rotate access tokens
- Review security scan results

### Performance

- Use caching for dependencies
- Minimize artifact sizes
- Optimize workflow triggers
- Use matrix strategies efficiently

### Maintenance

- Keep workflows updated with latest actions
- Monitor for deprecated features
- Review and update Python versions
- Maintain clear documentation

## Integration with MLOps Pipeline

The CI/CD workflows integrate seamlessly with the MLOps pipeline:

1. **Development**: Local development with immediate feedback
2. **Integration**: Automated testing on code changes
3. **Training**: Scheduled model retraining and tracking
4. **Deployment**: Artifact generation for production deployment
5. **Monitoring**: Continuous validation and quality assurance

This ensures a complete MLOps lifecycle with professional CI/CD practices.
