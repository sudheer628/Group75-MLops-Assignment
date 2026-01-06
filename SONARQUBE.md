# SonarCloud Integration

This document explains how we integrated SonarCloud for static code analysis in our project. SonarCloud automatically scans our code for bugs, vulnerabilities, code smells, and security hotspots on every push.

Dashboard: https://sonarcloud.io/project/overview?id=sudheer628_Group75-MLops-Assignment

---

## Why SonarCloud

We wanted automated code quality checks as part of our CI/CD pipeline. SonarCloud provides:

- Static code analysis for Python
- Security vulnerability detection
- Code smell identification
- Technical debt tracking
- Quality gate enforcement

---

## Free Plan Limitations

We're using the SonarCloud Free plan which works well for our public repository. Here are the key limits:

| Feature              | Free Plan Limit         |
| -------------------- | ----------------------- |
| Public projects      | Unlimited               |
| Private projects     | 50,000 LOC total        |
| Organization members | Maximum 5               |
| Branch analysis      | Main branch only        |
| PR analysis          | Only PRs to main branch |

Our project has about 4.4k lines of code, so we're well within the limits.

What's NOT included in Free plan:

- Advanced SAST (Security Analysis)
- Software Composition Analysis (SCA)
- Portfolio management
- Audit logs

---

## Setup Steps

### Step 1: Create SonarCloud Account

1. Go to https://sonarcloud.io
2. Click "Log in" and select "GitHub"
3. Authorize SonarCloud to access your GitHub account
4. Select the Free plan when prompted

### Step 2: Import Project

1. After login, click "Analyze new project" or "+" button
2. Select your GitHub organization/account
3. Find and select the repository: `Group75-MLops-Assignment`
4. Click "Set Up" to create the project
5. SonarCloud will automatically detect it's a Python project

### Step 3: Generate Token

1. Click on your profile avatar (top right corner)
2. Go to "My Account" -> "Security" tab
3. Under "Generate Tokens", enter a name: `github-actions`
4. Click "Generate"
5. Copy the token immediately (it won't be shown again)

### Step 4: Add GitHub Secret

1. Go to your GitHub repository
2. Navigate to Settings -> Secrets and variables -> Actions
3. Click "New repository secret"
4. Name: `SONAR_TOKEN`
5. Value: (paste the token from Step 3)
6. Click "Add secret"

### Step 5: Create Configuration File

Create `sonar-project.properties` in the repository root:

```properties
sonar.projectKey=sudheer628_Group75-MLops-Assignment
sonar.organization=sudheer628

sonar.projectName=Heart Disease Prediction MLOps
sonar.projectVersion=1.0

# Source directories
sonar.sources=src,app
sonar.tests=tests

# Python specific
sonar.python.version=3.12

# Exclusions
sonar.exclusions=**/node_modules/**,**/__pycache__/**,**/*.joblib,**/*.parquet

# Encoding
sonar.sourceEncoding=UTF-8
```

### Step 6: Add GitHub Actions Workflow

Add the SonarCloud job to `.github/workflows/ci.yml`:

```yaml
sonarcloud:
  name: SonarCloud Analysis
  runs-on: ubuntu-latest
  needs: [lint, feature-validation]

  steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

---

## How It Works

After setup, SonarCloud analysis runs automatically:

1. Push code to main/develop branch
2. CI pipeline runs lint and feature-validation jobs
3. SonarCloud job runs in parallel with unit tests
4. Results appear in SonarCloud dashboard

```
Push to main
     |
     v
+----------+     +-------------------+
|   Lint   | --> | Feature Validation|
+----------+     +-------------------+
                          |
          +---------------+---------------+
          |                               |
          v                               v
   +------------+                 +-------------+
   | Unit Tests |                 | SonarCloud  |
   +------------+                 +-------------+
```

---

## Viewing Results

### SonarCloud Dashboard

Go to: https://sonarcloud.io/project/overview?id=sudheer628_Group75-MLops-Assignment

The dashboard shows:

- Overall code quality rating (A to E)
- Bugs count
- Vulnerabilities count
- Code smells count
- Coverage percentage (if configured)
- Duplications percentage

### Quality Gate

SonarCloud has a "Quality Gate" that passes or fails based on:

- No new bugs
- No new vulnerabilities
- Code smells below threshold
- Coverage above threshold (if configured)

If the Quality Gate fails, you'll see it in the GitHub Actions check.

### Pull Request Analysis

When you create a PR to main branch:

1. SonarCloud analyzes only the changed code
2. Comments appear directly on the PR
3. Quality Gate status shows in PR checks

---

## Configuration Details

### Project Key and Organization

These values come from SonarCloud when you create the project:

- Organization: `sudheer628` (your GitHub username)
- Project Key: `sudheer628_Group75-MLops-Assignment`

### Source and Test Directories

We configured SonarCloud to analyze:

- Source code: `src/` and `app/` directories
- Test code: `tests/` directory

### Exclusions

We exclude files that shouldn't be analyzed:

- `__pycache__/` - Python bytecode
- `*.joblib` - Model files
- `*.parquet` - Data files
- `node_modules/` - Not applicable but good practice

---

## Troubleshooting

### "Project not found" error

Make sure the project key in `sonar-project.properties` matches exactly what's in SonarCloud.

### "Token invalid" error

1. Check that `SONAR_TOKEN` secret is set in GitHub
2. Verify the token hasn't expired
3. Generate a new token if needed

### Analysis not running

1. Check that the workflow file is correct
2. Verify the `needs` dependencies are satisfied
3. Look at GitHub Actions logs for errors

### Quality Gate not computed

This happens on the first analysis. Run a second analysis and it will show up.

---

## GitHub Secrets Summary

| Secret         | Purpose                               |
| -------------- | ------------------------------------- |
| `SONAR_TOKEN`  | Authentication for SonarCloud API     |
| `GITHUB_TOKEN` | Auto-provided, used for PR decoration |

---

## Links

- SonarCloud Dashboard: https://sonarcloud.io/project/overview?id=sudheer628_Group75-MLops-Assignment
- SonarCloud Documentation: https://docs.sonarcloud.io/
- GitHub Actions Integration: https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/github-actions-for-sonarcloud/
