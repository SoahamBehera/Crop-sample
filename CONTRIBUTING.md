# 🤝 Contributing to CultivaSense

Thank you for your interest in contributing to **CultivaSense**! We're excited to have you join our mission to empower Indian agriculture with AI-powered solutions. This guide will help you get started with contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Coding Standards](#-coding-standards)
- [Commit Guidelines](#-commit-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Testing Guidelines](#-testing-guidelines)
- [Documentation](#-documentation)
- [Community](#-community)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to **dasouvik122005@gmail.com**.

---

## 🌟 How Can I Contribute?

There are many ways to contribute to CultivaSense:

### 1. **Report Bugs** 🐛

Found a bug? Help us fix it!

- **Check existing issues** to avoid duplicates
- **Use the bug report template** when creating an issue
- **Provide detailed information:**
  - Steps to reproduce
  - Expected vs actual behavior
  - Screenshots (if applicable)
  - Environment details (OS, Python version, browser)

### 2. **Suggest Features** 💡

Have an idea to improve CultivaSense?

- **Check the roadmap** in README.md
- **Open a feature request** with detailed description
- **Explain the use case** and benefits
- **Consider implementation** complexity

### 3. **Improve Documentation** 📚

Documentation is crucial for users and developers:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve code comments
- Translate documentation (future)

### 4. **Write Code** 💻

Contribute to the codebase:

- Fix bugs
- Implement new features
- Improve performance
- Enhance UI/UX
- Add tests

### 5. **Review Pull Requests** 👀

Help maintain code quality:

- Review open pull requests
- Test proposed changes
- Provide constructive feedback
- Suggest improvements

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Git** for version control
- **pip** package manager
- **Virtual environment** tool (venv)
- **Code editor** (VS Code, PyCharm, etc.)

### Fork and Clone

1. **Fork the repository** on GitHub
   - Click the "Fork" button at the top right

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cultiva-Sense.git
   cd Cultiva-Sense
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/SoahamBehera/Cultiva-Sense.git
   ```

4. **Verify remotes**
   ```bash
   git remote -v
   # origin    https://github.com/YOUR_USERNAME/Cultiva-Sense.git (fetch)
   # origin    https://github.com/YOUR_USERNAME/Cultiva-Sense.git (push)
   # upstream  https://github.com/SoahamBehera/Cultiva-Sense.git (fetch)
   # upstream  https://github.com/SoahamBehera/Cultiva-Sense.git (push)
   ```

### Set Up Development Environment

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Verify setup**
   - Open browser to `http://127.0.0.1:5000`
   - Test crop recommendation feature
   - Upload a test image for disease detection

---

## 🔄 Development Workflow

### 1. **Sync with Upstream**

Before starting work, sync your fork:

```bash
# Fetch upstream changes
git fetch upstream

# Checkout your main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main
```

### 2. **Create a Feature Branch**

Always create a new branch for your work:

```bash
# Create and checkout a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/crop-calendar` - New features
- `fix/disease-detection-bug` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/optimize-model` - Code refactoring
- `test/add-unit-tests` - Test additions

### 3. **Make Your Changes**

- Write clean, readable code
- Follow coding standards (see below)
- Add comments for complex logic
- Update documentation if needed
- Test your changes thoroughly

### 4. **Commit Your Changes**

Follow our commit message guidelines:

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add crop calendar feature"
```

### 5. **Push to Your Fork**

```bash
git push origin feature/your-feature-name
```

### 6. **Create Pull Request**

- Go to your fork on GitHub
- Click "Compare & pull request"
- Fill out the PR template
- Link related issues
- Request review

---

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some project-specific conventions:

#### 1. **Code Formatting**

```python
# ✅ Good - Clear, readable, PEP 8 compliant
def predict_crop(nitrogen, phosphorus, potassium, temperature, 
                 humidity, ph, rainfall):
    """
    Predict the best crop based on soil and environmental parameters.
    
    Args:
        nitrogen (float): Nitrogen content in soil (0-140 kg/ha)
        phosphorus (float): Phosphorus content (5-145 kg/ha)
        potassium (float): Potassium content (5-205 kg/ha)
        temperature (float): Temperature in Celsius
        humidity (float): Relative humidity (0-100%)
        ph (float): Soil pH level (3.5-9.5)
        rainfall (float): Annual rainfall in mm
    
    Returns:
        str: Recommended crop name
    """
    features = np.array([[nitrogen, phosphorus, potassium, 
                         temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    return prediction[0]


# ❌ Bad - Poor formatting, no documentation
def predict(n,p,k,t,h,ph,r):
    f=np.array([[n,p,k,t,h,ph,r]])
    return model.predict(f)[0]
```

#### 2. **Naming Conventions**

```python
# Variables and functions: snake_case
crop_name = "Rice"
def calculate_fertilizer_amount():
    pass

# Classes: PascalCase
class CropRecommendationModel:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_FILE_SIZE = 5242880
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Private methods: _leading_underscore
def _validate_input(value):
    pass
```

#### 3. **Imports**

```python
# Standard library imports first
import os
import sys
from datetime import datetime

# Third-party imports second
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Local imports last
from config import Config
from models.crop_model import CropPredictor
```

#### 4. **Documentation**

```python
def analyze_soil(n, p, k, ph):
    """
    Analyze soil nutrient levels and provide recommendations.
    
    This function compares user's soil values with ideal ranges
    and generates recovery plans for deficient nutrients.
    
    Args:
        n (float): Nitrogen level in kg/ha
        p (float): Phosphorus level in kg/ha
        k (float): Potassium level in kg/ha
        ph (float): Soil pH level
    
    Returns:
        dict: Analysis results containing:
            - status: 'optimal', 'deficient', or 'excess'
            - recommendations: List of recovery actions
            - ideal_range: Tuple of (min, max) ideal values
    
    Raises:
        ValueError: If any parameter is outside valid range
    
    Example:
        >>> analyze_soil(50, 30, 40, 6.5)
        {'status': 'optimal', 'recommendations': [], 'ideal_range': (40, 60)}
    """
    pass
```

### JavaScript Style Guide

```javascript
// Use const/let, not var
const MAX_FILE_SIZE = 5 * 1024 * 1024;
let uploadedFile = null;

// Function naming: camelCase
function validateFileSize(file) {
    return file.size <= MAX_FILE_SIZE;
}

// Clear, descriptive names
function showLoadingSpinner() {
    document.getElementById('loading-spinner').classList.remove('hidden');
}

// Use template literals
const message = `File ${fileName} is too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB`;
```

### CSS Style Guide

```css
/* Use meaningful class names */
.crop-recommendation-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
}

/* Group related properties */
.submit-button {
    /* Positioning */
    position: relative;
    
    /* Box model */
    padding: 1rem 2rem;
    margin: 1rem 0;
    
    /* Typography */
    font-size: 1.1rem;
    font-weight: 600;
    
    /* Visual */
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 12px;
    
    /* Animation */
    transition: all 0.3s ease;
}

/* Use CSS custom properties for colors */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --text-color: #ffffff;
    --bg-color: #0f172a;
}
```

---

## 💬 Commit Guidelines

We follow **Conventional Commits** specification:

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **ci**: CI/CD changes

### Examples

```bash
# Feature addition
git commit -m "feat(crop-calendar): add seasonal planting guide"

# Bug fix
git commit -m "fix(disease-detection): resolve image upload validation error"

# Documentation
git commit -m "docs(readme): update installation instructions"

# Multiple lines
git commit -m "feat(market-price): add price trend visualization

- Implement Chart.js integration
- Add historical price data
- Create responsive chart component

Closes #123"
```

### Best Practices

✅ **Do:**
- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- Keep subject line under 50 characters
- Capitalize subject line
- Don't end subject line with a period
- Reference issues in footer

❌ **Don't:**
- Use vague messages like "fix bug" or "update code"
- Commit unrelated changes together
- Include WIP (work in progress) commits in PR

---

## 🔀 Pull Request Process

### Before Submitting

1. **Test your changes thoroughly**
   - Run the application locally
   - Test all affected features
   - Check for console errors
   - Verify responsive design

2. **Update documentation**
   - Update README.md if needed
   - Add/update code comments
   - Update API documentation

3. **Check code quality**
   ```bash
   # Format code (if using black)
   black app.py
   
   # Check for security issues
   bandit -r .
   
   # Lint JavaScript (if using eslint)
   eslint static/script.js
   ```

4. **Ensure clean commit history**
   ```bash
   # Squash commits if needed
   git rebase -i HEAD~3
   ```

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Closes #123
Related to #456

## Changes Made
- Added crop calendar feature
- Updated UI for better mobile responsiveness
- Fixed disease detection accuracy issue

## Screenshots (if applicable)
![Before](url)
![After](url)

## Testing
- [ ] Tested locally
- [ ] All features working
- [ ] No console errors
- [ ] Responsive design verified

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Automated checks** will run (if configured)
2. **Maintainers will review** your code
3. **Address feedback** by pushing new commits
4. **Once approved**, maintainers will merge

### After Merge

1. **Delete your branch**
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Sync your fork**
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

---

## 🧪 Testing Guidelines

### Manual Testing

Before submitting a PR, test:

1. **Crop Recommendation**
   - Enter valid soil parameters
   - Test edge cases (min/max values)
   - Verify recovery plans display correctly

2. **Disease Detection**
   - Upload various image formats (JPG, PNG)
   - Test file size limits
   - Verify disease identification accuracy

3. **Market Price Prediction**
   - Select different crops and states
   - Test various months
   - Verify price predictions are reasonable

4. **UI/UX**
   - Test on different screen sizes
   - Verify all buttons and links work
   - Check form validation
   - Test error handling

### Writing Tests (Future)

We plan to add automated testing. When contributing tests:

```python
# Example unit test structure
import unittest
from app import predict_crop

class TestCropPrediction(unittest.TestCase):
    def test_rice_prediction(self):
        """Test rice prediction with typical values"""
        result = predict_crop(
            nitrogen=80,
            phosphorus=40,
            potassium=40,
            temperature=25,
            humidity=80,
            ph=6.5,
            rainfall=200
        )
        self.assertEqual(result, 'rice')
    
    def test_invalid_input(self):
        """Test handling of invalid input"""
        with self.assertRaises(ValueError):
            predict_crop(
                nitrogen=-10,  # Invalid negative value
                phosphorus=40,
                potassium=40,
                temperature=25,
                humidity=80,
                ph=6.5,
                rainfall=200
            )
```

---

## 📚 Documentation

### Code Comments

```python
# ✅ Good - Explains WHY, not WHAT
# Use MinMaxScaler to normalize features between 0 and 1
# This improves model performance and prevents feature dominance
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# ❌ Bad - States the obvious
# Create a MinMaxScaler
scaler = MinMaxScaler()
```

### README Updates

When adding features, update README.md:

- Add to Features section
- Update usage instructions
- Add screenshots if UI changed
- Update technology stack if needed

### API Documentation

Document API endpoints:

```python
@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Crop Recommendation API Endpoint
    
    POST /api/predict
    
    Request Body:
        {
            "nitrogen": 80,
            "phosphorus": 40,
            "potassium": 40,
            "temperature": 25,
            "humidity": 80,
            "ph": 6.5,
            "rainfall": 200
        }
    
    Response:
        {
            "crop": "rice",
            "confidence": 0.95,
            "alternatives": ["wheat", "maize"]
        }
    
    Status Codes:
        200: Success
        400: Invalid input
        500: Server error
    """
    pass
```

---

## 🌍 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Email**: dasouvik122005@gmail.com for general inquiries

### Getting Help

- Check existing issues and documentation
- Search closed issues for similar problems
- Ask questions in issue comments
- Be patient and respectful

### Recognition

Contributors will be:
- Listed in release notes
- Mentioned in README.md (future Contributors section)
- Credited in commit history

---

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- 🧪 **Testing**: Unit tests, integration tests
- 🌐 **Localization**: Hindi, Tamil, Telugu translations
- 📱 **Mobile**: Progressive Web App features
- 🔒 **Security**: Security enhancements and audits

### Medium Priority
- 🎨 **UI/UX**: Design improvements
- 📊 **Analytics**: Dashboard and reporting features
- 🤖 **ML Models**: Accuracy improvements
- 📚 **Documentation**: Tutorials and guides

### Future Features
- ☁️ **Cloud Integration**: Weather API, satellite imagery
- 💬 **Community**: Forums, knowledge sharing
- 🏪 **Marketplace**: Buyer-seller connections
- 📱 **Mobile Apps**: Native iOS/Android apps

---

## 📋 Contribution Checklist

Before submitting your contribution:

- [ ] Code follows project style guidelines
- [ ] Commits follow conventional commit format
- [ ] All tests pass (when available)
- [ ] Documentation updated
- [ ] Self-review completed
- [ ] No merge conflicts
- [ ] PR template filled out completely
- [ ] Related issues linked
- [ ] Screenshots added (if UI changes)

---

## 🙏 Thank You!

Every contribution, no matter how small, makes CultivaSense better. Whether you're fixing a typo, reporting a bug, or implementing a major feature, we appreciate your effort!

**Together, we're building technology that empowers farmers and promotes sustainable agriculture.** 🌱

---

## 📞 Questions?

If you have questions about contributing:

- **Email**: dasouvik122005@gmail.com
- **GitHub Issues**: Open a discussion issue
- **Documentation**: Check README.md and SECURITY.md

---

<div align="center">

**🌱 Happy Contributing! 🚀**

*Built with ❤️ for Indian Agriculture*

**© 2026 CultivaSense | Open Source Community**

</div>
