# Contributing to AI-Powered Financial Research Agent

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/your-username/FinTechAgent/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/your-username/FinTechAgent/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/FinTechAgent.git
cd FinTechAgent
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

4. **Set up pre-commit hooks** (optional)
```bash
pre-commit install
```

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable and function names
- Write docstrings for all public functions and classes
- Use type hints where appropriate

### Example Function

```python
def calculate_pe_ratio(price: float, earnings_per_share: float) -> float:
    """
    Calculate the price-to-earnings ratio.
    
    Args:
        price: Current stock price
        earnings_per_share: Earnings per share for the last 12 months
        
    Returns:
        P/E ratio as a float
        
    Raises:
        ValueError: If earnings_per_share is zero or negative
    """
    if earnings_per_share <= 0:
        raise ValueError("Earnings per share must be positive")
    
    return price / earnings_per_share
```

### Commit Message Format

We use conventional commit messages:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Example:
```
feat: add smart ticker validation with suggestions

- Implement real-time ticker validation using yfinance
- Add intelligent suggestion system for invalid tickers
- Integrate validation into orchestrator agent
- Update dashboard with user-friendly error handling
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python src/test_ticker_validation.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Include both positive and negative test cases
- Mock external API calls

Example test:
```python
def test_ticker_validation_valid_ticker():
    """Test that valid tickers pass validation."""
    collector = FinancialDataCollector()
    is_valid, message, suggestions = collector.validate_ticker("AAPL")
    
    assert is_valid is True
    assert "valid" in message.lower()
    assert len(suggestions) == 0

def test_ticker_validation_invalid_ticker():
    """Test that invalid tickers fail validation with suggestions."""
    collector = FinancialDataCollector()
    is_valid, message, suggestions = collector.validate_ticker("INVALIDTICKER")
    
    assert is_valid is False
    assert "invalid" in message.lower()
    assert len(suggestions) > 0
```

## Adding New Features

### Feature Request Process

1. Check existing issues to avoid duplicates
2. Open a new issue with the "enhancement" label
3. Describe the feature and its benefits
4. Discuss implementation approach
5. Get approval before starting development

### Implementation Guidelines

1. **Start Small**: Begin with a minimal viable implementation
2. **Follow Patterns**: Use existing code patterns and architecture
3. **Add Tests**: Include comprehensive tests for new functionality
4. **Update Documentation**: Update README and docstrings
5. **Consider Performance**: Ensure new features don't degrade performance

## Areas for Contribution

### High Priority

- [ ] **Additional Data Sources**: Integrate new financial data APIs
- [ ] **Enhanced Visualizations**: More chart types and interactive features
- [ ] **Performance Optimization**: Improve caching and processing speed
- [ ] **Error Handling**: Better error messages and recovery mechanisms

### Medium Priority

- [ ] **Portfolio Analysis**: Multi-stock analysis capabilities
- [ ] **Technical Indicators**: Add technical analysis features
- [ ] **Mobile Responsive**: Improve mobile experience
- [ ] **Internationalization**: Support for non-US markets

### Documentation

- [ ] **API Documentation**: Comprehensive API docs
- [ ] **Tutorial Videos**: Video guides for common use cases
- [ ] **Use Case Examples**: Real-world usage examples
- [ ] **Architecture Diagrams**: Detailed system architecture docs

## Questions?

Don't hesitate to ask questions! You can:

- Open an issue with the "question" label
- Start a discussion in GitHub Discussions
- Contact the maintainers directly

## Recognition

Contributors will be recognized in:

- The project README
- Release notes for their contributions
- GitHub contributor list

Thank you for contributing to the AI-Powered Financial Research Agent! 🚀