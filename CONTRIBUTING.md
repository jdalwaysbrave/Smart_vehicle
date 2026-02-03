# Contributing to Smart Vehicle Project

Thank you for your interest in contributing to the Smart Vehicle Traffic Sign Recognition project! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, library versions)
- Screenshots or logs if applicable

### Suggesting Enhancements

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain why this feature would be useful
- Include examples or mockups if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Write clear, commented code
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed

3. **Test your changes**:
   - Ensure existing functionality still works
   - Test edge cases
   - Verify with actual hardware if possible

4. **Commit your changes**:
   - Use clear commit messages
   - Reference issues if applicable (e.g., "Fixes #123")

5. **Submit a pull request**:
   - Provide a clear description of the changes
   - Link to related issues
   - Include screenshots/videos for visual changes

## 💻 Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Smart_vehicle.git
cd Smart_vehicle
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions small and focused
- Comment complex logic

### Example:
```python
def process_image(image, threshold=127):
    """
    Process an image by applying threshold.
    
    Args:
        image (np.ndarray): Input image
        threshold (int): Threshold value (default: 127)
    
    Returns:
        np.ndarray: Processed binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary
```

## 🧪 Testing

While we don't currently have automated tests, please:
- Test your changes manually
- Verify with different lighting conditions
- Test with various traffic signs
- Check edge cases (no signs, multiple signs, etc.)
- Document your testing process in the PR

## 📚 Documentation

When adding features:
- Update README.md if needed
- Update code comments
- Add docstrings to new functions
- Update configuration examples

## 🔧 Priority Areas for Contribution

We especially welcome contributions in these areas:

1. **Code Modernization**:
   - Update deprecated `sklearn.externals.joblib` to direct `joblib` import
   - Improve error handling
   - Add logging system

2. **Testing**:
   - Add unit tests
   - Add integration tests
   - Create test data/mocks

3. **Documentation**:
   - Add inline code documentation
   - Create tutorials
   - Add more examples

4. **Features**:
   - Support for more traffic sign types
   - Configuration file support
   - GUI for monitoring
   - Performance optimizations

5. **Model Training**:
   - Add training scripts
   - Add data augmentation
   - Improve classification accuracy

## ❓ Questions?

If you have questions:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Be respectful and constructive

## 📋 Checklist for Pull Requests

Before submitting a PR, ensure:
- [ ] Code follows the project style guidelines
- [ ] Changes have been tested
- [ ] Documentation has been updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included
- [ ] Branch is up to date with main

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inspiring community for all.

### Our Standards
- Be respectful and inclusive
- Be patient and welcoming
- Be considerate and constructive
- Focus on what is best for the community

### Unacceptable Behavior
- Harassment or discrimination of any kind
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing others' private information

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Smart Vehicle project! Your efforts help make this project better for everyone. 🚗✨
