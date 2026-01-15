# Testing Guide

This document explains how to run the test suite for the Insurance Claim Timeline Retrieval System.

## Test Structure

The test suite is organized into three main test folders, each containing its test file and data:

```
tests/
├── test_needle_in_haystack_hard_case/
│   ├── data.py                              # Test cases for hard tests
│   └── test_needle_in_haystack_hard_case.py # Hard tests for NeedleInHaystackAgent
├── test_needle_in_haystack_llm_based_case/
│   ├── data.py                                  # Test cases for LLM-based tests
│   └── test_needle_in_haystack_llm_based_case.py # LLM-based tests for NeedleInHaystackAgent
└── test_summariztion_llm_based_case/
    ├── data.py                              # Test cases for summarization tests
    └── test_summariztion_llm_based_case.py  # LLM-based tests for SummarizationExpertAgent
```

### Test Categories

- **Hard Tests**: Tests that verify specific, deterministic behavior (e.g., exact value extraction, format validation)
- **LLM-based Tests**: Tests that use LLM evaluation to assess answer quality and correctness

### Test Data Format

Test data is stored in `data.py` files within each test folder. Each `data.py` file contains a `TEST_CASES` list of `EvalCase` objects:

```python
from src.evaluation.eval_case import EvalCase

TEST_CASES = [
    EvalCase(
        query="Your test question here",
        expected_answer="The expected answer",
        expected_context=[...],  # Optional
        category="needle",       # Optional
        description="Test description"  # Optional
    ),
    # ... more test cases
]
```

## Prerequisites

1. **Install dependencies** (including pytest):
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
   - Ensure your `.env` file contains `OPENAI_API_KEY`
   - The system must have indices built (run `python main.py` at least once to build indices)

3. **Verify indices exist**:
   - The tests require the hierarchical and summary indices to be built
   - If indices don't exist, run the main application first to build them

## Running Tests

### Run All Tests

To run the entire test suite:

```bash
pytest tests/
```

### Run Tests for a Specific Test Category

Run all hard tests for NeedleInHaystackAgent:
```bash
pytest tests/test_needle_in_haystack_hard_case/
```

Run all LLM-based tests for NeedleInHaystackAgent:
```bash
pytest tests/test_needle_in_haystack_llm_based_case/
```

Run all LLM-based tests for SummarizationExpertAgent:
```bash
pytest tests/test_summariztion_llm_based_case/
```

### Run a Specific Test File

Run only hard tests for NeedleInHaystackAgent:
```bash
pytest tests/test_needle_in_haystack_hard_case/test_needle_in_haystack_hard_case.py
```

Run only LLM-based tests for NeedleInHaystackAgent:
```bash
pytest tests/test_needle_in_haystack_llm_based_case/test_needle_in_haystack_llm_based_case.py
```

Run LLM-based tests for SummarizationExpertAgent:
```bash
pytest tests/test_summariztion_llm_based_case/test_summariztion_llm_based_case.py
```

### Run a Specific Test Function

Run a specific test by name:
```bash
pytest tests/test_needle_in_haystack_hard_case/test_needle_in_haystack_hard_case.py::test_needle_in_haystack_hard_case
```

### Run Tests with Verbose Output

Get detailed output for each test:
```bash
pytest tests/ -v
```

### Run Tests with Output Capturing Disabled

See print statements and logs during test execution:
```bash
pytest tests/ -s
```

### Run Tests and Show Coverage

If you have pytest-cov installed:
```bash
pytest tests/ --cov=src
```

### Generate HTML Test Report

Generate an HTML report of test results:
```bash
pytest tests/ --html=report.html --self-contained-html
```

The HTML report will be saved as `report.html` in the current directory. The `--self-contained-html` flag creates a single HTML file with all CSS and JavaScript embedded, making it easy to share.

To save the report in a specific directory:
```bash
pytest tests/ --html=reports/test_report.html --self-contained-html
```

You can also combine HTML reports with verbose output:
```bash
pytest tests/ -v --html=report.html --self-contained-html
```

## Test Data Management

### Adding New Test Cases

1. **Edit the appropriate `data.py` file** in the test folder:
   - For NeedleInHaystackAgent hard tests: `tests/test_needle_in_haystack_hard_case/data.py`
   - For NeedleInHaystackAgent LLM-based tests: `tests/test_needle_in_haystack_llm_based_case/data.py`
   - For SummarizationExpertAgent LLM-based tests: `tests/test_summariztion_llm_based_case/data.py`

2. **Add a new `EvalCase` entry** to the `TEST_CASES` list:
```python
EvalCase(
    query="Your new test question",
    expected_answer="Expected answer",
    expected_context=[...],  # Optional: list of expected context strings
    category="needle",       # Optional: category name
    description="Test description"  # Optional: description of what this tests
),
```

3. The test file will automatically pick up the new test case since it imports `TEST_CASES` from `data.py`.

### Test Data File Locations

- `tests/test_needle_in_haystack_hard_case/data.py` - Hard test cases for NeedleInHaystackAgent
- `tests/test_needle_in_haystack_llm_based_case/data.py` - LLM-based test cases for NeedleInHaystackAgent
- `tests/test_summariztion_llm_based_case/data.py` - LLM-based test cases for SummarizationExpertAgent

## Writing New Tests

### Test File Structure

Each test folder contains:
- `data.py`: Contains the `TEST_CASES` list with all test case definitions
- `test_*.py`: Contains the actual test functions

Test files follow this pattern:

```python
"""Test description."""

import pytest
from src.evaluation.eval_case import EvalCase
from tests.test_folder_name.data import TEST_CASES
from src.helpers.agent_helper import assert_hard_query  # or assert_llm_based_query

@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.query[:50] + "..." if len(tc.query) > 50 else tc.query)
def test_example(orchestrator, logger, test_case: EvalCase):
    """Test description."""
    # Your test implementation here
    assert_hard_query(orchestrator, test_case, logger)
```

### Example Test Implementation

Here's a template for implementing tests:

```python
"""Hard tests for NeedleInHaystackAgent."""

import pytest
from src.evaluation.eval_case import EvalCase
from tests.test_needle_in_haystack_hard_case.data import TEST_CASES
from src.helpers.agent_helper import assert_hard_query

@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.query[:50] + "..." if len(tc.query) > 50 else tc.query)
def test_needle_in_haystack_hard_case(orchestrator, logger, test_case: EvalCase):
    """Test hard cases for NeedleInHaystackAgent."""
    assert_hard_query(orchestrator, test_case, logger)
```

The `@pytest.mark.parametrize` decorator automatically creates separate test instances for each test case in `TEST_CASES`, so each case runs as an individual test.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running tests from the project root directory
   ```bash
   cd /path/to/MidProject
   pytest tests/
   ```

2. **Missing Indices**: If you get errors about missing indices:
   ```bash
   python main.py
   # Let it build indices, then exit
   pytest tests/
   ```

3. **API Key Errors**: Ensure your `.env` file is properly configured with `OPENAI_API_KEY`

4. **Module Not Found**: Make sure you're in the project root and the `src` directory is in your Python path

### Running Tests in Debug Mode

For detailed debugging information:
```bash
pytest tests/ -v -s --tb=long
```

## Continuous Integration

If you're setting up CI/CD, ensure:
- Environment variables are set (especially `OPENAI_API_KEY`)
- Indices are built before running tests (or use test fixtures to build them)
- All dependencies from `requirements.txt` are installed

## Additional pytest Options

- `-k EXPRESSION`: Run tests matching the expression (e.g., `pytest -k "hard"`)
- `-x`: Stop after first failure
- `--maxfail=N`: Stop after N failures
- `-m MARKER`: Run tests with specific markers (if you add markers to tests)
- `--lf`: Run only tests that failed in the last run
- `--ff`: Run failed tests first, then the rest
- `--html=report.html`: Generate an HTML test report
- `--self-contained-html`: Create a self-contained HTML file (all CSS/JS embedded)

For more pytest options, see: `pytest --help`

