"""
Pytest configuration and shared fixtures for tests.
"""

import pytest
from typing import Generator, Tuple
from logging import Logger

from src.helpers.agent_helper import init
from src.agents.orchestrator_system import OrchestratorSystem


@pytest.fixture(scope="session")
def _initialized_system() -> Generator[Tuple[Logger, OrchestratorSystem], None, None]:
    """
    Internal fixture that initializes the system once per session.
    
    This ensures init() is only called once, and both logger and orchestrator
    fixtures can reuse the same initialization.
    """
    log, orchestrator_instance = init()
    yield log, orchestrator_instance


@pytest.fixture(scope="session")
def logger(_initialized_system) -> Generator[Logger, None, None]:
    """
    Pytest fixture that provides the logger instance.
    
    Session-scoped, so it's initialized once per test session and reused.
    
    Usage:
        def test_something(logger):
            logger.info("Test log message")
    """
    log, _ = _initialized_system
    yield log


@pytest.fixture(scope="session")
def orchestrator(_initialized_system) -> Generator[OrchestratorSystem, None, None]:
    """
    Pytest fixture that provides an initialized OrchestratorSystem.
    
    Session-scoped, so it's initialized once per test session and reused.
    
    Usage:
        def test_something(orchestrator):
            result = orchestrator.handle_query("test question")
            assert result is not None
    """
    _, orchestrator_instance = _initialized_system
    yield orchestrator_instance

