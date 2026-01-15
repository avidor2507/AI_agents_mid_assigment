from logging import Logger
import json

from src.indexing.index_manager import IndexManager
from src.agents.orchestrator_system import OrchestratorSystem
from src.utils.exceptions import IndexingError, AgentError, ConfigurationError
from src.utils.logger import logger
from src.data.pdf_loader import PDFLoader
from src.config.settings import config
from src.data.chunker import HierarchicalChunker
from src.evaluation import EvalSuite
from src.evaluation.eval_case import EvalCase


def _load_pdf(index_manager: IndexManager, log: Logger) -> None:
    pdf_path = config.RAW_DATA_DIR / "claim.pdf"
    if not pdf_path.exists():
        message = f"PDF not found at {pdf_path}"
        log.error(message)
        raise Exception(message)
    try:
        loader = PDFLoader()
        document = loader.load(pdf_path)
        log.info(f"✓ Loaded PDF: {pdf_path}")
    except Exception as e:
        log.error(f"✗ Error loading PDF from {pdf_path}: {e}")
        raise e

    try:
        chunker = HierarchicalChunker()
        hierarchical_structure = chunker.chunk_document(
            document,
            document_id="test_document_1",
            claim_id="test_claim_1"
        )
        log.info("✓ Created hierarchical chunks")
    except Exception as e:
        log.error(f"✗ Error chunking document: {e}")
        raise e
    indices_exist = index_manager.check_indices_exist()
    if not indices_exist:
        try:
            index_manager.build_indices(hierarchical_structure)
            log.info("✓ Indices built successfully")
        except Exception as e:
            log.error(f"✗ Error building indices: {e}")
            raise e
def init() -> tuple[Logger, OrchestratorSystem]:
    log: Logger = logger

    # Initialize and load indices
    index_manager = IndexManager()
    try:
        log.info("Starting Insurance Claim Assistant CLI")
        _load_pdf(index_manager, log)
    except (IndexingError, ConfigurationError) as e:
        log.error(f"Failed to initialize indices: {e}", exc_info=True)
        raise e
    except Exception as e:
        log.error(f"Unexpected error during index initialization: {e}", exc_info=True)
        raise e

    # Initialize orchestrator and agents
    try:
        orchestrator = OrchestratorSystem()
    except AgentError as e:
        log.error(f"Failed to initialize agents: {e}", exc_info=True)
        raise e
    except Exception as e:
        log.error(f"Unexpected error initializing agents: {e}", exc_info=True)
        raise e

    return log, orchestrator

def assert_hard_query(orchestrator: OrchestratorSystem, test_case: EvalCase) -> None:
    result = orchestrator.handle_query(test_case.query)
    assert result.lower() == test_case.expected_answer.lower(), f"Answer does not match expected value. query: {test_case.query}, expected: {test_case.expected_answer.lower()}, actual: {result.lower()}"
def assert_hard_queries(orchestrator: OrchestratorSystem, test_cases: list[EvalCase]) -> None:
    for test_case in test_cases:
        assert_hard_query(orchestrator, test_case)

def assert_llm_based_query(orchestrator: OrchestratorSystem, test_case: EvalCase, expected_result: float, logger: Logger) -> None:
    test_suite = EvalSuite(orchestrator=orchestrator)
    result = test_suite.evaluate_average(test_case, get_retrieval_context=False)
    assert result.answer_correctness >= expected_result, f"Answer does not match expected score. query: {test_case.query}, expected: {test_case.expected_score}, actual: {result.answer_correctness}"
def assert_llm_based_queries(orchestrator: OrchestratorSystem, test_cases: list[EvalCase], expected_result: float, logger: Logger) -> None:
    for test_case in test_cases:
        assert_llm_based_query(orchestrator, test_case, expected_result, logger)