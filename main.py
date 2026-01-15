from __future__ import annotations
from typing import NoReturn
from src.agents.orchestrator_system import OrchestratorSystem
from src.utils.exceptions import  AgentError, EvaluationError
from src.utils.logger import logger
from src.config.settings import config
from src.evaluation import EvalSuite, get_test_cases
from datetime import datetime
from logging import Logger
from src.helpers.agent_helper import init

def _system(log: Logger, orchestrator: OrchestratorSystem) -> None:
    print("==============================================")
    print(" Insurance Claim Assistant")
    while True:
        print("==============================================")
        print("Type your question about the claim.")
        print("Type 'evel' or 'evaluation' for evaluation")
        print("Type 'exit' or 'quit' to exit.\n")
        print("==============================================")

        try:
            query = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        print(f"query: {query}")

        if not query:
            print("no query entered")
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        elif query.lower() in {"eval", "evaluation", "e"}:
            #evaluation mode
            print("enter evaluation mode")
            _evaluation_mode(orchestrator, log)
            continue
        else:
            #query mode
            print("enter query mode")
            _query_mode(orchestrator, query, log)
            continue
def _query_mode(orchestrator: OrchestratorSystem, query: str, log: Logger) -> None:
    try:
        result = orchestrator.handle_query(query)

        print("\n--- Answer ---")
        print(result)
        print("--------------")

    except AgentError as e:
        log.error(f"Agent error while handling query: {e}", exc_info=True)
        print(f"Error: Agent failed to answer the query: {e}")
    except Exception as e:
        log.error(f"Unexpected error while handling query: {e}", exc_info=True)
        print(f"Unexpected error while handling query: {e}")
def _evaluation_mode(orchestrator: OrchestratorSystem, log: Logger) -> None:
    """Run evaluation test suite and generate report."""
    print("\n" + "=" * 60)
    print("EVALUATION MODE")
    print("=" * 60)
    print("Running evaluation test suite...\n")
    
    try:
        # Load test cases
        test_cases = get_test_cases()
        log.info(f"Loaded {len(test_cases)} test cases for evaluation")
        print(f"Loaded {len(test_cases)} test cases\n")
        
        # Initialize orchestrator for evaluation
        try:
            log.info("Orchestrator initialized for evaluation")
        except AgentError as e:
            log.error(f"Failed to initialize orchestrator for evaluation: {e}", exc_info=True)
            print(f"Error: Failed to initialize orchestrator: {e}")
            return
        except Exception as e:
            log.error(f"Unexpected error initializing orchestrator: {e}", exc_info=True)
            print(f"Unexpected error initializing orchestrator: {e}")
            return
        
        # Create test suite
        try:
            test_suite = EvalSuite(orchestrator=orchestrator)
            test_suite.load_test_cases(test_cases)
            log.info("Test suite initialized")
        except EvaluationError as e:
            log.error(f"Failed to initialize test suite: {e}", exc_info=True)
            print(f"Error: Failed to initialize test suite: {e}")
            return
        except Exception as e:
            log.error(f"Unexpected error initializing test suite: {e}", exc_info=True)
            print(f"Unexpected error initializing test suite: {e}")
            return
        
        # Run all test cases
        try:
            print("Running test cases...")
            results = test_suite.run_all(get_retrieval_context=True)
            log.info(f"Completed running {len(results)} test cases")
            print(f"\nCompleted running {len(results)} test cases\n")
        except EvaluationError as e:
            log.error(f"Evaluation failed: {e}", exc_info=True)
            print(f"Error: Evaluation failed: {e}")
            return
        except Exception as e:
            log.error(f"Unexpected error during evaluation: {e}", exc_info=True)
            print(f"Unexpected error during evaluation: {e}")
            return
        
        # Generate report
        try:
            evaluation_dir = config.RESULTS_DIR / "evaluation"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            
            report_filename = evaluation_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = test_suite.generate_report(
                output_file=report_filename,
                include_details=True
            )
            log.info(f"Evaluation report generated: {report_filename}")
            
            # Print summary
            test_suite.print_summary()
            
            # Print report location
            print(f"\nFull evaluation report saved to: {report_filename}")
            
        except Exception as e:
            log.error(f"Failed to generate report: {e}", exc_info=True)
            print(f"Error: Failed to generate report: {e}")
            return
        
        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        log.error(f"Unexpected error in evaluation mode: {e}", exc_info=True)
        print(f"Unexpected error in evaluation mode: {e}")

def main() -> NoReturn:
    #initialize
    log, orchestrator = init()

    # Simple CLI loop
    _system(log, orchestrator)


if __name__ == "__main__":
    main()
