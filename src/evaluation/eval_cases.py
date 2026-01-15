"""
Test cases for evaluation.

This module defines test cases covering various query types:
- High-level summary questions
- Precise factual queries
- Needle-in-haystack queries
- Timeline questions
- Mixed complexity queries
"""

from src.evaluation.eval_case import EvalCase


# Test cases for evaluation
EVALUATION_TEST_CASES = [
    # High-level summary questions
    EvalCase(
        query="Give me a high-level summary of what happened in this insurance claim",
        expected_answer=(
            "The claim involves a motor vehicle collision where the insured vehicle, "
            "a 2022 BMW 320i M Sport (registration LK22 RWT), was struck by a third-party vehicle. "
            "The incident occurred at an intersection, and liability was accepted by the third-party insurer. "
            "The total claim exposure was £22,625.20, including vehicle repairs, hire vehicle charges, "
            "medical treatment, and recovery costs. The policy excess of £650 remains recoverable."
        ),
        expected_context=[
            "Summary The claim was reported on the day of loss",
            "Total claim exposure amounted to £22,625.20",
        ],
        category="summarization",
        description="High-level overview of the entire claim",
    ),
    
    # Timeline question
    EvalCase(
        query="What is the overall timeline of the insurance claim from incident to resolution?",
        expected_answer=(
            "The claim timeline includes: the incident occurred (collision at intersection), "
            "the claim was reported on the day of loss, liability was formally accepted by the "
            "third-party insurer following review, vehicle repairs were carried out, strip-down "
            "inspection revealed additional structural deformation requiring supplemental repair "
            "authorisation, repairs were completed to manufacturer standards, and the claim was resolved."
        ),
        expected_context=[
            "The claim was reported on the day of loss",
            "Vehicle repairs were carried out",
            "Repairs were completed to manufacturer standards",
        ],
        category="summarization",
        description="Timeline of events from incident to resolution",
    ),
    
    # Precise factual query - Registration number
    EvalCase(
        query="What is the exact registration number of the insured vehicle?",
        expected_answer="LK22 RWT",
        expected_context=[
            "The insured vehicle is a 2022 BMW 320i M Sport, registration LK22 RWT",
        ],
        category="needle",
        description="Exact value retrieval - registration number",
    ),
    
    # Precise factual query - Amount
    EvalCase(
        query="What was the total claim exposure amount?",
        expected_answer="£22,625.20",
        expected_context=[
            "Total claim exposure amounted to £22,625.20",
        ],
        category="needle",
        description="Exact value retrieval - monetary amount",
    ),
    
    # Precise factual query - Policy excess
    EvalCase(
        query="What was the policy excess amount?",
        expected_answer="£650",
        expected_context=[
            "The applicable collision excess under the policy is £650",
        ],
        category="needle",
        description="Exact value retrieval - policy excess",
    ),
    
    # Needle-in-haystack query - Specific detail
    EvalCase(
        query="What color was the insured vehicle?",
        expected_answer="Alpine White",
        expected_context=[
            "finished in Alpine White",
        ],
        category="needle",
        description="Specific detail retrieval from detailed description",
    ),
    
    # Section-specific summary
    EvalCase(
        query="Please summarize for me section 1 content",
        expected_answer=(
            "Section 1 describes the policyholder's comprehensive private motor insurance policy, "
            "which includes full own-damage protection, third-party liability, uninsured driver coverage, "
            "personal injury benefits, and legal expense protection. The policy was active at the time "
            "of the incident with a collision excess of £650. The insured vehicle is a 2022 BMW 320i M Sport "
            "(registration LK22 RWT) in Alpine White, with 18,462 miles at the time of loss, in good condition."
        ),
        expected_context=[
            "The policyholder maintains a comprehensive private motor insurance policy",
            "The insured vehicle is a 2022 BMW 320i M Sport, registration LK22 RWT, finished in Alpine White",
        ],
        category="summarization",
        description="Section-specific summarization",
    ),
    
    # Mixed complexity - Specific timestamp query
    EvalCase(
        query="What time did the collision occur on March 3rd?",
        expected_answer="08:20:05",
        expected_context=[
            "08:20:05 – Following the primary impact",
        ],
        category="needle",
        description="Precise timestamp retrieval from incident description",
    ),
]


def get_test_cases() -> list[EvalCase]:
    """
    Get all evaluation test cases.
    
    Returns:
        List of EvalCase objects
    """
    return EVALUATION_TEST_CASES.copy()

