from src.evaluation.eval_case import EvalCase


TEST_CASES = [
    # From eval_cases.py - Precise factual query - Registration number
    EvalCase(
        query="What is the exact registration number of the insured vehicle?",
        expected_answer="LK22 RWT",
        expected_context=[
            "The insured vehicle is a 2022 BMW 320i M Sport, registration LK22 RWT",
        ],
        category="needle",
        description="Exact value retrieval - registration number",
    ),
    
    # From eval_cases.py - Precise factual query - Amount
    EvalCase(
        query="What was the total claim exposure amount?",
        expected_answer="£22,625.20",
        expected_context=[
            "Total claim exposure amounted to £22,625.20",
        ],
        category="needle",
        description="Exact value retrieval - monetary amount",
    ),
    
    # From eval_cases.py - Precise factual query - Policy excess
    EvalCase(
        query="What was the policy excess amount?",
        expected_answer="£650",
        expected_context=[
            "The applicable collision excess under the policy is £650",
        ],
        category="needle",
        description="Exact value retrieval - policy excess",
    ),
    
    # From eval_cases.py - Needle-in-haystack query - Specific detail
    EvalCase(
        query="What color was the insured vehicle?",
        expected_answer="Alpine White",
        expected_context=[
            "finished in Alpine White",
        ],
        category="needle",
        description="Specific detail retrieval from detailed description",
    ),
    
    # Policyholder name query
    EvalCase(
        query="What is the name of the policyholder?",
        expected_answer="Daniel Whitmore",
        expected_context=[
            "policyholder Daniel Whitmore",
            "The policyholder maintains",
        ],
        category="needle",
        description="Policyholder name retrieval",
    ),
    
    # Additional needle case 1 - Vehicle details
    EvalCase(
        query="What was the year of manufacture of the insured vehicle?",
        expected_answer="2022",
        expected_context=[
            "The insured vehicle is a 2022 BMW 320i M Sport",
        ],
        category="needle",
        description="Year of manufacture retrieval",
    ),
    
    # Additional needle case 2 - Third-party vehicle
    EvalCase(
        query="What is the make and model of the third-party vehicle?",
        expected_answer="2018 Vauxhall Insignia",
        expected_context=[
            "third-party vehicle is a 2018 Vauxhall Insignia",
        ],
        category="needle",
        description="Third-party vehicle identification",
    ),
    
    # Additional needle case 3 - Third-party driver
    EvalCase(
        query="What is the name of the third-party driver?",
        expected_answer="Thomas Ellison",
        expected_context=[
            "third-party driver was Thomas Ellison",
        ],
        category="needle",
        description="Third-party driver name retrieval",
    ),
    
    # Additional needle case 4 - Location details
    EvalCase(
        query="At which junction did the collision occur?",
        expected_answer="The collision occurred at the signal-controlled junction of Euston Road and Judd Street, London NW1.",
        expected_context=[
            "collision occurred at the signal-controlled junction of Euston Road and Judd Street",
        ],
        category="needle",
        description="Collision location retrieval",
    ),
    
    # Additional needle case 5 - Vehicle condition
    EvalCase(
        query="What was the condition of the insured vehicle before the incident?",
        expected_answer="good condition",
        expected_context=[
            "vehicle was in good condition",
        ],
        category="needle",
        description="Vehicle condition assessment",
    ),#0.8
]

