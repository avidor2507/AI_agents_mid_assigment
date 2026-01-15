from src.evaluation.eval_case import EvalCase


TEST_CASES = [
    EvalCase(
        query="What are the details of the insured vehicle in this claim?",
        expected_answer=(
            "The insured vehicle is a 2022 BMW 320i M Sport with registration number LK22 RWT, finished in Alpine White. "
            "At the time of loss, the vehicle had 18,462 miles on the odometer and was in good condition. "
            "The vehicle is covered under a comprehensive private motor insurance policy."
        ),
        expected_context=[
            "The insured vehicle is a 2022 BMW 320i M Sport, registration LK22 RWT, finished in Alpine White",
            "18,462 miles at the time of loss",
            "in good condition",
        ],
        category="summarization",
        description="Insured vehicle details summarization",
    ),
    EvalCase(
        query="Where did the collision occur and what were the circumstances?",
        expected_answer=(
            "The collision occurred on March 3rd at 08:20:05 at the signal-controlled junction of Euston Road and Judd Street "
            "in London NW1. The primary impact was on the driver-side front quarter of the insured vehicle. "
            "The collision involved the insured vehicle and a third-party vehicle."
        ),
        expected_context=[
            "collision occurred at the signal-controlled junction of Euston Road and Judd Street",
            "primary impact on the driver-side front quarter",
            "March 3rd at 08:20:05",
        ],
        category="summarization",
        description="Collision location and circumstances summarization",
    ),
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
    EvalCase(
        query="What coverage does the policy include?",
        expected_answer=(
            "The policy includes full own-damage protection, third-party liability, uninsured driver coverage, "
            "personal injury benefits, and legal expense protection."
        ),
        expected_context=[
            "comprehensive private motor insurance policy",
            "full own-damage protection, third-party liability",
        ],
        category="summarization",
        description="Policy coverage summarization",
    ),
    EvalCase(
        query="What benefits and protections does the policyholder's insurance policy provide?",
        expected_answer=(
            "The policyholder's comprehensive private motor insurance policy provides several key benefits including: "
            "full own-damage protection for the insured vehicle, third-party liability coverage, protection against "
            "uninsured drivers, personal injury benefits, and legal expense protection. The policy was active at the "
            "time of the incident."
        ),
        expected_context=[
            "comprehensive private motor insurance policy",
            "full own-damage protection, third-party liability",
            "uninsured driver coverage, personal injury benefits",
        ],
        category="summarization",
        description="Policy benefits and protections summarization",
    ),
    EvalCase(
        query="What happened to the vehicle after the collision and how was it repaired?",
        expected_answer=(
            "The insured vehicle sustained damage to the driver-side front quarter and was determined to be repairable. "
            "Vehicle repairs were carried out. During strip-down inspection, additional structural deformation was revealed "
            "requiring supplemental repair authorisation. The repairs were completed to manufacturer standards."
        ),
        expected_context=[
            "vehicle was repairable",
            "repairs were completed to manufacturer standards",
        ],
        category="summarization",
        description="Damage and repair process summarization",
    ),
    EvalCase(
        query="Provide a summary of the financial aspects of this claim",
        expected_answer=(
            "The total claim exposure amounted to £22,625.20, which includes vehicle repairs, hire vehicle charges, "
            "medical treatment, and recovery costs. The policy excess of £650 remains recoverable from the third-party insurer."
        ),
        expected_context=[
            "Total claim exposure amounted to £22,625.20",
            "policy excess of £650",
        ],
        category="summarization",
        description="Financial aspects summarization",
    ),
    EvalCase(
        query="Can you break down and summarize all the costs associated with this claim?",
        expected_answer=(
            "The total claim exposure amounted to £22,625.20, which includes costs for vehicle repairs, "
            "hire vehicle charges, medical treatment, and recovery costs. Additionally, there is a policy "
            "excess of £650 that remains recoverable from the third-party insurer."
        ),
        expected_context=[
            "Total claim exposure amounted to £22,625.20",
            "vehicle repairs, hire vehicle charges, medical treatment, and recovery costs",
            "policy excess of £650",
        ],
        category="summarization",
        description="Cost breakdown and financial summary",
    ),
    EvalCase(
        query="What was the medical impact of this incident?",
        expected_answer=(
            "The medical impact was classified as minor to moderate. Medical treatment was provided as part of the claim, "
            "and the costs were included in the total claim exposure."
        ),
        expected_context=[
            "medical impact classification was minor to moderate",
            "medical treatment was provided",
        ],
        category="summarization",
        description="Medical impact summarization",
    ),
    EvalCase(
        query="What is the policy excess amount and is it recoverable?",
        expected_answer=(
            "The policy excess amount is £650. This excess is recoverable from the third-party insurer "
            "since liability was accepted by the third party."
        ),
        expected_context=[
            "policy excess of £650",
            "collision excess under the policy is £650",
            "recoverable from the third-party insurer",
        ],
        category="summarization",
        description="Policy excess and recoverability summarization",
    ),
    EvalCase(
        query="What was the incident classification for this claim?",
        expected_answer=(
            "The incident was classified as a third-party liability event. The collision occurred at a "
            "signal-controlled junction, and the third-party insurer accepted liability for the incident."
        ),
        expected_context=[
            "incident classification was third-party liability event",
            "collision occurred at the signal-controlled junction",
            "liability was accepted",
        ],
        category="summarization",
        description="Incident classification summarization",
    ),
]

