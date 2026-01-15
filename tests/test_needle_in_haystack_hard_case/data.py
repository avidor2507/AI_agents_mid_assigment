from src.evaluation.eval_case import EvalCase

TEST_CASES = [
    EvalCase(
        query="What is the color of the insured vehicle? please answer only the color name!",
        expected_answer="Alpine White"
    ),
    EvalCase(
        query="What is the exact registration number of the insured vehicle?",
        expected_answer="LK22 RWT"
    ),
    EvalCase(
        query="What was the total claim exposure amount?",
        expected_answer="£22,625.20"
    ),
    EvalCase(
        query="What was the policy excess amount?",
        expected_answer="£650"
    ),
    EvalCase(
        query="What is the make and model of the insured vehicle? please answer only the make and model!",
        expected_answer="2022 BMW 320i M Sport"
    ),
    EvalCase(
        query="What was the mileage of the insured vehicle at the time of loss?",
        expected_answer="18,462 miles"
    ),
    EvalCase(
        query="What is the registration number of the third-party vehicle?",
        expected_answer="Not found in the provided context."
    ),
    EvalCase(
        query="What is the make and model of the third-party vehicle?",
        expected_answer="2018 Vauxhall Insignia"
    ),
    EvalCase(
        query="What is the name of the third-party driver?",
        expected_answer="Thomas Ellison"
    ),
    EvalCase(
        query="At which junction did the collision occur? please answer only the junction name!",
        expected_answer="Euston Road and Judd Street"
    ),
    EvalCase(
        query="What type of insurance policy does the policyholder have? please answer only the policy type!",
        expected_answer="comprehensive private motor insurance"
    ),
    EvalCase(
        query="What was the year of manufacture of the insured vehicle?",
        expected_answer="2022"
    ),
    EvalCase(
        query="What was the primary impact location on the insured vehicle?",
        expected_answer="driver-side front quarter"
    ),
    EvalCase(
        query="Was the vehicle repairable or a total loss? please answer only 'repairable' or 'total loss'!",
        expected_answer="repairable"
    ),
    EvalCase(
        query="What was the medical impact classification? please answer only in simple words like 'minor', 'moderate', 'major', etc.",
        expected_answer="minor to moderate"
    ),
    EvalCase(
        query="Was liability accepted by the third-party insurer? please answer only 'Yes' or 'No'!",
        expected_answer="Yes"
    ),
    EvalCase(
        query="What was the policy number? please answer only the policy number!",
        expected_answer="UK-AUTO-992174"
    ),
    EvalCase(
        query="What was the name of the policyholder? please answer only the name!",
        expected_answer="Daniel Whitmore"
    ),
    EvalCase(
        query="Is the policyholder love pizza? please answer only 'Yes' or 'No' or 'Not found in the provided context'!",
        expected_answer="Not found in the provided context."
    ),
    EvalCase(
        query="Is the car was in space? please answer only 'Yes' or 'No' or 'Not found in the provided context'!",
        expected_answer="Not found in the provided context."
    ),
    EvalCase(
        query="Is the policyholder learning diving? please answer only 'Yes' or 'No' or 'Not found in the provided context'!",
        expected_answer="Not found in the provided context."
    ),
    EvalCase(
        query="Is the police officer was old or young? please answer only 'old' or 'young' or 'Not found in the provided context'!",
        expected_answer="Not found in the provided context."
    ),
    EvalCase(
        query="How many free rooms where in the hospital? please answer only the amountor 'Not found in the provided context'!",
        expected_answer="Not found in the provided context."
    ),
]

