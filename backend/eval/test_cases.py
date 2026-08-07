"""
All test scenarios.
To add a new test, append a dict to TEST_CASES.
"""

TEST_CASES = [
    {
        "id": 1,
        "category": "low",
        "description": "Mild cold symptoms",
        "input": (
            "I have a mild runny nose and slight sore throat. "
            "Severity 2 out of 10. Started yesterday. "
            "I am 28 years old with no existing conditions."
        ),
        "expected_urgency": "low",
    },
    {
        "id": 2,
        "category": "moderate",
        "description": "Persistent fever with sore throat",
        "input": (
            "I have had a fever of 38.5 degrees celsius and a bad sore throat "
            "for 3 days. Severity 5 out of 10. "
            "I am 32 years old with no existing conditions."
        ),
        "expected_urgency": "moderate",
    },
    {
        "id": 3,
        "category": "high",
        "description": "Severe abdominal pain",
        "input": (
            "I have severe abdominal pain rated 8 out of 10 "
            "for the past 6 hours. It started suddenly. "
            "I am 40 years old with no existing conditions."
        ),
        "expected_urgency": "high",
    },
    {
        "id": 4,
        "category": "emergency",
        "description": "Chest pain with cardiac risk factors",
        "input": (
            "I have crushing chest pain radiating to my left arm, "
            "severity 9 out of 10. I am 55 years old with hypertension "
            "and diabetes. This started 30 minutes ago."
        ),
        "expected_urgency": "emergency",
    },
    {
        "id": 5,
        "category": "low",
        "description": "Minor ankle sprain",
        "input": (
            "I twisted my ankle while walking. Mild swelling and pain "
            "rated 3 out of 10. It happened 2 hours ago. "
            "I am 25 years old with no existing conditions."
        ),
        "expected_urgency": "low",
    },
    {
        "id": 6,
        "category": "high",
        "description": "Very high fever in adult",
        "input": (
            "I have a fever of 40 degrees celsius, severe headache, "
            "and body aches all over. Severity 7 out of 10. "
            "This has been going on for 2 days. "
            "I am 35 years old with no existing conditions."
        ),
        "expected_urgency": "high",
    },
    {
        "id": 7,
        "category": "low",
        "description": "Mild lower back pain",
        "input": (
            "I have lower back pain after sitting at a desk all day. "
            "Severity 3 out of 10. Started today. "
            "I am 30 years old with no existing conditions."
        ),
        "expected_urgency": "low",
    },
    {
        "id": 8,
        "category": "emergency",
        "description": "Breathing difficulty with cyanosis",
        "input": (
            "I am having serious difficulty breathing, tightness in my chest, "
            "and my lips feel slightly blue. Severity 9 out of 10. "
            "I am 60 years old with a history of asthma."
        ),
        "expected_urgency": "emergency",
    },
    {
        "id": 9,
        "category": "moderate",
        "description": "Persistent headache with nausea",
        "input": (
            "I have had a persistent headache with nausea for 2 days. "
            "Severity 6 out of 10. No vision changes or neck stiffness. "
            "I am 38 years old with no existing conditions."
        ),
        "expected_urgency": "moderate",
    },
    {
        "id": 10,
        "category": "moderate",
        "description": "Urinary tract infection symptoms",
        "input": (
            "I have a burning sensation when urinating and a frequent urge "
            "to go to the toilet. Severity 5 out of 10. Started 2 days ago. "
            "I am 27 years old with no existing conditions."
        ),
        "expected_urgency": "moderate",
    },
]

VALID_URGENCY_LEVELS = {"low", "moderate", "high", "emergency"}
URGENCY_ORDER        = ["low", "moderate", "high", "emergency"]