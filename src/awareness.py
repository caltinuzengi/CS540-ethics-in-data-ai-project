"""
Awareness & Literacy module - Yasemin Gunindi

This module adds a role-based literacy layer to an AI early-warning system.

It does not change the model prediction.
It explains how students and educators should interpret the prediction.
"""

import pandas as pd


GENERAL_MODEL_NOTICE = """
AI EARLY-WARNING SYSTEM: GENERAL MODEL INFORMATION

This system uses a machine learning model to estimate whether a student may need early academic support.

The prediction is not a final verdict about the student.
It should be understood as a decision-support signal.

The model was trained on historical educational data. It may use information such as:
- VLE click activity
- Assessment scores
- Registration records
- Course information

Important limitations:
- The model cannot fully understand personal circumstances.
- The model cannot know recent life events, health issues, motivation, family context, or financial stress.
- The model may reflect patterns and inequalities present in historical educational data.
- A numerical score can look objective, but it is still produced by a model with limitations.

Responsible use:
- Use predictions to offer support, not punishment.
- Do not use predictions as the only basis for decisions.
- A human advisor or educator should review the prediction before any action is taken.
"""


def print_general_model_notice():
    """Prints a general model notice before any role-specific information."""
    print("=" * 80)
    print(GENERAL_MODEL_NOTICE.strip())
    print("=" * 80)


def get_role_guidance(role: str) -> str:
    """
    Returns role-specific guidance for either students or educators.

    Parameters
    ----------
    role : str
        Expected values: 'student' or 'educator'
    """
    role = role.strip().lower()

    if role == "student":
        return """
STUDENT INFORMATION

What this system does:
- The system estimates whether you may need academic support.
- If you are marked as "At-Risk", this does not mean you will fail.
- It means the system thinks additional support may be useful.

What you should know:
- The prediction is based on data patterns.
- It may not include your personal situation.
- It may not understand why your activity or scores changed.
- The prediction should not define your ability or potential.

Your rights:
- You can ask for a human review.
- You can ask what information contributed to the prediction.
- You can explain your situation to an advisor if you want.
- You do not have to disclose personal circumstances.
- The prediction should not be used to punish or exclude you.

Best interpretation:
- Treat the prediction as an invitation to support, not as a judgment.
""".strip()

    if role == "educator":
        return """
EDUCATOR INFORMATION

What this system does:
- The system estimates which students may benefit from early academic support.
- It can help prioritize outreach, advising, or additional resources.

What educators should avoid:
- Do not treat the prediction as a final truth.
- Do not use the score as the only basis for intervention.
- Do not stigmatize students marked as "At-Risk".
- Do not assume that low engagement always means low effort.

Responsible interpretation:
- Use the prediction as a prompt for supportive outreach.
- Combine the model output with qualitative knowledge of the student.
- Consider whether the prediction may be affected by data quality or missing context.
- Give students a chance to clarify or contest the prediction.
- Monitor whether some groups are more often misclassified or over-flagged.

Best interpretation:
- The model can help identify possible support needs, but educators remain responsible for contextual judgment.
""".strip()

    return """
Unknown role.

Please enter either:
- student
- educator
""".strip()


def annotate_prediction(student_id, risk_label: str, confidence: float, role: str = "student") -> str:
    """
    Returns an annotated prediction in role-sensitive language.

    Parameters
    ----------
    student_id:
        Student identifier.
    risk_label:
        Predicted label, for example 'At-Risk' or 'Pass'.
    confidence:
        Model confidence for the predicted class.
    role:
        'student' or 'educator'.
    """
    role = role.strip().lower()
    confidence_text = "Not available" if pd.isna(confidence) else f"{confidence:.1%}"

    base_lines = [
        f"Student ID: {student_id}",
        f"Predicted outcome: {risk_label}",
        f"Model confidence: {confidence_text}",
        "",
        "Interpretation:",
        "This output is a decision-support signal, not a final verdict.",
    ]

    if role == "student":
        role_lines = [
            "",
            "For the student:",
            "- If the result is 'At-Risk', it means support may be helpful.",
            "- It does not mean you will fail.",
            "- You may request human review.",
            "- You may ask what information contributed to the prediction.",
            "- The prediction should not be used to punish or exclude you.",
        ]
    else:
        role_lines = [
            "",
            "For the educator:",
            "- Use this prediction as a prompt for supportive outreach.",
            "- Review the student's context before taking action.",
            "- Do not rely only on the model score.",
            "- Consider whether the model may be missing relevant information.",
            "- Document any intervention decision made after reviewing the prediction.",
        ]

    return "\n".join(base_lines + role_lines)


def generate_educator_briefing(model_summary: str = None, fairness_summary: pd.DataFrame = None) -> str:
    """
    Generates an educator-facing briefing.

    This function does not calculate fairness or explanations.
    It explains how educators should interpret the available model outputs responsibly.
    """
    lines = [
        "EDUCATOR BRIEFING",
        "",
        "Purpose:",
        "The early-warning model is designed to support timely academic intervention.",
        "It should not replace educator judgment.",
        "",
        "Responsible use checklist:",
        "- Has the prediction been reviewed by a human advisor or educator?",
        "- Is the intervention supportive rather than punitive?",
        "- Has the student been given a chance to provide context?",
        "- Are there signs that the model may be less accurate for some groups?",
        "- Is the decision documented in a way that can be reviewed later?",
    ]

    if model_summary:
        lines.extend([
            "",
            "Model summary:",
            model_summary,
        ])

    if fairness_summary is not None:
        lines.extend([
            "",
            "Available group-level performance summary:",
            str(fairness_summary),
            "",
            "Interpretation:",
            "If performance differs across groups, educators should be especially careful before relying on the prediction."
        ])

    return "\n".join(lines)


def generate_comprehension_check(role: str = "educator") -> pd.DataFrame:
    """
    Creates a short comprehension check.
    """
    role = role.strip().lower()

    common_questions = [
        {
            "question": "Is the model prediction a final decision?",
            "correct_answer": "No. It is a decision-support signal that requires human review."
        },
        {
            "question": "Can the model fully understand personal circumstances?",
            "correct_answer": "No. The model may miss health, family, financial, motivational, or recent life circumstances."
        },
        {
            "question": "Should an 'At-Risk' prediction be used for punishment?",
            "correct_answer": "No. It should be used to offer support, not punishment or exclusion."
        }
    ]

    if role == "student":
        common_questions.append({
            "question": "What can a student do if they disagree with the prediction?",
            "correct_answer": "They can request human review and ask what information contributed to the prediction."
        })

    if role == "educator":
        common_questions.append({
            "question": "What should an educator do before acting on a prediction?",
            "correct_answer": "Review the student's context and combine the model output with human judgment."
        })

    return pd.DataFrame(common_questions)