"""
Text Moderation Evaluation Suite
...
"""

import sys
from pathlib import Path
from typing import List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
import tenacity
from pydantic_ai.retries import RetryConfig

from multimodal_moderation.agents.text_agent import moderate_text
from multimodal_moderation.types.moderation_result import TextModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import TextModerationCheck

judge_model, judge_settings = get_judge_model()


class TextInput(BaseModel):
    text_file: str = Field(description="Path to text file to moderate")


async def run_text_moderation(inputs: List[TextInput]) -> TextModerationResult:
    assert len(inputs) == 1, "Text moderation expects exactly one input"
    text = Path(inputs[0].text_file).read_text()
    model_choice = get_model_under_test()
    return await moderate_text(model_choice, text)


cases: List[Case[List[TextInput], TextModerationResult, Any]] = [
    Case(
        name="professional_text",
        inputs=[TextInput(text_file=get_test_data_path("professional_text.txt"))],
        metadata={"category": "text_moderation"},
        evaluators=(
            TextModerationCheck(
                expected_pii=False,
                expected_unfriendly=False,
                expected_unprofessional=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the text is professional and friendly, with no flags raised.",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="text_with_pii",
        inputs=[TextInput(text_file=get_test_data_path("text_with_pii.txt"))],
        metadata={"category": "text_moderation"},
        evaluators=(
            TextModerationCheck(
                expected_pii=True,
                expected_unfriendly=False,
                expected_unprofessional=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should identify specific PII items (name, address, email, phone number, account number)",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="unfriendly_text",
        inputs=[TextInput(text_file=get_test_data_path("unfriendly_text.txt"))],
        metadata={"category": "text_moderation"},
        evaluators=(
            TextModerationCheck(
                expected_pii=False,
                expected_unfriendly=True,
                expected_unprofessional=True,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the tone is unfriendly and unprofessional, citing specific problematic phrases",
                include_input=True,
            ),
        ),
    ),
]


text_dataset = Dataset[List[TextInput], TextModerationResult, Any](
    cases=create_repeated_cases(cases),
    evaluators=[
        IsInstance(type_name="TextModerationResult"),
        HasRationale(),
    ],
)


async def main():
    retry_config = RetryConfig(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_full_jitter(multiplier=0.5, max=15),
    )

    report = await text_dataset.evaluate(
        run_text_moderation,
        retry_task=retry_config,
        retry_evaluators=retry_config,
    )

    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())