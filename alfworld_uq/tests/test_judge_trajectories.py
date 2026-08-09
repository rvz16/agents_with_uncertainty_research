import pytest

from experiments.judge_trajectories import build_judge_prompt, parse_judge_response


def test_prompt_uses_only_visible_transcript_fields() -> None:
    prompt = build_judge_prompt(
        [
            {
                "step": 1,
                "task": "put the apple on the table",
                "thought": "I should take the apple.",
                "action": "take apple 1 from counter 1",
                "observation": "You pick up the apple 1.",
                "final_success": True,
                "done": True,
                "stop_reason": "won",
            }
        ]
    )
    assert "put the apple" in prompt
    assert "take apple" in prompt
    assert "final_success" not in prompt
    assert "stop_reason" not in prompt
    assert "won" not in prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict":"PASS","confidence":0.8,"reason":"done"}', True),
        (
            '```json\n{"verdict":"FAIL","confidence":0.95,"reason":"missing"}\n```',
            False,
        ),
    ],
)
def test_parse_judge_response(text: str, expected: bool) -> None:
    passed, confidence, reason = parse_judge_response(text)
    assert passed is expected
    assert 0.0 <= confidence <= 1.0
    assert reason


def test_parse_judge_response_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        parse_judge_response(
            '{"verdict":"PASS","confidence":1.5,"reason":"invalid"}'
        )
