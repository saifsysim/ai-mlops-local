"""
Phase 5 — Agent Evaluation with DeepEval
Tests the RAG agent for answer relevancy and faithfulness (no hallucination).

Run with:
    deepeval test run tests/test_agent_evals.py

Or with pytest:
    pytest tests/test_agent_evals.py -v

Metrics:
  - AnswerRelevancyMetric: Is the answer relevant to the question?
  - FaithfulnessMetric:    Is the answer grounded in retrieved context?
  - ContextualRecallMetric: Did the agent retrieve the right documents?
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Use a lightweight local model for evaluation (can be different from the agent model)
from deepeval.models import OllamaModel

eval_model = OllamaModel(model="llama3.2")


def run_agent(question: str) -> tuple[str, list[str]]:
    """Helper: run the agent and return the answer + retrieved context chunks."""
    from src.agents.rag_agent import ask_agent
    from src.data.embed import query_vector_store

    result = ask_agent(question)
    retrieved = query_vector_store(question, n_results=3)
    context = [r["text"] for r in retrieved]
    return result["answer"], context


# ── Test 1: Answer Relevancy ───────────────────────────────────────────────────
@pytest.mark.parametrize("question", [
    "What is RAG and why is it useful?",
    "How does LangFuse help with debugging agents?",
    "What is the difference between RAG and fine-tuning?",
])
def test_answer_is_relevant(question: str):
    """The agent's answer must be relevant to the question asked."""
    answer, context = run_agent(question)
    metric = AnswerRelevancyMetric(threshold=0.6, model=eval_model)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    assert_test(test_case, [metric])


# ── Test 2: Faithfulness (no hallucination) ────────────────────────────────────
def test_no_hallucination_on_rag_question():
    """The agent must not fabricate facts not present in the retrieved context."""
    question = "What is Retrieval-Augmented Generation?"
    answer, context = run_agent(question)

    metric = FaithfulnessMetric(threshold=0.7, model=eval_model)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    assert_test(test_case, [metric])


# ── Test 3: Agent gracefully handles unknown topics ────────────────────────────
def test_agent_handles_unknown_topic():
    """The agent should say it doesn't know rather than hallucinate."""
    question = "What is the GDP of Mars in 2024?"
    answer, _ = run_agent(question)
    # The answer should indicate uncertainty, not make up a number
    uncertainty_phrases = [
        "don't have", "not in", "no information", "cannot find",
        "not enough", "outside", "not available",
    ]
    answer_lower = answer.lower()
    found = any(phrase in answer_lower for phrase in uncertainty_phrases)
    assert found, (
        f"Expected agent to express uncertainty, but got: '{answer[:200]}'"
    )
