"""
Phase 7 — Prefect Pipeline Orchestration
Runs the full pipeline (ingest → preprocess → embed → eval agent) as a scheduled Prefect flow.

Start Prefect server first:
    prefect server start

Then run the pipeline:
    python pipelines/full_pipeline.py

View in Prefect UI:
    http://localhost:4200
"""

import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger


@task(name="Ingest Raw Data", retries=2, retry_delay_seconds=10)
def ingest_data():
    logger = get_run_logger()
    logger.info("📥 Ingesting raw data...")
    from src.data.ingest import ingest
    ingest()
    logger.info("✅ Ingestion complete")


@task(name="Preprocess Data", retries=1)
def preprocess_data():
    logger = get_run_logger()
    logger.info("⚙️  Preprocessing data...")
    from src.data.preprocess import main
    main()
    logger.info("✅ Preprocessing complete")


@task(name="Build Vector Embeddings", retries=1)
def build_embeddings():
    logger = get_run_logger()
    logger.info("🔢 Building ChromaDB embeddings...")
    from src.data.embed import build_vector_store
    build_vector_store()
    logger.info("✅ Embeddings built")


@task(name="Prepare Fine-tune Dataset")
def prepare_finetune():
    logger = get_run_logger()
    logger.info("📚 Preparing fine-tune dataset...")
    from src.training.prepare_dataset import prepare
    dataset = prepare()
    logger.info(f"✅ Fine-tune dataset ready: {len(dataset)} examples")
    return len(dataset)


@task(name="Smoke Test Agent", retries=1)
def smoke_test_agent():
    """Quick sanity check that the agent can respond without errors."""
    logger = get_run_logger()
    logger.info("🤖 Running agent smoke test...")
    try:
        from src.agents.rag_agent import ask_agent
        result = ask_agent("What is RAG?", session_id="pipeline-smoke-test")
        answer = result["answer"]
        assert len(answer) > 10, "Answer too short — something is wrong"
        logger.info(f"✅ Agent smoke test passed. Answer preview: {answer[:120]}...")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Agent smoke test failed: {e}")
        return False


@flow(
    name="AI MLOps Full Pipeline",
    description="End-to-end: ingest → preprocess → embed → prepare training data → smoke test agent",
    log_prints=True,
)
def full_pipeline():
    logger = get_run_logger()
    logger.info("🚀 Starting AI MLOps pipeline...")

    # Run data stages sequentially (each depends on the previous)
    ingest_data()
    preprocess_data()
    build_embeddings()

    # These two can run once embeddings are ready
    prepare_finetune()
    passed = smoke_test_agent()

    if passed:
        logger.info("🎉 Pipeline complete — all stages passed!")
    else:
        logger.warning("⚠️  Pipeline complete — agent smoke test had issues. Check logs.")


if __name__ == "__main__":
    # Run once immediately
    full_pipeline()

    # To schedule: uncomment below and run as a long-running server
    # full_pipeline.serve(
    #     name="nightly-ai-mlops",
    #     cron="0 2 * * *",        # Run at 2am every night
    # )
