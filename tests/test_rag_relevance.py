"""
Phase 2.5: RAG retrieval reliability (src/rag/rag.py, src/rag/config.py).

Before: every retrieved document (top_k, any distance) went to the LLM,
so an off-topic question still received unrelated Sentinel context.
Now results whose cosine DISTANCE is above RAG_MAX_DISTANCE (default
0.85; smaller = more relevant) are dropped, identical documents are kept
once, and each context block carries its source id. With nothing relevant
left, ask() returns the existing "not available" answer without calling
the LLM (unchanged path). The LLM and retriever are fakes: no NVIDIA call.

CALIBRATION below is the measured distance table from the live index
(nvidia/nemotron-3-embed-1b, collection sentinel_knowledge, 2026-10-07):
for each query, the distance of each retrieved document. The regression
tests replay it through the real filter.
"""

import importlib.util
import os
import sys
import types

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

UNAVAILABLE = ("The requested information is not available "
               "in the Sentinel-AI knowledge base.")

DOCS = {
    "monitoring_run:7": "Sentinel-AI Monitoring Run 7.\nRecommended action: WAIT.",
    "monitoring_run:8": "Sentinel-AI Monitoring Run 8.\nRecommended action: WAIT.",
    "monitoring_run:9": "Sentinel-AI Monitoring Run 9.\nOverall drift score: 0.000.",
    "sentinel_chunk_0": "Sentinel-AI drift monitoring report: {\"action\": \"WAIT\"}.",
    "retraining_event:2": "Sentinel-AI Retraining Event 2.\nPromotion decision: REJECTED.",
}

SOURCES = {
    "monitoring_run": "monitoring_runs",
    "retraining_event": "retraining_events",
    "sentinel_chunk": "reports/drift_summary.json",
}

# query -> (doc id, measured cosine distance), in retrieval order
CALIBRATION = {
    "legit": {
        "What is the current drift status?": [
            ("monitoring_run:8", 0.591), ("monitoring_run:7", 0.601),
            ("monitoring_run:9", 0.602), ("sentinel_chunk_0", 0.622),
            ("retraining_event:2", 0.954)],
        "वर्तमान ड्रिफ्ट स्थिति क्या है?": [
            ("sentinel_chunk_0", 0.711), ("monitoring_run:8", 0.713),
            ("monitoring_run:9", 0.719), ("monitoring_run:7", 0.722),
            ("retraining_event:2", 0.979)],
        "Quel est l'état actuel du drift ?": [
            ("monitoring_run:8", 0.649), ("monitoring_run:7", 0.663),
            ("monitoring_run:9", 0.665), ("sentinel_chunk_0", 0.672),
            ("retraining_event:2", 0.978)],
        "Why was the challenger rejected?": [
            ("retraining_event:2", 0.691), ("monitoring_run:9", 0.944),
            ("monitoring_run:7", 0.949), ("sentinel_chunk_0", 0.953),
            ("monitoring_run:8", 0.956)],
        "What was the recommended action in the latest monitoring run?": [
            ("monitoring_run:7", 0.644), ("monitoring_run:9", 0.646),
            ("monitoring_run:8", 0.647), ("sentinel_chunk_0", 0.772),
            ("retraining_event:2", 0.870)],
        "What is the overall drift score of monitoring run 9?": [
            ("monitoring_run:9", 0.354), ("monitoring_run:8", 0.485),
            ("monitoring_run:7", 0.495), ("sentinel_chunk_0", 0.563),
            ("retraining_event:2", 0.895)],
        "चैलेंजर मॉडल को क्यों अस्वीकार किया गया?": [
            ("retraining_event:2", 0.704), ("sentinel_chunk_0", 0.959),
            ("monitoring_run:9", 0.964), ("monitoring_run:7", 0.968),
            ("monitoring_run:8", 0.969)],
        "Pourquoi le modèle challenger a-t-il été rejeté ?": [
            ("retraining_event:2", 0.692), ("monitoring_run:9", 0.943),
            ("sentinel_chunk_0", 0.949), ("monitoring_run:7", 0.949),
            ("monitoring_run:8", 0.950)],
        "What was the RMSE of the challenger model in retraining event 2?": [
            ("retraining_event:2", 0.423), ("monitoring_run:8", 0.877),
            ("monitoring_run:7", 0.879), ("monitoring_run:9", 0.884),
            ("sentinel_chunk_0", 0.910)],
        "Did the last retraining get promoted?": [
            ("retraining_event:2", 0.624), ("sentinel_chunk_0", 0.946),
            ("monitoring_run:7", 0.946), ("monitoring_run:8", 0.949),
            ("monitoring_run:9", 0.950)],
    },
    "junk": {
        "What is the capital of France?": [
            ("monitoring_run:7", 1.033), ("monitoring_run:8", 1.033),
            ("monitoring_run:9", 1.033), ("sentinel_chunk_0", 1.037),
            ("retraining_event:2", 1.038)],
        "Give me a recipe for chocolate cake.": [
            ("retraining_event:2", 0.940), ("monitoring_run:8", 0.973),
            ("monitoring_run:7", 0.992), ("sentinel_chunk_0", 0.993),
            ("monitoring_run:9", 0.994)],
        "Who won the FIFA World Cup in 2022?": [
            ("retraining_event:2", 0.954), ("sentinel_chunk_0", 0.974),
            ("monitoring_run:8", 0.976), ("monitoring_run:7", 0.977),
            ("monitoring_run:9", 0.988)],
        "How do I change a car tyre?": [
            ("retraining_event:2", 1.000), ("monitoring_run:9", 1.004),
            ("sentinel_chunk_0", 1.005), ("monitoring_run:7", 1.016),
            ("monitoring_run:8", 1.017)],
        "Write a poem about the ocean.": [
            ("sentinel_chunk_0", 0.939), ("retraining_event:2", 0.946),
            ("monitoring_run:8", 0.953), ("monitoring_run:9", 0.956),
            ("monitoring_run:7", 0.969)],
        "What is the weather in Mumbai today?": [
            ("sentinel_chunk_0", 0.932), ("monitoring_run:9", 0.943),
            ("retraining_event:2", 0.945), ("monitoring_run:8", 0.949),
            ("monitoring_run:7", 0.950)],
        "Explain quantum entanglement.": [
            ("sentinel_chunk_0", 0.977), ("monitoring_run:9", 0.991),
            ("monitoring_run:8", 1.000), ("monitoring_run:7", 1.001),
            ("retraining_event:2", 1.025)],
        "मुझे पास्ता बनाने की विधि बताओ": [
            ("monitoring_run:8", 0.997), ("monitoring_run:7", 0.999),
            ("monitoring_run:9", 1.000), ("retraining_event:2", 1.005),
            ("sentinel_chunk_0", 1.006)],
    },
}

# What each legitimate question needs in its context to be answerable
REQUIRED = {
    "What is the current drift status?": {"monitoring_run:8"},
    "वर्तमान ड्रिफ्ट स्थिति क्या है?": {"sentinel_chunk_0", "monitoring_run:8"},
    "Quel est l'état actuel du drift ?": {"monitoring_run:8"},
    "Why was the challenger rejected?": {"retraining_event:2"},
    "What was the recommended action in the latest monitoring run?":
        {"monitoring_run:7", "monitoring_run:8", "monitoring_run:9"},
    "What is the overall drift score of monitoring run 9?": {"monitoring_run:9"},
    "चैलेंजर मॉडल को क्यों अस्वीकार किया गया?": {"retraining_event:2"},
    "Pourquoi le modèle challenger a-t-il été rejeté ?": {"retraining_event:2"},
    "What was the RMSE of the challenger model in retraining event 2?":
        {"retraining_event:2"},
    "Did the last retraining get promoted?": {"retraining_event:2"},
}


def _result(doc_id, distance, document=None):
    return {
        "id": doc_id,
        "document": DOCS.get(doc_id, document) if document is None else document,
        "distance": distance,
        "metadata": {"source": next(
            source for prefix, source in SOURCES.items() if doc_id.startswith(prefix))},
    }


def _results(query):
    table = CALIBRATION["legit"] if query in CALIBRATION["legit"] else CALIBRATION["junk"]
    return [_result(doc_id, distance) for doc_id, distance in table[query]]


# ============================================================
# Fixtures: real rag.py over a fake retriever and fake LLM
# ============================================================

class FakeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, question, context):
        self.calls.append({"question": question, "context": context})
        return f"answer from {len(context)} chars"


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def rag(monkeypatch, llm):

    retrieved = {"results": []}

    fake_retriever = types.ModuleType("src.rag.retriever")
    fake_retriever.retrieve = lambda query, top_k: retrieved["results"][:top_k]

    fake_llm = types.ModuleType("src.rag.llm")
    fake_llm.generate_answer = llm

    monkeypatch.setitem(sys.modules, "src.rag.retriever", fake_retriever)
    monkeypatch.setitem(sys.modules, "src.rag.llm", fake_llm)

    spec = importlib.util.spec_from_file_location(
        "sentinel_real_rag_relevance", os.path.join(REPO_ROOT, "src", "rag", "rag.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.retrieved = retrieved
    return module


def _load_config(monkeypatch, raw=None):

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    if raw is None:
        monkeypatch.delenv("RAG_MAX_DISTANCE", raising=False)
    else:
        monkeypatch.setenv("RAG_MAX_DISTANCE", raw)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-FAKE-for-config-validation")

    spec = importlib.util.spec_from_file_location(
        "sentinel_test_rag_config", os.path.join(REPO_ROOT, "src", "rag", "config.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Config
# ============================================================

def test_default_cutoff_is_documented_value(monkeypatch):

    config = _load_config(monkeypatch)

    assert config.RAG_MAX_DISTANCE == 0.85
    config.validate_config()


def test_cutoff_is_tunable_without_code_changes(monkeypatch):

    assert _load_config(monkeypatch, "1.2").RAG_MAX_DISTANCE == 1.2


@pytest.mark.parametrize("raw", ["0", "-0.1", "2.5"])
def test_out_of_range_cutoff_rejected(monkeypatch, raw):

    config = _load_config(monkeypatch, raw)

    with pytest.raises(ValueError, match="RAG_MAX_DISTANCE"):
        config.validate_config()


def test_collection_is_cosine_so_distance_smaller_is_better():

    with open(os.path.join(REPO_ROOT, "src", "rag", "chroma_store.py"), encoding="utf-8") as f:
        source = f.read()

    assert source.count('metadata={"hnsw:space": "cosine"}') == 2


# ============================================================
# Filter: direction, boundary, dedupe, metadata
# ============================================================

def test_relevant_kept_irrelevant_excluded(rag):

    kept = rag.select_relevant(
        [_result("monitoring_run:9", 0.35), _result("retraining_event:2", 0.95)],
        max_distance=0.85,
    )

    assert [r["id"] for r in kept] == ["monitoring_run:9"]


@pytest.mark.parametrize("distance, kept", [
    (0.849999, True), (0.85, True), (0.850001, False),
])
def test_boundary_is_inclusive(rag, distance, kept):

    result = rag.select_relevant([_result("monitoring_run:9", distance)], max_distance=0.85)

    assert bool(result) is kept


def test_missing_distance_is_not_trusted(rag):

    assert rag.select_relevant([_result("monitoring_run:9", None)], 0.85) == []


def test_duplicates_kept_once_closest_first(rag):

    same = DOCS["monitoring_run:9"]
    kept = rag.select_relevant([
        _result("monitoring_run:9", 0.30),
        {**_result("monitoring_run:9", 0.40), "id": "sentinel_chunk_9", "document": same + "  "},
        _result("monitoring_run:8", 0.50),
    ], 0.85)

    assert [r["id"] for r in kept] == ["monitoring_run:9", "monitoring_run:8"]
    assert kept[0]["distance"] == 0.30


def test_metadata_and_ids_preserved(rag):

    original = [_result("retraining_event:2", 0.42), _result("sentinel_chunk_0", 0.60)]
    kept = rag.select_relevant(original, 0.85)

    assert kept == original
    assert kept[0]["metadata"] == {"source": "retraining_events"}
    assert kept[1]["metadata"] == {"source": "reports/drift_summary.json"}


def test_context_labels_carry_source_ids(rag):

    context = rag.build_context([_result("retraining_event:2", 0.4),
                                 _result("monitoring_run:9", 0.5)])

    assert context == (
        f"[Source 1: retraining_event:2]\n{DOCS['retraining_event:2']}\n\n"
        f"[Source 2: monitoring_run:9]\n{DOCS['monitoring_run:9']}"
    )


def test_context_label_without_id_unchanged(rag):

    assert rag.build_context([{"document": "x"}]) == "[Source 1]\nx"


# ============================================================
# ask(): unavailable path, top_k, LLM only with useful context
# ============================================================

def test_no_relevant_docs_returns_unavailable_without_llm(rag, llm):

    rag.retrieved["results"] = _results("What is the capital of France?")

    assert rag.ask("What is the capital of France?") == UNAVAILABLE
    assert llm.calls == []


def test_empty_index_still_unavailable(rag, llm):

    rag.retrieved["results"] = []

    assert rag.ask("What is the current drift status?") == UNAVAILABLE
    assert llm.calls == []


def test_top_k_respected(rag, llm):

    rag.retrieved["results"] = _results("What is the current drift status?")

    rag.ask("What is the current drift status?", top_k=2)

    context = llm.calls[0]["context"]
    assert "monitoring_run:8" in context and "monitoring_run:7" in context
    assert "monitoring_run:9" not in context
    assert context.count("[Source") == 2


def test_sub_threshold_docs_never_reach_llm(rag, llm):

    rag.retrieved["results"] = _results("Why was the challenger rejected?")

    rag.ask("Why was the challenger rejected?")

    context = llm.calls[0]["context"]
    assert context == f"[Source 1: retraining_event:2]\n{DOCS['retraining_event:2']}"


def test_question_passed_through_unchanged(rag, llm):

    rag.retrieved["results"] = _results("Quel est l'état actuel du drift ?")

    rag.ask("  Quel est l'état actuel du drift ?  ")

    assert llm.calls[0]["question"] == "Quel est l'état actuel du drift ?"


def test_system_instruction_unchanged():

    with open(os.path.join(REPO_ROOT, "src", "rag", "llm.py"), encoding="utf-8") as f:
        source = f.read()

    for line in (
        '"Answer using only the provided context. "',
        '"If the context does not contain the answer, "',
        '"say that the information is unavailable. "',
        '"Respond in the same language as the user\'s question. "',
    ):
        assert line in source


# ============================================================
# Regression: measured live distances through the real filter
# ============================================================

@pytest.mark.parametrize("query", sorted(CALIBRATION["legit"]))
def test_existing_working_queries_keep_their_answers(rag, llm, monkeypatch, query):

    config = _load_config(monkeypatch)          # the shipped default
    rag.retrieved["results"] = _results(query)

    kept = rag.select_relevant(_results(query), config.RAG_MAX_DISTANCE)
    answer = rag.ask(query)

    assert answer != UNAVAILABLE
    assert REQUIRED[query] <= {r["id"] for r in kept}
    for doc_id in REQUIRED[query]:
        assert doc_id in llm.calls[-1]["context"]


@pytest.mark.parametrize("query", sorted(CALIBRATION["junk"]))
def test_off_topic_queries_get_unavailable(rag, llm, monkeypatch, query):

    config = _load_config(monkeypatch)
    rag.retrieved["results"] = _results(query)

    assert rag.select_relevant(_results(query), config.RAG_MAX_DISTANCE) == []
    assert rag.ask(query) == UNAVAILABLE
    assert llm.calls == []


def test_default_sits_inside_the_measured_gap(monkeypatch):

    cutoff = _load_config(monkeypatch).RAG_MAX_DISTANCE

    worst_legit = max(
        min(d for _, d in hits) for hits in CALIBRATION["legit"].values())
    best_junk = min(
        min(d for _, d in hits) for hits in CALIBRATION["junk"].values())

    assert worst_legit == 0.711 and best_junk == 0.932
    assert worst_legit < cutoff < best_junk
