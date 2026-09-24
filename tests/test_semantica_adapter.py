"""The adapter against the real, pinned Semantica. Skipped where Semantica is not
installed (the default test command); run with it added:

    uv run --python 3.12 --with pytest --with "numpy>=1.26" --with "psycopg[binary]>=3.1" \
      --with "pgserver>=0.1.4" --with "semantica[llm-anthropic,viz]==0.7.0" pytest tests -q
"""
import pytest

pytest.importorskip("semantica")

import ingest_semantica as ing  # noqa: E402


def test_regex_is_case_sensitive():
    # Semantica's regex method forces re.IGNORECASE; unscoped, the Method pattern
    # matches "scores 0" in "scores 0.74".
    x = ing.SemanticaAdapter("regex", None).extract(
        "c", "on Brakus-Bench , Kavrel-7 scores 0.74 vex-score . we measure 0.61 for KA-7 .")
    assert x.error is None
    assert sorted(set(x.entities)) == [("Brakus-Bench", "Dataset"), ("KA-7", "Method"),
                                       ("Kavrel-7", "Method"), ("vex-score", "Metric")]


def test_no_fallback_entities_when_nothing_matches():
    # With no regex match, Semantica silently falls back to its built-in pattern NER,
    # which labels a Title Case card title PERSON.
    x = ing.SemanticaAdapter("regex", None).extract(
        "c", "Metaulaance Subibraency Gratilosis . we observe that the modrievment holds .")
    assert x.error is None and x.entities == []
