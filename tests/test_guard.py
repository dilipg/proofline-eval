import os
import subprocess
import sys
from pathlib import Path

import proofline_eval as pe

SRC = Path(pe.__file__).read_text(encoding="utf-8")
FRAME = "head\n# §4  Retrieval index\n{body}\n# §11  Branch\ntail\n"


def test_guard_trips_on_every_planted_table():
    for lit in ("true_support", "true_mentions", "true_facts", "true_entities"):
        assert pe.truth_leak_in(FRAME.format(body=f"q = '{lit}'")) == lit


def test_guard_ignores_text_outside_the_span():
    src = "true_facts\n# §4  Retrieval index\nok = 1\n# §11  Branch\ntrue_mentions\n"
    assert pe.truth_leak_in(src) is None


def test_real_source_is_clean_and_span_covers_scoring():
    assert pe.truth_leak_in(SRC) is None
    span = pe.guarded_span(SRC)
    # banner LINE to banner LINE: the whole of §4..§10, the guard's own code included
    for name in ("class Scorer", "def score_runs", "def hard_checks", "def build_index",
                 "def assert_no_truth_leak"):
        assert name in span, name
    assert "def branch1_labels" not in span


def test_a_renamed_banner_fails_loudly():
    # a renamed §11 banner stretches the span into B1's planted reads, which trip it
    assert pe.truth_leak_in("x\n# §4  Retrieval index\nok\n# §11 Branch\nq = 'true_facts'\n") == "true_facts"
    # a renamed §4 banner leaves nothing to check: that must stop the run, not pass it
    import pytest
    with pytest.raises(SystemExit):
        pe.truth_leak_in("x\n# §4 Retrieval index\nok\n# §11  Branch\n")


def test_assert_no_truth_leak_runs_on_this_platform():
    pe.assert_no_truth_leak(None)          # IndexError on Windows before the utf-8 fix


def test_pgserver_marker_covers_every_platform_with_a_wheel():
    from packaging.markers import Marker
    for script in ("proofline_eval.py", "ingest_semantica.py"):
        head = (Path(pe.__file__).parent / script).read_text(encoding="utf-8").splitlines()[:15]
        deps = [ln for ln in head if "pgserver" in ln]
        marker = Marker(deps[0].split(";", 1)[1].strip().rstrip('",'))
        for plat, machine in (("win32", "AMD64"), ("linux", "x86_64"), ("darwin", "arm64")):
            assert marker.evaluate({"sys_platform": plat, "platform_machine": machine}), (script, plat)
        assert not marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"})


def test_cli_prints_through_a_pipe():
    # the console's own encoding, not an override: this is what a Windows pipe gets
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    r = subprocess.run([sys.executable, str(Path(pe.__file__)), "scorers"],
                       capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
