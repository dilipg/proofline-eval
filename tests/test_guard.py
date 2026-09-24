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
    span = SRC.split("# §4  Retrieval index")[1].split("# §11  Branch")[0]
    assert "class Scorer" in span and "def score_runs" in span


def test_assert_no_truth_leak_runs_on_this_platform():
    pe.assert_no_truth_leak(None)          # IndexError on Windows before the utf-8 fix


def test_pgserver_marker_covers_every_platform_with_a_wheel():
    from packaging.markers import Marker
    deps = [ln for ln in SRC.splitlines()[:15] if "pgserver" in ln]
    marker = Marker(deps[0].split(";", 1)[1].strip().rstrip('",'))
    for plat, machine in (("win32", "AMD64"), ("linux", "x86_64"), ("darwin", "arm64")):
        assert marker.evaluate({"sys_platform": plat, "platform_machine": machine}), (plat, machine)
    assert not marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"})


def test_cli_prints_through_a_pipe():
    # the console's own encoding, not an override: this is what a Windows pipe gets
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    r = subprocess.run([sys.executable, str(Path(pe.__file__)), "scorers"],
                       capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
