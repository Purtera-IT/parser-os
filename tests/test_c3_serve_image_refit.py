"""The T4 image serves a Questions refit only when it was fitted on the packed checkpoint."""
import subprocess
from pathlib import Path

IMG = Path(__file__).resolve().parents[1] / "ml" / "c3" / "serve_image"


def test_start_and_pack_check_the_refit_against_the_checkpoint():
    for name in ("start.sh", "pack_live.sh"):
        text = (IMG / name).read_text()
        assert "stale_refit" in text, name
        subprocess.run(["bash", "-n", str(IMG / name)], check=True)
    start = (IMG / "start.sh").read_text()
    assert "unset SOWSMITH_C3_QUESTION_REFIT" in start
    assert '[[ -f "$DST/c3_base/question_refit.pt" ]] && export SOWSMITH_C3_QUESTION_REFIT' not in start
