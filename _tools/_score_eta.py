# -*- coding: utf-8 -*-
"""Score the panel's ETA against what the compiles actually did.

Samples compile-progress.json at the panel's own poll rate, then replays the
SAME rules the TypeScript uses -- the 45s window, the 4s floor, the two-item
floor, the "stop before the counter does" guard -- and compares each estimate
it would have shown against the stage's real remaining time.

Written because reasoning about this has been wrong twice. The cumulative rate
scored 734% median error on a real compile; a first scoring script measured
from the batch loop instead of from what the panel computes and reported a
comfortable 28-49%, which was not the same question.

    python _tools/_score_eta.py sample 900   # seconds to watch
    python _tools/_score_eta.py score
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from azure.storage.blob import BlobServiceClient  # noqa: E402

CONN = (HERE / ".bloburl").read_text().strip()
CONTAINER = "orbitbrief-artifacts"
OUT = Path(os.environ.get("ETA_SAMPLES", r"D:/temp/claude/eta_newcounter.json"))

# The panel's rules, kept in one place so the score cannot drift from the code.
POLL_MS = 2_000
RATE_WINDOW_MS = 45_000
MIN_WINDOW_MS = 4_000
MIN_ITEMS_IN_WINDOW = 2
NEARLY_DONE_ITEMS = 2  # mirrors compileEta.ts
NEARLY_DONE_FRACTION = 0.1  # mirrors compileEta.ts


def _running_deals() -> list[str]:
    """Only the deals actually compiling.

    The first version of this walked all ~500 deal folders every cycle, so one
    pass took ~50 SECONDS and each deal was really sampled every 50s rather
    than every 2. A stage that lasts two minutes yields two or three readings
    at that rate -- not enough to measure a rate, let alone score one, which is
    why the first scoring run had two usable estimates.
    """
    import urllib.request
    url = ("https://purpulse-dev-api-eus2.azurewebsites.net"
           "/api/proxy/api/data/pm/compile-queue")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            snap = json.loads(r.read())["data"]
        return [j["dealId"] for j in (snap.get("parser") or {}).get("nowPlaying") or []]
    except Exception:
        return []


def sample(seconds: int) -> None:
    cc = BlobServiceClient.from_connection_string(CONN).get_container_client(CONTAINER)
    log, t0 = [], time.time()
    deals, refreshed = _running_deals(), time.time()
    while time.time() - t0 < seconds:
        # Re-ask every 20s: slots change hands and a finished deal is dead
        # weight on a 2s loop.
        if time.time() - refreshed > 20:
            deals, refreshed = _running_deals() or deals, time.time()
        for deal in deals:
            try:
                raw = cc.get_blob_client(
                    f"deals/{deal}/orbitbrief/latest/compile-progress.json"
                ).download_blob().readall()
                d = json.loads(raw)
            except Exception:
                continue
            if d.get("status") not in ("running", "starting", "done"):
                continue
            log.append({
                "deal": deal[:8], "t": round((time.time() - t0) * 1000),
                "compile": str(d.get("compile_id") or "")[:8],
                "status": d.get("status"), "stage": d.get("current_stage"),
                "done": d.get("stage_items_done"), "total": d.get("stage_items_total"),
                "elapsed": d.get("elapsed_ms"),
                "stages": [[s.get("stage_name"), s.get("duration_ms")] for s in (d.get("stages") or [])],
            })
        OUT.write_text(json.dumps(log), encoding="utf-8")
        time.sleep(POLL_MS / 1000)
    print(f"samples: {len(log)} -> {OUT}")


def score() -> None:
    log = json.loads(OUT.read_text(encoding="utf-8"))
    runs: dict[tuple, list] = {}
    for s in log:
        if not s.get("total") or s.get("done") is None:
            continue
        runs.setdefault((s["compile"], s["stage"]), []).append(s)

    # Every reading per compile, in order, so a stage's exit can be seen.
    by_compile: dict[str, list] = {}
    for s in sorted(log, key=lambda x: x["t"]):
        if s.get("compile"):
            by_compile.setdefault(s["compile"], []).append(s)

    errs, shown, silent, rows = [], 0, 0, []
    for (compile_id, stage), samples in runs.items():
        samples.sort(key=lambda x: x["t"])
        # WHEN THE STAGE ENDED IS OBSERVED, NOT DERIVED.
        #
        # This used to be `elapsed_ms - sum(duration_ms of prior stages)`, and
        # that is wrong in the one direction that flatters nothing: elapsed_ms
        # is WALL CLOCK while stages[].duration_ms excludes the gaps between
        # stages, so the subtraction credits every gap to this stage. On a live
        # compile it reported 4 seconds left at 288/1111 when 67 remained, and
        # scored a dead-linear estimator at 334% error.
        #
        # The stage's exit is in the sample stream: the first reading in which
        # this compile is no longer in this stage. Resolution is one sampling
        # interval (~2s), which is noise against the 30s+ this predicts.
        exit_t = None
        for x in by_compile.get(compile_id, ()):
            if x["t"] > samples[-1]["t"] and x.get("stage") != stage:
                exit_t = x["t"]
                break
        if exit_t is None:
            # Stage never observed to finish -- nothing to score against.
            continue
        for i, now in enumerate(samples):
            total, done = now["total"], now["done"]
            left = total - done
            if left <= NEARLY_DONE_ITEMS:
                silent += 1
                continue
            # MIRROR THE PANEL, INCLUDING ITS SILENCE.
            #
            # compileEta.ts refuses the last tenth, because the stage has a
            # cheap tail the counter does not measure. Scoring readings the
            # panel suppresses measures a number nobody is ever shown -- and
            # those readings are the worst ones, so it slanders the estimate.
            if left < total * NEARLY_DONE_FRACTION:
                silent += 1
                continue
            window = [x for x in samples[:i + 1] if now["t"] - x["t"] <= RATE_WINDOW_MS]
            if len(window) < 2:
                silent += 1
                continue
            d_items = window[-1]["done"] - window[0]["done"]
            d_ms = window[-1]["t"] - window[0]["t"]
            if d_items < MIN_ITEMS_IN_WINDOW or d_ms < MIN_WINDOW_MS:
                silent += 1
                continue
            pred = left * (d_ms / d_items)
            actual = exit_t - now["t"]
            if actual < 3_000:
                continue
            shown += 1
            err = abs(pred - actual) / actual
            errs.append(err)
            rows.append((compile_id, stage, done, total, pred / 1000, actual / 1000, err * 100))

    if not errs:
        print("no scoreable estimates in this sample")
        print(f"  (silent at {silent} readings -- that is the estimator refusing, not a failure)")
        return
    errs.sort()
    print(f"estimates the panel would have shown : {shown}")
    print(f"readings where it said nothing       : {silent}")
    print(f"median error : {errs[len(errs)//2]*100:6.1f}%")
    print(f"p25 / p75    : {errs[len(errs)//4]*100:6.1f}% / {errs[int(len(errs)*.75)]*100:6.1f}%")
    # These two lines were LABELLED WRONG: the 0.5 bucket was printed as
    # "within 2x" and the 0.25 bucket as "within 50%", each one factor out.
    # Every figure quoted off this script during the #257/#258/#259 work was a
    # within-50% number called a within-2x number. The comparisons between
    # counter designs were still like-for-like, but the labels were not.
    print(f"within 25%   : {sum(1 for e in errs if e <= 0.25)*100//len(errs)}%")
    print(f"within 50%   : {sum(1 for e in errs if e <= 0.50)*100//len(errs)}%")
    print(f"within 2x    : {sum(1 for e in errs if e <= 1.00)*100//len(errs)}%")
    print("\nworst 8:")
    for r in sorted(rows, key=lambda r: -r[6])[:8]:
        print(f"  {r[0]} {r[1][:26]:26} {r[2]:>5}/{r[3]:<5} pred {r[4]:6.0f}s actual {r[5]:6.0f}s  {r[6]:6.0f}%")


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(int(sys.argv[2]))
    else:
        score()
