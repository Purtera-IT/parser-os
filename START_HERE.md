# Labelling session — read this first

This is a **second** parser-os checkout, made so two sessions can work at once
without trampling each other. It is a git worktree of
`C:\Users\lilli\parser-os-registry`, on branch `labeling/session-2`, cut from
`main` at `234e005`.

Your job here is **labelling deals**. Not parser changes, not deploys.

## The one hard rule: do not deploy

Dev is a single mutable pointer and every deploy is last-writer-wins. From the
deploy workflow's own comments:

> When two agents deploy different refs the loser's work vanishes with no
> error anywhere — it looks exactly like a feature that was never built. That
> happened twice in two days.

So from this session, **never**:

- run the `Deploy parser-os-worker to dev` workflow
- run `compile_fast.sh` (deliberately not copied here — it flips
  `SOWSMITH_DEFER_TO_WARM_WORKER` on the Container App JOB, which is global,
  and two sessions doing that fight each other)
- push parser code to `main`

If this deal needs a parser fix, commit it on `labeling/session-2`, push the
branch, and hand the branch name to the other session. It lands there.

## What IS safe in parallel

- Labelling any deal that is not `c79db726-323e-41f8-899d-1d8ca29a579a`
  (010180 belongs to the other session)
- `compile_one.sh` — normal-priority enqueue, the queue handles concurrency
- Reading blob, reading and writing `atom_labels` rows

One caveat: a deploy from the other session swaps the running image for
everyone. If a compile of yours is in flight when that happens it may finish
on a different image than it started on. If a result looks impossible, ask
whether a deploy landed mid-compile, and just re-run.

## The toolkit

Everything in `_tools/`. All of it takes the deal from `$DEAL` — the deal id
used to be a constant in each script, and left that way it would have written
this deal's labels under 010180's rows with no audit catching it.

    export DEAL=<deal-uuid>
    export PG=$(cat _tools/.pgurl)
    export PYTHONPATH=C:/Users/lilli/parser-os-labeling

    python _tools/walk_180.py          # pull the envelope, build walk_180.json
    python _tools/audit_complete.py    # completeness: 7 fields, registry-valid
    python _tools/audit_complete.py --invariants   # the 5 checks that must pass
    bash   _tools/compile_one.sh       # re-compile on whatever is deployed

Write labels by building a pass file and applying it:

    python _tools/write_labels.py pass_01.json           # dry run
    python _tools/write_labels.py pass_01.json --apply

`write_labels.py` rebuilds every column — use `patch_labels.py` for partial
updates, or a keys-only pass will blank the pointers you already wrote.

## What "labelled" means here

`_LABELING_DOCTRINE.md` in this checkout is the authority. The parts that bite
most often:

- **Every label needs all seven fields.** `label_type`, `weight_tier`, `note`,
  `hint_refs`, `entity_keys`, `about`, `wants`. A label that only says what
  something IS teaches a type head and nothing else.
- **A pointer must be literally in its atom.** `hint_refs` quoting text the
  atom does not contain is a fabricated citation.
- **A site key only goes on an atom that names the site.** This was violated
  once at scale — 45 atoms claiming an address they never stated.
- **`_keep` must also set `rejected='true'`.** The workspace styles the red
  row off `rejected`, not off the type. 49 rejections once rendered as
  ordinary rows.
- **A numeric entity key must name a number the atom states.** A cross
  reference between two atoms is a LINK, not a key.

`audit_complete.py --invariants` checks all five. Run it before you call a
deal done.

## Where the other session is

Working 010180 (`c79db726-…`): the CAD/drawing derivation, the PDF-export
withholding, and a `deal_state` consolidation. If you want the reasoning, read
the "A drawing is measured, not listed" section of `_LABELING_DOCTRINE.md`.
