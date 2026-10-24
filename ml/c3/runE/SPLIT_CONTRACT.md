# Verified main17 split policy

`split_contract.json` version1 pins the owner-approved Run E split. Its canonical
JSON SHA256 is checked by `ml.c3.holdout.load_split_contract`. A new or modified
policy requires an explicit code/contract review; it cannot silently change a
training/evaluation role through an arbitrary JSON file.

- Reserved tests: 000026, 010269, 010155, 010358.
- Golden tests: 010038, 010387.
- Selection only: 010043. It cannot train or be reused as a final holdout test.
- Pending test: 010279, protected before its draft is ready.
- Published Deal Kit tests remain protected: 000051, 010095, 010164, 010253, 010287.
- 010215 was explicitly released to training by the owner; its old hard-coded
  holdout membership is removed. This does not approve its data or INIT exposure.

`C3_SPLIT_CONTRACT`, if set, must resolve to the exact known semantic contract.
An empty path, missing file, unknown version/role/policy, duplicate/overlapping
role or modified role set fails closed. Without an override the bundled verified
contract is used, including on hosts without customer-local files.

`C3_HOLDOUT2`, if set, must exist and match both the reserved `deals` and `pending`
sets. Otherwise a present `~/c3-local/runE/holdout2.json` is checked the same way.
Malformed, stale or unknown manifests fail closed; no failed explicit manifest
falls back to defaults. If no optional local file exists, the verified bundled
policy still protects the same known tests and selection source.

Training guards protect every view/window of these sources. `lock_validation`
allows selection in its proper role while refusing actual tests; `lock_split`
continues to refuse source reuse across train/test and selection/test. Existing
train/validation disjointness checks remain active. Split policy is not a data
allowlist: the main launch must still bind the exact approved31 train paths and
2 validation paths, actual source/task admission, holds, balance and artifact
hashes. Unknown source data is not approved by absence from a holdout set.

This successor changes only the split guard. Maxwell must bind its exact code
SHA and split-contract hash in a refreshed freeze/code receipt without editing
customer sources or silently relabeling prior-SHA receipts. Teacher/WHY/card/link
budgets stay nonzero; teacher artifact compatibility, INIT admissibility/load,
description, real signal/bypass/runtime checks and all postrun gates remain
required. This policy does not resolve the missing real teacher or authorize a
local/external GPU run.
