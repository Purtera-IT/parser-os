# Running the C3 base on a RunPod pod

Everything lives under `$C3` (default `/root/c3`): `repo/` (this branch),
`raw/` (the labeling page exports `exportNNN.json` and `dealanswersNNN.json`),
`train/` (converted training files), `out/` (logs and checkpoints). The raw
exports hold customer text: they go to the pod only, never into a repo.

1. `bash ml/c3/pod/setup.sh` installs the one missing package, runs the C3
   tests and downloads the encoder and teacher.
2. `bash ml/c3/pod/convert.sh` turns the 20 exports into training files and
   writes `train/split.json` with each file's sha256.
3. `bash ml/c3/pod/preflight.sh` loads every deal, then runs every stage once
   on the 4 largest parts on both GPUs (A frozen encoder, B top 2 blocks
   learning) and records peak GPU memory and seconds per part.
4. `STEPS=<n> bash ml/c3/pod/run.sh` starts Run A on GPU 0 and Run B on GPU 1
   in the background. `tail -f $C3/out/runA.log` follows one.
