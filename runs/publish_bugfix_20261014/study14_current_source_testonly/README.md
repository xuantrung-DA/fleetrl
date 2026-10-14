# Study14 checkpoint transfer test

This is an evaluation launcher, not a training launcher. It evaluates the three
Study14 MAPPO replicas and three DQN replicas, plus heuristic and fixed CP-SAT,
on S1-S9 with five tapes per scenario (360 episodes, 3600 simulated seconds each).
It imports the current repository source and refuses a different source hash.
Checkpoint files and the original Study14 results are read only.

Results are labeled `checkpoint train trên source cũ`. Seeds 3005-3009 have
already been observed; this is a source transfer assessment, not fresh holdout
confirmation. It is separate from the earlier 300-case nonlearning report and
does not include that report's custom stress fixtures or delay sweep.

Run from the repository root:

```powershell
python -u runs/study14_current_source_testonly/run.py
```

Progress is printed per case and written to `status.json`. Results are saved per
episode immediately. At 360 completed episodes, `episodes.csv` and `summary.json`
are generated. Solver timeout may remain unknown; this does not fix telemetry.
The program stops for safety, lost tasks, completion below 50%, p95 above 250 ms,
or a known solver timeout rate of at least 5%. Failure evidence is preserved in
`STOP_REQUIRED.json` and needs review before continuation. Crashes produce a
traceback and retain partial episode artifacts.

Each invocation pauses after approximately 5.5 hours at an episode boundary.
Run the same command to resume a time-limited run. A completed episode is skipped;
partial episodes or a stale lock require review, not automatic deletion.
Do not modify source or run other CPU/GPU-heavy tasks while evaluating.
