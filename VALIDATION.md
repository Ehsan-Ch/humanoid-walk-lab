# Validation record

Date: 2026-10-15. Windows, Python 3.12.14, MuJoCo 3.3.7, NumPy 2.5.4. The Microsoft C++ dependency was loaded from an existing trusted runtime directory for the Python process; no system installation or DLL replacement was needed.

## Training and evaluation

- One real CEM training run, optimizer seed 42, 50 generations, 48 candidates, two simulation seeds: 4,800 rollouts. Recorded wall time: 99.415 seconds with four CPU workers.
- Frozen policy evaluated on 100 trials: 20 initial-controller nominal trials, and 20 learned-controller trials for each of nominal, low-friction, forward-push, and backward-push conditions.
- Learned nominal: 20/20 walking successes, mean distance 7.0966 m, mean speed 0.5914 m/s over twelve seconds. Initial nominal: 0/20, all fell.
- Learned low-friction and forward-push: 20/20 each. Backward-push: 0/20, all fell, mean duration 3.108 seconds.
- All results and failures are included. Success criteria were fixed before the held-out evaluation. No post-evaluation retraining was performed.
- Videos render the first seed in each group from recorded MuJoCo states. The report is a recorded experiment viewer, not a live simulator.

## Software checks

The editable package installed successfully in an isolated environment. All 22 tests passed. Tests cover the unactuated base, torque limits, deterministic reset/rollout, fall termination, replay states, invalid inputs, disturbance timing, and walking-success criteria. Browser checks confirmed that normal walking and backward-push videos decode, playback advances, scenario selection updates metrics/charts, and the failure is labeled correctly. The report layout and plots were visually inspected. GitHub Actions run [38709771316](https://github.com/Ehsan-Ch/humanoid-walk-lab/actions/runs/38709771316) was blocked before any steps started. GitHub reported: "The job was not started because your account is locked due to a billing issue." This is not a failing test result; only local test execution is verified. The workflow can be rerun after the account issue is resolved. The remaining baseline and low-friction videos also decoded in the browser without media errors; no browser warnings/errors were recorded.
