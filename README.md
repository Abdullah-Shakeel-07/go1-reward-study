# How reward terms shape a quadruped gait (Unitree Go1, MuJoCo Playground)

A small study I did to learn reinforcement learning for legged robots. I trained the
same velocity-tracking policy (PPO, `Go1JoystickFlatTerrain` from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)) several times,
each time changing some reward weights, and compared the resulting gaits.

## Setup
- Environment: `Go1JoystickFlatTerrain`, PPO from Brax, default domain randomization
- Training: 200M environment steps per run on Kaggle (2x T4 GPU), about 30 min per run
- Evaluation: the same 1.0 m/s forward command for every policy, 10 s rollout. I measure the
  distance actually walked (from the robot's position), velocity-tracking error, and mean
  mechanical power, and record a video. Training rewards are not compared, since each run
  uses different reward weights.

## Experiments
| Experiment | Change | Speed reached (m/s, target 1.0) | Velocity error (m/s) | Power (W) | Fell? |
|---|---|---|---|---|---|
| baseline | default weights | 0.95 | 0.08 | 110 | no |
| energy_x10 | energy and torque penalties x10 | 0.005 (stands still) | 1.00 | 5 | no |

Training reward (episode return) during training:

| Steps (M) | 0 | 23 | 46 | 69 | 92 | 115 | 138 | 161 | 184 | 206 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0 | 18.5 | 19.4 | 25.9 | 27.0 | 27.3 | 27.5 | 28.2 | 28.5 | 28.6 |
| energy_x10 | 0.0 | 0.0 | 0.0 | 15.6 | 16.5 | 16.3 | 16.8 | 17.2 | 17.4 | 18.0 |

An earlier run with only 50M steps gave a baseline that also stood still (0.005 m/s): the return
plateaued around 19 before walking emerged after ~60M steps.

Videos are in `videos/`, raw numbers in `results/results.json`, trained weights in `params/`.

## What I observed
[FILL IN YOUR OWN WORDS after watching the videos. E.g. what happened to the gait without
feet rewards? Did the energy penalty make it smoother or slower? Did anything fail?]

## What I learned
[FILL: 2-4 sentences. What surprised you, and what you would try next.]

## Reproduce
Open the official
[locomotion notebook](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/rss2025/learning/notebooks/locomotion.ipynb),
add two cells:
1. Right after "Install pre-requisites": `!pip install -q -U "flax>=0.12.12"` (Colab's JAX 0.11 needs a newer Flax).
2. Right after "Import The Playground": the contents of `colab_compat_fixes.py` (uses the JAX backend instead of Warp,
   and adds the JAX-docs replacements for `jax.device_put_replicated`, which brax still calls).

Then run the notebook up to the `train_fn` cell, and run `ablation_cell.py` in a new cell followed by:

```python
run_all({"baseline": {}}, 5_000_000)      # quick test
results = run_all(EXPERIMENTS, 50_000_000)  # full study
```
