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
- **Baseline:** the robot walks forward steadily for the whole 10 s rollout. It reaches 0.95 m/s for the
  1.0 m/s command and never falls.
- **10x energy penalty:** the robot stands still for the whole rollout, even though the policy receives
  the same 1.0 m/s command. It stays upright and uses very little power (about 5 W against 110 W for
  the baseline), but it never takes a real step forward.
- **Training curves:** with the stronger penalty, the return stayed at 0 for the first ~45M steps and
  then settled around 16-18, below the baseline's ~28. The policy found a stable way to collect reward
  by standing, and more training did not move it towards walking.
- **Training length matters too:** in an earlier run with only 50M steps, even the baseline stood still.
  Its return plateaued around 19, and walking only appeared after ~60M steps.

## What I learned
- A reward weight does not just tune the style of a behaviour; it can change which behaviour is learned
  at all. Walking costs energy on every step, while the tracking reward only pays off once the robot is
  already moving close to the commanded speed. With the energy penalty ten times stronger, standing still
  became the easier local optimum, and PPO never left it.
- A higher training return is not the same as the behaviour I want. Both "standing" policies had a
  reasonable return, so I evaluate every policy with a task metric that does not depend on the reward
  weights: the distance actually walked under a fixed command.
- Next I would sweep the energy weight between 1x and 10x to find where walking breaks down, and try a
  curriculum that starts with a small penalty and increases it once the robot already walks. This is how
  the MuJoCo Playground handstand example adds its energy penalty.

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
