# Go1 reward ablation — paste this as ONE new cell in the official locomotion Colab,
# after you have run the cells up to and including the "train_fn = functools.partial(...)" cell
# (the cell right after "### Train"). You can skip the original training cell.

import json, os
os.makedirs("results", exist_ok=True)
os.makedirs("videos", exist_ok=True)

NUM_TIMESTEPS = 50_000_000   # lower = faster. Try 20_000_000 first to check everything runs.

# Each experiment changes some reward weights. Default weights are printed by `env_cfg`.
EXPERIMENTS = {
    "baseline": {},
    "no_feet_rewards": {"feet_air_time": 0.0, "feet_clearance": 0.0,
                        "feet_height": 0.0, "feet_slip": 0.0},
    "energy_x10": {"energy": -0.01, "torques": -0.002},
    "no_orientation": {"orientation": 0.0},
}


def make_cfg(overrides):
  cfg = registry.get_default_config(env_name)
  for term, value in overrides.items():
    assert term in cfg.reward_config.scales, f"unknown reward term {term}"
    setattr(cfg.reward_config.scales, term, value)
  return cfg


def train_one(name, cfg):
  curve = []
  def progress(num_steps, metrics):
    curve.append((int(num_steps), float(metrics["eval/episode_reward"])))
    print(f"[{name}] {num_steps:,} steps  reward={curve[-1][1]:.2f}")
  params_dict = dict(ppo_training_params)
  params_dict["num_timesteps"] = NUM_TIMESTEPS
  fn = functools.partial(ppo.train, **params_dict, network_factory=network_factory,
                         randomization_fn=randomizer, progress_fn=progress)
  make_inf, params, _ = fn(environment=registry.load(env_name, config=cfg),
                           eval_env=registry.load(env_name, config=cfg),
                           wrap_env_fn=wrapper.wrap_for_brax_training)
  return make_inf, params, curve


def _rollout(ev, policy, cmd, steps, seed=0):
  reset, step = jax.jit(ev.reset), jax.jit(ev.step)
  rng = jax.random.PRNGKey(seed)
  command = jp.array(cmd)
  state = reset(rng)
  state.info["command"] = command
  traj, vel_err, power, fell, obs_cmd = [], [], [], False, []
  start_xy = np.array(state.data.qpos[:2])
  for t in range(steps):
    rng, key = jax.random.split(rng)
    action, _ = policy(state.obs, key)
    state = step(state, action)
    state.info["command"] = command
    traj.append(state)
    v = ev.get_local_linvel(state.data)
    vel_err.append(float(jp.linalg.norm(v[:2] - command[:2])))
    power.append(float(jp.sum(jp.abs(state.data.actuator_force * state.data.qvel[6:]))))
    fell = fell or bool(state.done)
    if t in (0, steps // 2, steps - 1):
      o = state.obs["state"] if isinstance(state.obs, dict) else state.obs
      obs_cmd.append([round(float(x), 3) for x in o[-3:]])
  dist = float(np.linalg.norm(np.array(state.data.qpos[:2]) - start_xy))
  return traj, {"mean_velocity_error": float(np.mean(vel_err)),
                "mean_power_W": float(np.mean(power)),
                "distance_m": dist,
                "avg_speed_from_position_mps": dist / (steps * ev.dt),
                "fell": fell,
                "obs_command_seen_by_policy": obs_cmd}


def evaluate(name, cfg, make_inf, params, cmd=(1.0, 0.0, 0.0), steps=500):
  # Same command and same metrics for every run, so results are comparable.
  # (The training reward itself is NOT comparable, because each run uses different weights.)
  policy = jax.jit(make_inf(params, deterministic=True))
  # (a) plain env with this run's config
  traj, m_plain = _rollout(registry.load(env_name, config=cfg), policy, cmd, steps)
  # (b) same settings as the official notebook's rollout (perturbations on but kicks off, wider command range)
  cfg2 = make_cfg(EXPERIMENTS.get(name, {}))
  cfg2.pert_config.enable = True
  cfg2.pert_config.velocity_kick = [0.0, 0.0]
  cfg2.pert_config.kick_wait_times = [5.0, 15.0]
  cfg2.command_config.a = [1.5, 0.8, 2 * jp.pi]
  ev2 = registry.load(env_name, config=cfg2)
  traj2, m_nb = _rollout(ev2, policy, cmd, steps)
  print(f"[{name}] plain eval: {m_plain}")
  print(f"[{name}] notebook-style eval: {m_nb}")
  try:   # rendering needs a working EGL setup; keep the numbers even if video fails
    frames = ev2.render(traj2[::2], camera="track", width=640, height=480)
    media.write_video(f"videos/{name}.mp4", frames, fps=1.0 / ev2.dt / 2)
  except Exception as e:
    print(f"[{name}] video not saved: {e}")
  # keep the trained weights so evaluation can be redone without retraining
  import pickle
  os.makedirs("params", exist_ok=True)
  with open(f"params/{name}.pkl", "wb") as f:
    pickle.dump(jax.tree.map(np.asarray, params), f)
  out = dict(m_nb)
  out["plain_eval"] = m_plain
  return out


def run_all(experiments, num_timesteps):
  global NUM_TIMESTEPS
  NUM_TIMESTEPS = num_timesteps
  all_results = {}
  for name, overrides in experiments.items():
    cfg = make_cfg(overrides)
    make_inf, params, curve = train_one(name, cfg)
    all_results[name] = {"overrides": overrides, "num_timesteps": num_timesteps,
                         "curve": curve, **evaluate(name, cfg, make_inf, params)}
    print(name, {k: v for k, v in all_results[name].items() if k != "curve"})
    with open("results/results.json", "w") as f:   # saved after every run
      json.dump(all_results, f, indent=2)
  print(f"\n{'experiment':18s} {'vel_error':>10s} {'power_W':>10s} {'fell':>6s}")
  for name, r in all_results.items():
    print(f"{name:18s} {r['mean_velocity_error']:10.3f} {r['mean_power_W']:10.1f} {str(r['fell']):>6s}  speed_from_position={r['avg_speed_from_position_mps']:.2f} m/s")
  return all_results


# Quick test first (one short run) to check everything works:
#   run_all({"baseline": {}}, 5_000_000)
# Then the real study:
#   results = run_all(EXPERIMENTS, NUM_TIMESTEPS)
