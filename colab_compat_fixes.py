# Fix: this playground version defaults to the Warp backend, which is not installed on Colab.
# Make every environment use the JAX (MJX) backend instead.
import importlib
importlib.reload(registry)   # start from the original functions, so this cell is safe to re-run
_orig_get_cfg = registry.get_default_config
_orig_load = registry.load

def _get_cfg_jax(env_name, _orig=_orig_get_cfg):
  cfg = _orig(env_name)
  if "impl" in cfg:
    cfg.impl = "jax"
  return cfg

def _load_jax(env_name, config=None, config_overrides=None, _orig=_orig_load):
  if config is None:
    config = _get_cfg_jax(env_name)
  return _orig(env_name, config=config, config_overrides=config_overrides)

registry.get_default_config = _get_cfg_jax
registry.load = _load_jax
print("Using JAX backend:", registry.get_default_config("Go1JoystickFlatTerrain").impl)

# Fix 2: JAX 0.11 removed jax.device_put_replicated / device_put_sharded, which brax 0.14.2 still uses.
# Drop-in replacements from the JAX docs (docs.jax.dev/en/latest/migrate_pmap.html).
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def _device_put_sharded(shards, devices):
  mesh = Mesh(np.array(devices), ("x",))
  sharding = NamedSharding(mesh, P("x"))
  return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)

def _device_put_replicated(x, devices):
  mesh = Mesh(np.array(devices), ("x",))
  sharding = NamedSharding(mesh, P("x"))
  return jax.tree.map(lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x)

jax.device_put_sharded = _device_put_sharded
jax.device_put_replicated = _device_put_replicated
print("Patched jax.device_put_replicated / device_put_sharded")
