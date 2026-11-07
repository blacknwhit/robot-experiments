"""Official full-model update; production uses native JAX FSDP placement."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from openpi.models import pi0_config
from openpi.training import config as train_config,optimizer,weight_loaders

def prepare_donation(state,state_sharding):
    # Initial EMA equals params and may share storage. Donation requires each
    # writable buffer exactly once. Preserve values while separating storage.
    # JAX 0.5.3 device_put(..., may_alias=False) can retain individual buffers
    # for an already sharded array. Array.copy() was checked on all 8 shards.
    ema=jax.tree.map(lambda array:array.copy(),state.ema_params)
    return dataclasses.replace(state,ema_params=ema)

def make_config(cfg):
    return train_config.TrainConfig(name='streampi_robocasa365',exp_name='8gpu64_70k',
      model=pi0_config.Pi0Config(pi05=True,hist_horizon=3,action_horizon=50,discrete_state_input=True),
      weight_loader=weight_loaders.CheckpointWeightLoader(cfg['base_checkpoint']),
      optimizer=optimizer.AdamW(**cfg['optimizer']),lr_schedule=optimizer.CosineDecaySchedule(**cfg['learning_rate']),
      ema_decay=cfg['ema_decay'],seed=cfg['seed'],batch_size=cfg['global_batch_size'],fsdp_devices=cfg['fsdp_devices'],
      wandb_enabled=False,num_workers=0)

def _group(path):
    name = '/'.join(map(str,path))
    if name.startswith('PaliGemma/img/'):
        return 'vision'
    if name.startswith('PaliGemma/llm/') and '_1/' not in name and not name.endswith('_1'):
        return 'vlm'
    return 'action'

def group_norms(values):
    sums = {name:jnp.float32(0) for name in ('vision','vlm','action')}
    for path, value in values.flat_state().items():
        array = value.value.astype(jnp.float32)
        sums[_group(path)] += jnp.sum(jnp.square(array))
    return {name:jnp.sqrt(value) for name,value in sums.items()}

def train_step(conf, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions = batch
    train_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(model):
        return jnp.mean(model.compute_loss(train_rng, obs, actions, train=True))
    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0,conf.trainable_filter))(model)
    params = state.params.filter(conf.trainable_filter)
    updates, opt_state = state.tx.update(grads,state.opt_state,params)
    nnx.update(model,optax.apply_updates(params,updates))
    new_params = nnx.state(model)
    ema = jax.tree.map(lambda old,new:state.ema_decay*old+(1-state.ema_decay)*new,
                      state.ema_params,new_params)
    metrics = {'loss':loss,'grad_norm':optax.global_norm(grads),
               'gradient_groups':group_norms(grads),'update_groups':group_norms(updates),
               'actual_parameter_delta_groups':group_norms(jax.tree.map(lambda new,old:new-old,new_params,state.params)),
               'learning_rate':conf.lr_schedule.create()(state.step)}
    return dataclasses.replace(state,step=state.step+1,params=new_params,opt_state=opt_state,ema_params=ema),metrics
