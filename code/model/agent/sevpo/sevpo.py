import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
from flax.core import freeze
import numpy as np
from model.agent.agent import Agent
from model.data.dataset import DatasetDict
from model.networks import MLP, Ensemble, StateActionValue, StateValue, Relu_StateActionValue, Relu_StateValue, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule
from model.networks.diffusion import dpm_solver_sampler_1st, vp_sde_schedule


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class SEVPO(Agent):
    score_model: TrainState
    target_score_model: TrainState
    beta_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    acsafe_critic: TrainState
    acsafe_target_critic: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int
    M: int = struct.field(pytree_node=False)
    iteration_count: int
    qcpi: int
    thres: int
    safeloss: int
    unsafeloss: int
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        iteration_count:int = 0,
        qcpi: int=10,
        thres: int=10,
        safeloss: int=1,
        unsafeloss: int=3,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        sampling_method: str = 'ddpm',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(1e6),
        extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key, acsafe_critic_key = jax.random.split(rng, 7)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=mish)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']
        actor_params = freeze(actor_params)

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        
        beta_score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                        weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                        mask=get_weight_decay_mask,))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        acsafe_critic_params = critic_def.init(acsafe_critic_key, observations, actions)["params"]
        acsafe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=acsafe_critic_params, tx=critic_optimiser
        )
        acsafe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=acsafe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )


        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        safe_value_params = value_def.init(safe_value_key, observations)["params"]

        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            beta_score_model = beta_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            acsafe_critic=acsafe_critic,
            acsafe_target_critic=acsafe_target_critic,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            iteration_count = iteration_count,
            qcpi= qcpi,
            thres= thres,
            safeloss=safeloss,
            unsafeloss=unsafeloss,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            sampling_method=sampling_method,
            critic_objective=critic_objective,
            critic_type=critic_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub
        )

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:

        rng = agent.rng
        score_params = agent.target_score_model.params
        
        if agent.sampling_method == 'ddpm':
            actions, _ = ddpm_sampler(agent.score_model.apply_fn, score_params, agent.T, rng, agent.act_dim, 
                                    batch["next_observations"], agent.alphas, agent.alpha_hats, agent.betas, 
                                    agent.ddpm_temperature, agent.M, agent.clip_sampler)
        elif agent.sampling_method == 'dpm_solver-1':
            actions, _ = dpm_solver_sampler_1st(agent.score_model.apply_fn, score_params, agent.T, rng, agent.act_dim, 
                                                batch["next_observations"], agent.alphas, agent.alpha_hats, agent.betas, 
                                                agent.ddpm_temperature, agent.M, agent.clip_sampler)
        next_q = agent.target_critic.apply_fn(
                {"params": agent.target_critic.params}, batch["next_observations"], actions
            )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_q

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info
    
    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])

            safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()

            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), 
                                     "vc_max": vc.max(), "vc_min": vc.min(),
                                     "vc_max_mean": (vc.max()).mean(),
                                     "vc_min_mean": (vc.min()).mean()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        agent = agent.replace(safe_value=safe_value)

        return agent, info
    
    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        alpha = 0.99
        actual_vio = batch['costs']
        next_vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["next_observations"]
            )
        qc_nonterminal = (1-0.99)*actual_vio + 0.99*(actual_vio+alpha * next_vc)

        target_qc = qc_nonterminal * batch["masks"] + actual_vio * (1 - batch["masks"])

        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    def update_actorqc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        alpha = 0.99
        actual_vio = batch['costs']
        rng = agent.rng
        score_params = agent.target_score_model.params
        
        if agent.sampling_method == 'ddpm':
            actions, _ = ddpm_sampler(agent.score_model.apply_fn, score_params, agent.T, rng, agent.act_dim, 
                                      batch["next_observations"], agent.alphas, agent.alpha_hats, agent.betas, 
                                      agent.ddpm_temperature, agent.M, agent.clip_sampler)
        elif agent.sampling_method == 'dpm_solver-1':
            actions, _ = dpm_solver_sampler_1st(agent.score_model.apply_fn, score_params, agent.T, rng, agent.act_dim, 
                                                batch["next_observations"], agent.alphas, agent.alpha_hats, agent.betas, 
                                                agent.ddpm_temperature, agent.M, agent.clip_sampler)
        next_qc = agent.acsafe_target_critic.apply_fn(
                {"params": agent.acsafe_target_critic.params}, batch["next_observations"], actions
            )
        target_qc = actual_vio + alpha*batch["masks"] * next_qc

        def acsafe_critic_loss_fn(acsafe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.acsafe_critic.apply_fn(
                {"params": acsafe_critic_params}, batch["observations"], batch["actions"]
            )
            
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "acsafe_critic_loss": safe_critic_loss,
                "acqc": qcs.mean(),
                "acqc_max": qcs.max(),
                "acqc_min": qcs.min(),
            }

        grads, info = jax.grad(acsafe_critic_loss_fn, has_aux=True)(agent.acsafe_critic.params)
        acsafe_critic = agent.acsafe_critic.apply_gradients(grads=grads)

        agent = agent.replace(acsafe_critic=acsafe_critic)

        acsafe_target_critic_params = optax.incremental_update(
            acsafe_critic.params, agent.acsafe_target_critic.params, agent.tau
        )
        acsafe_target_critic = agent.acsafe_target_critic.replace(params=acsafe_target_critic_params)

        new_agent = agent.replace(acsafe_critic=acsafe_critic, acsafe_target_critic=acsafe_target_critic)
        return new_agent, info

    def update_actor_beta(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        new_iteration_count = agent.iteration_count + 1

        if agent.sampling_method == 'dpm_solver-1':
            eps = 1e-3
            time = jax.random.uniform(key, (batch['actions'].shape[0], )) * (1. - eps) + eps
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            alpha_t, sigma_t = vp_sde_schedule(time)
            time = jnp.expand_dims(time, axis=1)
            noisy_actions = alpha_t[:, None] * batch['actions'] + sigma_t[:, None] * noise_sample
        elif agent.sampling_method == 'ddpm':
            time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            
            alpha_hats = agent.alpha_hats[time]
            time = jnp.expand_dims(time, axis=1)
            alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
            alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
            noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        actions, _ = ddpm_sampler(agent.score_model.apply_fn, agent.score_model.params, agent.T, rng, agent.act_dim, 
                                      batch["observations"], agent.alphas, agent.alpha_hats, agent.betas, 
                                      agent.ddpm_temperature, agent.M, agent.clip_sampler)

        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )

        qc_ori = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        ) 
        qc_ori = qc_ori.max(axis=0)

        qc_pi = agent.acsafe_critic.apply_fn(
            {"params": agent.acsafe_critic.params},
            batch["observations"],
            actions,
        )
        qc_pi = qc_pi.max(axis=0)

        vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"]
            )

        threshold = agent.thres
        unsafe_condition = jnp.where((vc-threshold) > 0. - eps, 1, 0)
        safe_condition = jnp.where((vc-threshold) <= 0. - eps, 1, 0) * jnp.where((qc_ori-threshold)<=0. - eps, 1, 0)
        
        safe_condition_infeasible = safe_condition*jnp.where((qc_pi) > agent.qcpi, 1, 0)
        safe_condition_feasible = safe_condition*jnp.where((qc_pi) <=agent.qcpi, 1, 0)

        feasible_q_value = safe_condition_feasible*q 
        infeasible_q_value = (safe_condition_infeasible*q)/jnp.abs(qc_pi)
        q = feasible_q_value + infeasible_q_value
            
        cost_exp_adv = jnp.exp((vc-qc_pi) * agent.cost_temperature)
        reward_exp_adv = jnp.exp((q)*agent.reward_temperature)


        unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub)
        safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        
        def actor_loss_fn(
                score_model_params, beta_score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            def true_fun(_):
                eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                        batch['observations'],
                                        noisy_actions,
                                        time,
                                        rngs={'dropout': key},
                                        training=True)

                safe_loss_ori = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * safe_weights).mean()
                actor_loss = agent.safeloss*safe_loss_ori+agent.unsafeloss*unsafe_weights.mean()
                return actor_loss
            
            def false_fun(_):
                eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                        batch['observations'],
                                        noisy_actions,
                                        time,
                                        rngs={'dropout': key},
                                        training=True)

                new_beta_policy = agent.beta_score_model.apply_fn({'params': beta_score_model_params},
                                        batch['observations'],
                                        noisy_actions,
                                        time,
                                        rngs={'dropout': key},
                                        training=True)
                loss_new_beta = agent.unsafeloss*(((eps_pred - new_beta_policy) ** 2).sum(axis = -1) * unsafe_weights).mean()\
                              + (((eps_pred - noise_sample) ** 2).sum(axis = -1) * safe_weights).mean()
                return loss_new_beta

            actor_loss = jax.lax.cond(new_iteration_count <= 1000000, true_fun, false_fun, None)
            
            return actor_loss, {'actor_loss': actor_loss, 
                                'new_iteration_count':new_iteration_count,
                                'unsafe_weights': unsafe_weights.mean(),
                                'safe count': np.sum(safe_condition==1),
                                'unsafe count': np.sum(unsafe_condition==1),
                                'total count': len(safe_condition)}
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params, agent.beta_score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        def true_fun(_):
            return agent.beta_score_model.replace(params=agent.score_model.params)

        def false_fun(_):
            def true_fun_update(_):
                updated_params = optax.incremental_update(agent.score_model.params, agent.beta_score_model.params, 1)
                return agent.beta_score_model.replace(params=updated_params)

            def false_fun_update(_):
                return agent.beta_score_model

            return jax.lax.cond((new_iteration_count % 5 == 0), true_fun_update, false_fun_update, None)

        beta_score_model = jax.lax.cond(new_iteration_count <= 1000000, true_fun, false_fun, None)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)

        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, beta_score_model=beta_score_model,
                                   rng=rng, iteration_count=new_iteration_count)

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        
        if self.sampling_method == 'ddpm':
            actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        elif self.sampling_method == 'dpm_solver-1':
            actions, rng = dpm_solver_sampler_1st(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            raise ValueError(f'Invalid sampling method: {self.sampling_method}')
        
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        qcs = compute_safe_q(self.safe_critic.apply_fn, self.safe_critic.params, observations, actions)

        qcs_5th_percentile = jnp.percentile(qcs, 5)
        mask= qcs<=qcs_5th_percentile
        any_valid = jnp.any(mask)
        valid_qs_values = jnp.where(mask, qs, -jnp.inf)

        idx = jnp.where(any_valid, jnp.argmax(valid_qs_values), jnp.argmin(qcs))

        action = actions[idx]
        new_rng = rng
        action = np.where(np.isnan(action), -1, action.squeeze())

        return np.array(action.squeeze()), self.replace(rng=new_rng)
       

    @jax.jit
    def update_safe_region(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch['observations'].shape[0]/2)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, safe_critic_info = new_agent.update_vc(mini_batch)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch)

        return new_agent, {**safe_critic_info, **safe_value_info}

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch['observations'].shape[0]/2)
    
        def first_half(x):
            return x[:batch_size]
        
        def second_half(x):
            return x[batch_size:]
        
        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor_beta(first_batch)
        new_agent, actor_info = new_agent.update_actor_beta(second_batch)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, value_info = new_agent.update_q(mini_batch)
        new_agent, acsafe_critic_info = new_agent.update_actorqc(mini_batch)

        return new_agent, {**actor_info, **value_info, **acsafe_critic_info}
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
