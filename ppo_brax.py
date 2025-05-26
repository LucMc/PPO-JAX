from ast import Tuple
from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import numpy as np
import optax
import distrax
from pprint import pprint
from functools import partial
import wandb
import tyro
import os
from pathlib import Path
import brax.envs

from ppo import PPO, Config, ActorNet, ValueNet


@dataclass(frozen=True)
class BraxConfig(Config):
    env_id: str = "ant" # BRAX env name
    # training_steps: int = 500_000*64  # total training time-steps
    # n_envs: int = 1*32  # number of parralel training envs
    # rollout_steps: int = 64 * 20 * 16  # env steps per rollout
    # batch_size: int = 64*3  # minibatch size

@dataclass(frozen=True)
class BraxPPO(PPO, BraxConfig):
    buffer_size: int = 2048

    @partial(jax.jit, static_argnames=["self", "env"])
    def get_rollout(
        self,
        actor_ts: TrainState,
        value_ts: TrainState,
        env,
        last_state,
        key: PRNGKeyArray,
    ):
        def step(carry, _):
            states, key = carry

            action_key, key = random.split(key)
            action_dist = actor_ts.apply_fn(actor_ts.params, jnp.array(states.obs))
            actions = action_dist.sample(seed=action_key)
            log_prob = action_dist.log_prob(actions)

            value = value_ts.apply_fn(value_ts.params, jnp.array(states.obs))
            next_states = env.step(states, actions)

            return (next_states, key), (
                next_states.reward,
                actions,
                next_states.done,  # should be episode starts?
                log_prob,
                states.obs,
                action_dist.stddev(),
                next_states.info,
            )

        (last_states, key), (rewards, actions, dones, log_probs, obss, stds, infos) = (
            jax.lax.scan(
                step, (last_state, key), jnp.arange(self.rollout_steps // self.n_envs)
            )
        )

        values = value_ts.apply_fn(value_ts.params, jnp.array(obss))
        last_values = value_ts.apply_fn(value_ts.params, jnp.array(last_states.obs))

        returns, advantages = jax.vmap(
            self.compute_returns_and_advantage, in_axes=(1, 1, 1, 0, 0)
        )(
            rewards,
            values.squeeze(axis=-1),
            dones,
            last_values,
            last_states.done,
        )

        # Fix shape bug
        returns = returns.T
        advantages = advantages.T

        rollout_info = {
            "mean rollout reward": np.mean(rewards),
            "advantage_mean": jnp.mean(advantages),
            "advantage_std": jnp.std(advantages),
            "explained_variance": 1 - (jnp.var(advantages.flatten()) / jnp.var(returns.flatten())+1e-8),  # fmt: skip
            "actor_lr": actor_ts.opt_state[-1].hyperparams["learning_rate"],
            "action_dist_std": stds.mean(),
            "value lr": value_ts.opt_state[-1].hyperparams["learning_rate"],
        }

        return (
            (
                obss,
                actions,
                values,
                log_probs,
                advantages,
                returns,
            ),
            rollout_info,
            infos,
        )


    def make_env(self, video_folder: str = None, env_args: dict = {}):
        return brax.envs.wrappers.training.wrap(brax.envs.get_environment(self.env_id))

    @staticmethod
    def learn(config: Config):
        ppo_agent = BraxPPO(buffer_size=config.rollout_steps, **config.__dict__)
        np.random.seed(ppo_agent.seed)  # Seeding for np operations

        if ppo_agent.log_video_every:
            base_video_dir = Path("videos")
            video_folder = base_video_dir / str(
                len(os.listdir(base_video_dir))
            )
            os.makedirs(video_folder)
            env_args = {"render_mode": "rgb_array"}
        else:
            video_folder = None
            env_args = {}

        if ppo_agent.log:
            wandb.init(
                project="jax-ppo",
                name="ppo",
                config=config.__dict__,  # Get from tyro
                tags=["PPO", ppo_agent.env_id],
                save_code=True,
            )

        ckpt_path = "./checkpoints"
        assert not ppo_agent.rollout_steps % ppo_agent.batch_size, "Must have rollout steps divisible into batches"

        key = random.PRNGKey(ppo_agent.seed)
        env_keys, actor_key, value_key, key = random.split(key, num=4)
        initial_reset_keys = random.split(env_keys, num=ppo_agent.n_envs)
        env = ppo_agent.make_env(video_folder=video_folder, env_args=env_args)

        states = env.reset(initial_reset_keys)
        current_global_step = 0

        actor_net = ActorNet(env.action_size)
        value_net = ValueNet()
        opt = optax.chain(
            optax.clip_by_global_norm(ppo_agent.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=optax.linear_schedule(
                    init_value=ppo_agent.learning_rate,
                    end_value=ppo_agent.learning_rate / 10,
                    transition_steps=ppo_agent.training_steps,
                ),
            ),
        )

        actor_ts = TrainState.create(
            apply_fn=jax.jit(actor_net.apply),
            params=actor_net.init(actor_key, states.obs),
            tx=opt,
        )
        value_ts = TrainState.create(
            apply_fn=jax.jit(value_net.apply),
            params=value_net.init(value_key, states.obs),
            tx=opt,
        )

        last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

        while current_global_step < ppo_agent.training_steps:
            print("\ncurrent_global_step:", current_global_step)
            rollout, rollout_info, env_infos = ppo_agent.get_rollout(
                actor_ts, value_ts, env, states, key
            )

            current_global_step += ppo_agent.rollout_steps * ppo_agent.n_envs

            actor_ts, value_ts, key, training_info = ppo_agent.outer_loop(
                key, actor_ts, value_ts, rollout
            )

            env_infos = {}  # Change this if there is anything in your env info you want to plot, i.e {"episode_length": env_infos["episode_length"]}

            full_logs = training_info | rollout_info | env_infos
            pprint(full_logs)

            if ppo_agent.log:
                wandb.log(full_logs, step=current_global_step)

                if current_global_step % 100_000 == 0:
                    wandb.save(ckpt_path)

        # Close stuff
        if ppo_agent.log:
            if abs(current_global_step % ppo_agent.log_video_every) < ppo_agent.rollout_steps:
                print("[ ] Uploading Videos ...", end="\r")
                for video_name in os.listdir(video_folder):
                    wandb.log({video_name: wandb.Video(str(base_video_dir / video_name))})
                print(r"[x] Uploading Videos ...")

            wandb.finish()
        envs.close()


if __name__ == "__main__":
    config = tyro.cli(BraxConfig)
    BraxPPO.learn(config)

