from brax.envs.fd.fd_env import FDEnv

import jax
import jax.numpy as jnp

from etils import epath

from brax.io import mjcf
from brax.envs.base import State

from brax import math, base


class Pusher(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/pusher.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl"}, **kwargs)

        # The tips_arm body gets fused with r_wrist_roll_link, so we use the parent
        # r_wrist_flex_link for tips_arm_idx.
        # self._tips_arm_idx = self.sys.link_names.index('r_wrist_flex_link')
        # self._object_idx = self.sys.link_names.index('object')
        # self._goal_idx = self.sys.link_names.index('goal')

    def step(self, state: State, action: jnp.ndarray, analytic: bool = False):
        # Scale action from [-1,1] to actuator limits
        # action_min = self.sys.actuator.ctrl_range[:, 0]
        # action_max = self.sys.actuator.ctrl_range[:, 1]
        # action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        dx_next = self.step_fn(state.pipeline_state, action)

        # x_i = state.pipeline_state.x.vmap().do(
        #     base.Transform.create(pos=self.sys.link.inertia.transform.pos)
        # )
        # vec_1 = x_i.pos[self._object_idx] - x_i.pos[self._tips_arm_idx]
        # vec_2 = x_i.pos[self._object_idx] - x_i.pos[self._goal_idx]

        # reward_near = -math.safe_norm(vec_1)
        # reward_dist = -math.safe_norm(vec_2)
        # reward_ctrl = -jnp.square(action).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        # obs = self._get_obs(dx_next)
        # state.metrics.update(
        #     reward_near=reward_near,
        #     reward_dist=reward_dist,
        #     reward_ctrl=reward_ctrl,
        # )
        return state.replace(pipeline_state=dx_next)

    def reset(self, rng: jax.Array):
        qpos = self.sys.init_q

        rng, rng1, rng2 = jax.random.split(rng, 3)

        # # randomly orient the object
        # cylinder_pos = jnp.concatenate([
        #     jax.random.uniform(rng, (1,), minval=-0.3, maxval=-1e-6),
        #     jax.random.uniform(rng1, (1,), minval=-0.2, maxval=0.2),
        # ])
        # # constrain minimum distance of object to goal
        # goal_pos = jnp.array([0.0, 0.0])
        # norm = math.safe_norm(cylinder_pos - goal_pos)
        # scale = jnp.where(norm < 0.17, 0.17 / norm, 1.0)
        # cylinder_pos *= scale
        # qpos = qpos.at[-4:].set(jnp.concatenate([cylinder_pos, goal_pos]))

        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-4:].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward_dist': zero, 'reward_ctrl': zero, 'reward_near': zero}
        return State(pipeline_state, obs, reward, done, metrics)

    def _get_obs(self, state: State) -> jax.Array:
        """Observes pusher body position and velocities."""
        # x_i = state.x.vmap().do(
        #     base.Transform.create(pos=self.sys.link.inertia.transform.pos)
        # )
        return jnp.concatenate([
            state.qpos,
            state.qvel,
        ])
