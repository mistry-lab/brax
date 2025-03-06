from brax.envs.fd.fd_env import FDEnv
from brax.envs.base import State

from etils import epath
from brax.io import mjcf
from brax import math

import jax
import numpy as np
from jax import numpy as jnp
import mujoco
from mujoco import mjx

class ShadowHand(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/scene_right.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl", "sensordata"}, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        qpos_init = self._init_gen(rng)
        qvel_init = jnp.zeros_like(qpos_init)  # or any distribution you like

        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_palm_position': zero,
            'reward_goal_orientation': zero,
            'reward_object_linear_velocity': zero,
            'reward_object_angular_velocity': zero,
            'reward_hand_joint_position': zero,
            'reward_hand_joint_velocity': zero,
        }

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray, analytic: bool = False):
        dx = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx)

        pos = self._parse_sensordata("object_position", self.sys, dx)
        pos_ref = self._parse_sensordata("palm_position", self.sys, dx)
        position_reward = -0.1*jnp.sum((pos - pos_ref)**2)

        quat = self._parse_sensordata("object_orientation", self.sys, dx)
        quat_ref = self._parse_sensordata("goal_orientation", self.sys, dx)
        # quat_ref = jnp.array([1.,0.,0.,0.])
        orientation_reward = 0.5*jnp.sum(math.relative_quat(quat, quat_ref)**2)

        vel = self._parse_sensordata("object_linear_velocity", self.sys, dx)
        object_velocity_reward = 0.05*jnp.sum(vel**2)

        ang_vel = self._parse_sensordata("object_angular_velocity", self.sys, dx)
        object_angular_velocity_reward = 0.05*jnp.sum(ang_vel**2)

        joint_position_reward = 0.2*jnp.sum(dx.qpos[:24]**2) 
        joint_velocity_reward = 0.2*jnp.sum(dx.qvel[:24]**2)

        reward = \
            position_reward + orientation_reward \
                + object_velocity_reward + object_angular_velocity_reward \
                    + joint_position_reward + joint_velocity_reward

        state.metrics.update(
            reward_palm_position = position_reward,
            reward_goal_orientation = orientation_reward,
            reward_object_linear_velocity = object_velocity_reward,
            reward_object_angular_velocity = object_angular_velocity_reward,
            reward_hand_joint_position = joint_position_reward,
            reward_hand_joint_velocity = joint_velocity_reward
        )

        return state.replace(pipeline_state=dx, obs=obs, reward=reward)

    def _parse_sensordata(name, mx, dx):
        id = mjx.name2id(mx, mujoco.mjtObj.mjOBJ_SENSOR, name)
        i = mx.sensor_adr[id]
        dim = mx.sensor_dim[id]
        return dx.sensordata[i:i+dim]

    def control_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
        """ Control cost. Due to position control, penalisation 
        of actuator_force instead.
        """
        x = dx.actuator_force
        return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.full((24,), 10.)), x))


    def _init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
        """ Initialise Shadow hand scene. (Right hand, sphere object and sphere goal)
        """
        # 1. Generate joint_pos and joint_vel
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)  # Splitting key for different random generations
        joint_pos = jax.random.uniform(subkey1, (total_batch, 24), minval=-0.01, maxval=0.01)
        joint_vel = jax.random.uniform(subkey2, (total_batch, 24), minval=-0.05, maxval=0.05)
        object_quat = math.random_quaternion(subkey3, total_batch)
        # object_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))
        goal_quat = math.random_quaternion(subkey4, total_batch)
        # goal_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))

        # 2. Fixed values for object_pos, object_vel, object_ang_vel, goal_ang_vel
        object_pos = jnp.array([0.3, 0.0, 0.065])  # Shape (3,)
        object_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)
        object_ang_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)
        goal_ang_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)

        object_pos_broadcast = jnp.tile(object_pos, (total_batch, 1))  # Shape (total_batch, 3)
        object_vel_broadcast = jnp.tile(object_vel, (total_batch, 1))  # Shape (total_batch, 3)
        object_ang_vel_broadcast = jnp.tile(object_ang_vel, (total_batch, 1))  # Shape (total_batch, 3)
        goal_ang_vel_broadcast = jnp.tile(goal_ang_vel, (total_batch, 1))  # Shape (total_batch, 3)

        # 5. Concatenate all components along the second axis (axis=1)
        xinits = jnp.concatenate([
            joint_pos,  # Shape (total_batch, 20)
            object_pos_broadcast,  # Shape (total_batch, 3)
            object_quat,  # Shape (total_batch, 4)
            goal_quat,  # Shape (total_batch, 4)
            joint_vel,  # Shape (total_batch, 20)
            object_vel_broadcast,  # Shape (total_batch, 3)
            object_ang_vel_broadcast,  # Shape (total_batch, 3)
            goal_ang_vel_broadcast  # Shape (total_batch, 3)
        ], axis=1)

        return xinits


    def _get_observation(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate([dx.qpos, dx.qvel], axis=0)

    def _is_terminal(self, mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
        pos = self._parse_sensordata("object_position", mx, dx)
        return  jnp.array([jnp.logical_or(pos[2] < -0.05, (dx.time / mx.opt.timestep) > (_cfg.ntotal-1))])
