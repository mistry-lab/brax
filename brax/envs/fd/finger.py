from brax.envs.fd.fd_env import FDEnv

from etils import epath

import jax
import jax.numpy as jnp

from brax.io import mjcf
from brax.envs.base import State

class Finger(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/finger_mjx.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl", "sensordata"}, **kwargs)

        def logistic(number):
            return 1 / (1 + jnp.exp(20 * (number + 1.)))
        
        def inverse_logistic(number):
            return 1 / (1 + jnp.exp(- 20 * (number - 1.)))

        def _surrogate_reward(
            qpos: jax.Array,
            qvel: jax.Array,
            p_finger: jax.Array,
            p_target: jax.Array,
            touch: jax.Array,
            action: jax.Array
        ):
            pos_finger = - (qpos[2] + jnp.pi / 2) ** 2

            target_dist = jnp.sum((p_finger - p_target)**2)
            # touch_reward = 0.001 * touch * pos_finger **2
            speed_reward = - qvel[2]

            # ctrl_reward = 0.5 - jnp.sum(action ** 2)
            ctrl_reward = - logistic(action[0]) - logistic(action[1])
            ctrl_reward += - inverse_logistic(action[0]) - inverse_logistic(action[1])

            movement_reward = jnp.where(
                touch == 1,
                jnp.sum(action ** 2),
                0.0
            )

            return 10 * movement_reward + ctrl_reward + pos_finger

        surrogate_derivative = jax.grad(_surrogate_reward, argnums=(0, 1, 2, 3, 4, 5))

        @jax.custom_vjp
        def _get_reward(
            qpos: jax.Array,
            qvel: jax.Array,
            p_finger: jax.Array,
            p_target: jax.Array,
            touch: jax.Array,
            action: jax.Array
        ):
            # pos_finger = qpos[2]

            # target_reward = - 1.0 * jnp.sum((p_finger - p_target)**2)
            # touch_reward = - 0.001 * touch * pos_finger **2
            ctrl_reward = - 0.00001 * jnp.sum(action ** 2)

            reward = ctrl_reward

            return reward, {}

        def _get_reward_forward(
            qpos: jax.Array,
            qvel: jax.Array,
            p_finger: jax.Array,
            p_target: jax.Array,
            touch: jax.Array,
            action: jax.Array
        ):
            reward = _get_reward(qpos, qvel, p_finger, p_target, touch, action)
            return reward, (qpos, qvel, p_finger, p_target, touch, action, reward)

        def _get_reward_backward(res, g):
            qpos_in, qvel_in, pfinger_in, ptarget_in, touch_in, u_in, reward = res
            
            d_qpos, d_qvel, d_pfinger, d_ptarget, d_touch, d_u = surrogate_derivative(
                qpos_in,
                qvel_in,
                pfinger_in,
                ptarget_in,
                touch_in,
                u_in,
            )

            # clamp_value = 0.1

            # d_qpos = jnp.clip(d_qpos, -clamp_value, clamp_value)
            # d_qvel = jnp.clip(d_qvel, -clamp_value, clamp_value)
            # d_pfinger = jnp.clip(d_pfinger, -clamp_value, clamp_value)
            # d_ptarget = jnp.clip(d_ptarget, -clamp_value, clamp_value)
            # d_touch = jnp.clip(d_touch, -clamp_value, clamp_value)
            # d_u = jnp.clip(d_u, -clamp_value, clamp_value)

            g_reward, g_metrics = g

            return (
                d_qpos * g_reward,
                d_qvel * g_reward,
                d_pfinger * g_reward,
                d_ptarget * g_reward,
                d_touch * g_reward,
                d_u * g_reward
            )

        _get_reward.defvjp(_get_reward_forward, _get_reward_backward)
        self.reward_fn = _get_reward

    def _angle_axis_to_quaternion(self, angle_axis):
        """
        Converts an angle-axis vector to a quaternion.

        Args:
            angle_axis: A JAX array of shape (3,) representing the angle-axis vector.

        Returns:
            A JAX array of shape (4,) representing the quaternion [w, x, y, z].
        """
        theta = jnp.linalg.norm(angle_axis)

        def not_zero(theta, angle_axis):
            half_theta = 0.5 * theta
            k = jnp.sin(half_theta) / theta
            w = jnp.cos(half_theta)
            xyz = angle_axis * k
            quaternion = jnp.concatenate([jnp.array([w]), xyz])
            return quaternion

        def is_zero(theta, angle_axis):
            k = 0.5
            w = 1.0
            xyz = angle_axis * k
            quaternion = jnp.concatenate([jnp.array([w]), xyz])
            return quaternion

        quaternion = jax.lax.cond(
            theta > 1e-8,  # A small threshold to handle numerical precision
            not_zero,
            is_zero,
            theta,
            angle_axis
        )
        return quaternion

    def reset(self, rng: jax.Array, flag: bool = False) -> State:
        qpos_init = self._generate_initial_conditions(rng)
        qvel_init = jnp.zeros_like(qpos_init)  # or any distribution you like

        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_target': zero,
            'reward_ctrl': zero,
        }

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray, flag: bool = False) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        touch = dx_next.sensordata[0]
        p_finger = dx_next.sensordata[1:4]
        p_target = dx_next.sensordata[4:7]

        reward, metrics = self.reward_fn(
            dx_next.qpos,
            dx_next.qvel,
            p_finger,
            p_target,
            touch,
            dx_next.ctrl
        )

        # state.metrics.update(
        #     reward_target=metrics["reward_target"],
        #     reward_ctrl=metrics["reward_ctrl"],
        # )

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )

    def _get_observation(self, dx: State):
        return jnp.concatenate([dx.qpos, dx.qvel, dx.sensordata])

    def _generate_initial_conditions(self, key: jax.Array) -> State:
        # Solution of IK
        _, key = jax.random.split(key)
        sign = 2.*jax.random.bernoulli(key, 0.5) - 1.

        # Reference target position in spinner local frame R_s
        _, key = jax.random.split(key, num=2)
        theta_l = jax.random.uniform(key, (1,), minval=0.6, maxval=2.5) # Polar Coord
        l_spinner = 0.22
        # r = jax.random.uniform(key, (1,), minval=l_spinner + 0.01, maxval=l_spinner + 0.01)
        r_l = l_spinner
        x_s,y_s = r_l*jnp.cos(theta_l), r_l*jnp.sin(theta_l) # Cartesian in R_l
        
        # Reference target position in finger frame R_f
        x, y = x_s, y_s - 0.39

        # Inverse kinematic formula
        l1, l2 = 0.17, 0.161
        q1 = sign * jnp.arccos( (x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2) )
        q0 = jnp.arctan2(y,x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2*jnp.cos(q1))

        # dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
        # dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))

        _, key = jax.random.split(key, num=2)
        theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)
        # dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))

        return jnp.concatenate([q0,q1,theta])
