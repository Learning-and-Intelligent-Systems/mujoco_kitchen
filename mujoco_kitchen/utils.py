import mujoco_kitchen.adept_envs
import gym
import numpy as np
from gym.spaces.box import Box
from mujoco_kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
)

from mujoco_kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS

ALL_KITCHEN_ENVIRONMENTS = {
    "microwave": KitchenMicrowaveV0,
    "kettle": KitchenKettleV0,
    "slide_cabinet": KitchenSlideCabinetV0,
    "hinge_cabinet": KitchenHingeCabinetV0,
    "top_left_burner": KitchenTopLeftBurnerV0,
    "light_switch": KitchenLightSwitchV0,
    "microwave_kettle_light_top_left_burner": KitchenMicrowaveKettleLightTopLeftBurnerV0,
    "hinge_slide_bottom_left_burner_light": KitchenHingeSlideBottomLeftBurnerLightV0,
}

primitive_idx_to_name = {0: 'angled_x_y_grasp', 1: 'move_delta_ee_pose', 2: 'rotate_about_y_axis', 3: 'lift', 4: 'drop', 5: 'move_left', 6: 'move_right', 7: 'move_forward', 8: 'move_backward', 9: 'open_gripper', 10: 'close_gripper', 11: 'rotate_about_x_axis'}
primitive_name_to_action_idx = {'angled_x_y_grasp': [0, 1, 2, 3], 'move_delta_ee_pose': [4, 5, 6], 'rotate_about_y_axis': 7, 'lift': 8, 'drop': 9, 'move_left': 10, 'move_right': 11, 'move_forward': 12, 'move_backward': 13, 'rotate_about_x_axis': 14, 'open_gripper': 15, 'close_gripper': 16}

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
        self._elapsed_steps = 0
        self._max_episode_steps = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self._step += 1
        self._elapsed_steps += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        self._elapsed_steps = 0
        return self.env.reset()


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(
                action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env, unused=None):
        gym.Wrapper.__init__(self, env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        o, r, d, i = self.env.step(
            original,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return o, r, d, i

    def reset(self):
        return self.env.reset()


class ImageUnFlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight),
            reward,
            done,
            info,
        )

def make_base_kitchen_env(env_class, env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS[env_class](**env_kwargs)
    return env

def make_env(env_suite, env_name, env_kwargs):
    usage_kwargs = env_kwargs["usage_kwargs"]
    max_path_length = usage_kwargs["max_path_length"]
    use_raw_action_wrappers = usage_kwargs.get("use_raw_action_wrappers", False)
    unflatten_images = usage_kwargs.get("unflatten_images", False)

    env_kwargs_new = env_kwargs.copy()
    if "usage_kwargs" in env_kwargs_new:
        del env_kwargs_new["usage_kwargs"]
    if "image_kwargs" in env_kwargs_new:
        del env_kwargs_new["image_kwargs"]

    if env_suite == "kitchen":
        env = make_base_kitchen_env(env_name, env_kwargs_new)
    if unflatten_images:
        env = ImageUnFlattenWrapper(env)

    if use_raw_action_wrappers:
        env = ActionRepeat(env, 2)
        env = NormalizeActions(env)
        env = TimeLimit(env, max_path_length // 2)
    else:
        env = TimeLimit(env, max_path_length)
    env.reset()
    return env

def primitive_and_params_to_primitive_action(primitive_name, params):
    primitive_name = primitive_name.lower()
    num_primitives = 12
    # assert env.primitive_idx_to_name == primitive_idx_to_name
    # assert env.primitive_name_to_action_idx == primitive_name_to_action_idx
    # assert env.num_primitives == num_primitives
    action = np.zeros(29) + 0.0
    for i, name in primitive_idx_to_name.items():
        if name == primitive_name:
            action[i] = 1.0
            idxs = primitive_name_to_action_idx[name]
            if isinstance(idxs, int):
                idxs = [idxs]
            for param_i, idx in enumerate(idxs):
                action[num_primitives + idx] = params[param_i]
            break
    else:
        assert False
    
    return action.astype(np.float32)
