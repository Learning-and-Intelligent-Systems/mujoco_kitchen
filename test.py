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
    action = np.zeros(29) + 0.0
    for i, name in env.primitive_idx_to_name.items():
        if name == primitive_name:
            action[i] = 1.0
            idxs = env.primitive_name_to_action_idx[name]
            for param_i, idx in enumerate(idxs):
                action[env.num_primitives + idx] = params[param_i]
            break
    else:
        assert False
    
    return action

##

env = make_env("kitchen", "microwave", {"usage_kwargs": {"max_path_length": 50, "use_raw_action_wrappers": False, "unflatten_images": False}})

# Display Useful Information
print("#"*30)
print("Env", env)
print("Primitive Funcs", env.primitive_name_to_func.keys())
print("Primitive Index -> Names", env.primitive_idx_to_name)
print("Primitive Names -> Action Index", env.primitive_name_to_action_idx)
print("Number of Parameters", env.max_arg_len)
print("Number of Primitives", env.num_primitives)
print("Action Space", env.action_space)
print("#"*30)

env.reset()
for _ in range(100):
    env.render()
    # Move to Kettle
    action = primitive_and_params_to_primitive_action('move_delta_ee_pose', env.get_site_xpos("kettle_site") - env.get_site_xpos("end_effector")) #env.action_space.sample()

    # Parse Action to Primitive and Parameters
    # Only needed for printing (This is done in env)
    primitive_idx, primitive_args = (
        np.argmax(action[: env.num_primitives]),
        action[env.num_primitives :],
    )
    primitive_name = env.primitive_idx_to_name[primitive_idx]
    parameters = None
    for key, val in env.primitive_name_to_action_idx.items():
        if key == primitive_name:
            if type(val) ==  int:
                parameters = [primitive_args[val]]
            else:
                parameters = [primitive_args[i] for i in val]
    assert parameters is not None
    print(primitive_name, parameters, "\n")

    state, reward, done, info = env.step(action)

    # Parse State into Object Centric State
    for key, val in OBS_ELEMENT_INDICES.items():
        print(key, [state[i] for i in val])
    print()
    important_sites = ["hinge_site1", "hinge_site2", "kettle_site", "microhandle_site", "knob1_site", "knob2_site", "knob3_site", "knob4_site", "light_site", "slide_site", "end_effector"]
    for site in important_sites:
        print(site, env.get_site_xpos(site))
        # Potentially can get this ^ from state

    if done:
        print("Done")
        break
