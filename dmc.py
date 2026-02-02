from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    """
    环境动作数据类型转换包装器。
    
    主要功能:
        用于修改环境接收动作时的预期数据类型（dtype）。它在不改变物理环境逻辑的前提下，
        负责在智能体（Agent）和环境（Environment）之间进行数据类型的强制转换。
    
    设计模式: 
        装饰器模式。它持有原始环境的引用，并模仿原始环境的行为。
    """
    def __init__(self, env, dtype):
        """
        初始化包装器。

        参数说明:
            env (dm_env.Environment): 被包装的原始环境实例。
            dtype (numpy.dtype): 目标数据类型（例如 np.float32）。
        """
        self._env = env
        # 获取原始环境的动作规格（包含动作的维度、最大值、最小值等信息）
        wrapped_action_spec = env.action_spec()
        
        # 内部逻辑: 重新定义一个新的动作规格，除了数据类型改为指定的 dtype，其余参数保持不变
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        """
        执行环境步进。

        输入参数:
            action (numpy.ndarray): 智能体输出的动作。
            
        内部逻辑:
            INFO: 在将动作传递给原始环境之前，使用 .astype() 将其强制转换为原始环境所要求的类型。
            这是解决数据类型不匹配报错的关键步骤。
        
        输出:
            TimeStep: 包含观测、奖励、折扣系数的元组。
        """
        # 强制类型转换，例如从 float32 转回原始的 float64
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        """
        返回环境的观测规格。直接转发给原始环境，不作修改。
        """
        return self._env.observation_spec()

    def action_spec(self):
        """
        返回修改后的动作规格。
        
        输出:
            specs.BoundedArray: 告知智能体，现在环境接收的是新的数据类型。
        """
        return self._action_spec

    def reset(self):
        """
        重置环境。直接调用原始环境的重置方法。
        """
        return self._env.reset()

    def __getattr__(self, name):
        """
        属性转发魔法方法。
        
        内部逻辑:
            如果外部访问的属性或方法在这个类里没定义（比如 env.physics），
            Python 会通过这个方法去原始环境 (self._env) 里查找。
            这确保了包装后的环境依然可以使用原始环境的所有功能。
        """
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
