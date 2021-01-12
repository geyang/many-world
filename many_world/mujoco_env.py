import os
import sys
from contextlib import contextmanager
from os import path

import gym
import numpy as np
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco_py,")

DEFAULT_SIZE = 640, 480


class MujocoEnv(gym.Env):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    def __init__(self, model_path, frame_skip=4, set_action_space=True, set_observation_space=True,
                 width=64, height=64):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.width = width
        self.height = height

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # to quickly reset the state of the simulator for consistency.
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        if set_action_space:
            bounds = self.model.actuator_ctrlrange.copy()
            if not bounds.all():  # use force bounds instead.
                bounds = self.model.actuator_forcerange.copy()
            self.action_space = spaces.Box(low=bounds[:, 0], high=bounds[:, 1])

        if set_observation_space:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *args, **kwargs):
        self.sim.reset()
        ob = self.reset_model(*args, **kwargs)
        old_viewer = self.viewer
        for v in self._viewers.values():
            self.viewer = v
            self.viewer_setup()
        self.viewer = old_viewer
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def set_color(self, id, rgba=None):
        self.sim.model.geom_rgba[id] = rgba or 0

    @contextmanager
    def with_color(self, id, rgba=None):
        old_rgba = self.sim.model.geom_rgba[id].copy()
        self.sim.model.geom_rgba[id] = rgba or 0
        yield
        self.sim.model.geom_rgba[id] = old_rgba

    def render(self, mode='human', width=None, height=None):
        """
        returns images of modality <modeL

        :param mode: One of ['human', 'rgb', 'rgbd', 'depth']
        :param kwargs: width, height (in pixels) of the image.
        :return: image(, depth). image is between [0, 1), depth is distance.
        """
        width = width or self.width
        height = height or self.height
        viewer = self._get_viewer(mode, cam_id=self.cam_id)

        viewer.render(width, height)

        if mode in ['rgb', 'rgb_array']:
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'rgbd':
            rgb, d = viewer.read_pixels(width, height, depth=True)
            # original image is upside-down, so flip it
            return rgb[::-1, :, :], d[::-1, :]
        elif mode == 'depth':
            _, d = viewer.read_pixels(width, height, depth=True)
            # original image is upside-down, so flip it
            return d[::-1, :]
        elif mode == 'grey':
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :].mean(axis=-1).astype(np.uint8)
        elif mode == 'notebook':
            from PIL import Image
            from IPython.display import display

            data = viewer.read_pixels(width, height, depth=False)
            img = Image.fromarray(data[::-1])
            display(img)
            return data[::-1]
        elif mode == 'human':
            if width and height:
                import glfw
                glfw.set_window_size(viewer.window, width, height)
            viewer.render()

    def close(self):
        self.viewer = None
        self._viewers.clear()

        for viewer in self._viewers.items():
            import glfw
            glfw.destroy_window(viewer.window)

    # default_window_width = DEFAULT_SIZE[0]
    # default_window_height = DEFAULT_SIZE[1]

    def _get_viewer(self, mode, cam_id) -> mujoco_py.MjViewer:
        mode_cam_id = mode, cam_id

        self.viewer = self._viewers.get(mode_cam_id)
        if self.viewer is not None:
            if sys.platform == 'darwin':
                # info: to fix the black image of death.
                self.viewer._set_mujoco_buffers()
            return self.viewer

        if mode == 'human':
            self.viewer = mujoco_py.MjViewer(self.sim)
            # we turn off the overlay and make the window smaller.
            self.viewer._hide_overlay = True
            import glfw
            glfw.set_window_size(self.viewer.window, self.width, self.height)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

        self.viewer_setup()
        self._viewers[mode_cam_id] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_dict(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ]).copy()

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    # def initialize_camera(self, init_fctn):
    #     sim = self.sim
    #     viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
    #     init_fctn(viewer.cam)
    #     sim.add_render_context(viewer)
