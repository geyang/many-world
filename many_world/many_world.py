import numpy as np
from gym import spaces

from many_world import mujoco_env, base_envs


class ManyWorldEnv(mujoco_env.MujocoEnv, base_envs.MazeCamEnv):
    """
    2D Point Mass Environment. Uses torque control.
    """
    achieved_key = 'x'
    desired_key = 'goal'
    is_good_goal = lambda *_: True
    is_good_state = lambda *_: True

    def __init__(self, frame_skip=6, obs_keys=(achieved_key, desired_key),
                 n_objs=4, qvel_high=0.1, obj_low=-0.2, obj_high=0.2, goal_low=-0.2, goal_high=0.2,
                 done_on_goal=False, **_):
        """

        :param frame_skip:
        :param done_on_goal: False, bool. flag for setting done to True when reaching the goal
        """
        self.done_on_goal = done_on_goal
        self.obs_keys = obs_keys
        self.n_objs = n_objs
        self.qvel_high = qvel_high

        set_observation_spaces = True

        # call super init after initializing the variables.
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/many-world-{n_objs}.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=frame_skip,
                                      set_observation_space=set_observation_spaces)
        base_envs.MazeCamEnv.__init__(self, **_)
        # utils.EzPickle.__init__(self)

        # note: Experimental, hard-coded
        self.width = 64
        self.height = 64
        _ = dict()
        if 'x' in obs_keys:
            _['x'] = spaces.Box(low=np.array([-0.3, -0.3]), high=np.array([0.3, 0.3]))
        if 'goal' in obs_keys:
            # todo: double check for agreement with actual goal distribution
            _['goal'] = spaces.Box(low=np.array([goal_low, goal_low]), high=np.array([goal_high, goal_high]))
        if 'img' in obs_keys:
            _['img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        if 'goal_img' in obs_keys:
            _['goal_img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        self.obj_low = obj_low
        self.obj_high = obj_high
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.observation_space = spaces.Dict(_)
        self.obs_keys = obs_keys

    def compute_reward(self, achieved, desired, *_):
        success = np.linalg.norm(achieved - desired, axis=-1) < 0.02
        return (success - 1).astype(float)

    reach_counts = 0

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        self.dist = np.linalg.norm(self._get_delta())
        self.do_simulation(a, self.frame_skip)
        # note: return observation *after* simulation. This is how DeepMind Lab does it.
        # remove the pos and vel of the target sphere
        ob = np.concatenate([qpos, qvel[:-2]])
        reward = np.linalg.norm(self._get_delta()) - self.dist
        done = False

        return ob, reward, done, dict(dist=self.dist, success=float(self.dist < 0.02))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        # side-view
        # self.viewer.cam.lookat[2] = -0.1
        # self.viewer.cam.distance = .7
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 90

        # ortho-view
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = .74
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def reset_model(self, x=None, goal=None):
        self.reach_counts = 0
        if x is None:
            x = self.np_random.uniform(low=self.obj_low, high=self.obj_high, size=2 * self.n_objs)
        vel = self.np_random.uniform(low=0, high=self.qvel_high, size=2 * self.n_objs)
        if goal is None:
            goal = self.np_random.uniform(low=self.goal_low, high=self.goal_high, size=2)
        # self.controls.sample_goal(goals)
        self.sim.data.qpos[:] = np.concatenate([x, goal])
        self.sim.data.qvel[:-2] = vel  # no velocity
        # self.set_state(qpos, qvel)
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        ob = np.concatenate([qpos, qvel[:-2]])
        return ob

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    # def _get_obs(self):
    #     obs = {}
    #     qpos = self.sim.data.qpos.flat.copy()
    #     if 'x' in self.obs_keys:
    #         obs['x'] = qpos[:2].copy()
    #     if 'img' in self.obs_keys:
    #         goal = qpos[2:].copy()
    #         qpos[2:] = [.3, .3]  # move goal out of frame
    #         self.set_state(qpos, self.sim.data.qvel)
    #         # todo: should use render('gray') instead.
    #         obs['img'] = self.render('grey', width=self.width, height=self.height).transpose(0, 1)[None, ...]
    #         qpos[2:] = goal
    #         self.set_state(qpos, self.sim.data.qvel)
    #     if 'goal' in self.obs_keys:
    #         obs['goal'] = qpos[2:].copy()
    #     if 'goal_img' in self.obs_keys:
    #         curr_qpos = qpos.copy()
    #         qpos[:2] = qpos[2:].copy()
    #         qpos[2:] = [.3, .3]  # move goal out of frame
    #         self.set_state(qpos, self.sim.data.qvel)
    #         # todo: should use render('gray') instead.
    #         obs['goal_img'] = self.render('grey', width=self.width, height=self.height).transpose(0, 1)[None, ...]
    #         self.set_state(curr_qpos, self.sim.data.qvel)
    #     return obs


from gym.envs import register

if __name__ == "__main__":
    import gym

    env = gym.make('GoalMassDiscrete-v0')
    env.reset()
    frame = env.render('rgb', width=200, height=200)
    from os.path import basename
    from ml_logger import logger

    logger.log_image(frame, f"../figures/{basename(__file__)}:{env.spec.id}.png")

    # to show thy human.
    from PIL import Image

    im = Image.fromarray(frame)
    im.show()
else:
    # note: kwargs are not passed in to the constructor when entry_point is a function.
    register(
        id="ManyWorld-v0",
        entry_point=ManyWorldEnv,
        kwargs=dict(n_objs=1, ),
        max_episode_steps=50
    )
