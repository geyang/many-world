import numpy as np
from gym import spaces

from many_world import mujoco_env, base_envs, rewards

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
                 done_on_goal=False, is_collide=False, pattern='vertical', **_):
        """

        :param frame_skip:
        :param done_on_goal: False, bool. flag for setting done to True when reaching the goal
        """
        self.done_on_goal = done_on_goal
        self.obs_keys = obs_keys
        self.n_objs = n_objs
        self.qvel_high = qvel_high
        self.is_collide = is_collide
        self.pattern = pattern

        set_observation_spaces = True

        # call super init after initializing the variables.
        import os
        if is_collide:
            xml_path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), f"assets/many-world-{n_objs}-c.xml")
        else:
            xml_path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), f"assets/many-world-{n_objs}.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=frame_skip,
                                      set_observation_space=set_observation_spaces)
        base_envs.MazeCamEnv.__init__(self, **_)
        # utils.EzPickle.__init__(self)

        # note: Experimental, hard-coded
        self.width = 32
        self.height = 32
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

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
        # push distractors when they become too slow
        for i in range(self.n_objs-1):
            if self.pattern == 'vertical':
                obj_qvel = qvel[2*i+3]
                if np.absolute(obj_qvel) < 0.1:
                    direction = 2 * float(obj_qvel > 0) - 1
                    qvel[2*i+3] = 1.0 * direction
            elif self.pattern == 'random':
                obj_qvel = np.linalg.norm(qvel[2*i+2:2*i+4])
                if obj_qvel < 0.1:
                    qvel[2*i+2:2*i+4] = np.random.uniform(
                        low=-self.qvel_high, high=self.qvel_high, size=2)                    
        self.sim.data.qvel[2:-2] = qvel[2:-2]
        self.dist = np.linalg.norm(self._get_delta())
        self.do_simulation(a, self.frame_skip)
        # remove momentum
        self.sim.data.qvel[:2] = 0
        ob = np.concatenate([qpos, qvel[:-2]])
        new_dist = np.linalg.norm(self._get_delta())
        reward = self.dmc_reward_dense(new_dist)
    
        done = False
        return ob, reward, done, dict(dist=new_dist, success=float(new_dist < 0.07))

    def dmc_reward_dense(self, dist):
        target_size = 0.1
        near_target = rewards.tolerance(dist,
                                        bounds=(0, target_size), margin=target_size)
        return near_target

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
        self.set_state(self.init_qpos, self.init_qvel)
        if x is None:
            x = self.np_random.uniform(
                low=self.obj_low, high=self.obj_high, size=2 * self.n_objs)
        if goal is None:
            goal = self.np_random.uniform(
                low=self.goal_low, high=self.goal_high, size=2)
        
        if self.pattern == 'vertical':
            vel = np.zeros(2 * self.n_objs)
            vel[2::2] = 0
            vel[3::2] = 1.5
            # TODO randomize this a bit?
            x[2::2] = -0.2 + 0.2 * np.arange(self.n_objs-1)
            x[3::2] = -0.3
        elif self.pattern == 'random':
            vel = self.np_random.uniform(
                low=-self.qvel_high, high=self.qvel_high, size=2 * self.n_objs)
            vel[:2] = 0
        elif self.pattern == 'still':
            vel = np.zeros(2 * self.n_objs)
        else:
            raise NotImplementedError

        self.sim.data.qpos[:] = np.concatenate([x, goal])
        self.sim.data.qvel[:-2] = vel
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        ob = np.concatenate([qpos, qvel[:-2]])
        self.sim.forward()
        return ob

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta


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
        max_episode_steps=500
    )
