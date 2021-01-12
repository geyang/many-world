import matplotlib.pyplot as plt
import numpy as np
from cmx import doc

from specs import render_video

doc @ """
# Many-world Environment

This environment contains a single table with multiple objects of 
different colors. Only one of the objects is controllable.

We also set the friction of the table to be small, to allow the 
blocks to slide, sometimes indefinitely.

## To Dos

- [x]  make the action space Box, and remove discrete option.
- [x]  dense reward only, use <dist to goal at t-1> - <dist to goal at t>
- [ ]  zero friction [looks good now, did not really tweak]
- [x]  test env.step with a servoing policy ( act = [goal - x] ), write the fixtures for gif gen.
- [x]  randomize the initial velocity, and add `init_vel=0.1` for obj and distractors.
- [x]  set `frame_skip=5` for slower motion.

"""

with doc @ """## Action Space: `Box(2,)`""":
    import gym

    env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=2)
    doc.print("The action space is: ", env.action_space)

doc @ """
## Dense Reward

The reward is computed as `r = old_dist - dist`, dist being the distance to the goal.
"""
with doc:
    env.seed(100)
    obs = env.reset()
    doc.print(obs)
    act = env.action_space.sample()
    obs, r, done, info = env.step(act)
    doc.print('the reward is:', r)

doc @ f"""
## Multiple Distractor Objects

Currently support up to 4.
"""
row = doc.table().figure_row()
with doc:
    for i in range(1, 5):
        env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
        env.seed(100)
        env.reset()
        img = env.render('rgb')

        plt.imshow(img)
        row.savefig(f"figures/many_world-{i}.png?ts={doc.now('%f')}", title=f"n_objs = {i}")

doc @ """
## Friction and Dynamics

We set the friction to 0, so that the distractor object move indefinitely.
The initial velocity is sampled with `vel_high` parameter in the environment
constructor.
"""
row = doc.table().figure_row()
with doc:
    for i in range(1, 5):
        def servo_pi(obs, env):
            act = np.zeros_like(env.action_space.high)
            act[:2] = (obs[2 * i:2 * i + 2] - obs[:2])
            return act

        env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
        env.seed(100)
        render_video("ManyWorld-v0", 15, row, env=env, pi=servo_pi,
                     title=f"n_objs={i}", filename=f"many_world-{i}.gif")
