# Many-world Environment

This environment contains a single table with multiple objects of 
different colors. Only one of the objects is controllable.

We also set the friction of the table to be small, to allow the 
blocks to slide, sometimes indefinitely.

## To Install

You can install directly from pip

```bash
pip install many-world
```



## To Dos

- [x]  make the action space Box, and remove discrete option.
- [x]  dense reward only, use <dist to goal at t-1> - <dist to goal at t>
- [ ]  zero friction [looks good now, did not really tweak]
- [x]  test env.step with a servoing policy ( act = [goal - x] ), write the fixtures for gif gen.
- [x]  #### randomize the initial velocity, and add `init_vel=0.1` for obj and distractors.
- [x]  set `frame_skip=5` for slower motion.


## Action Space: `Box(2,)`

```python
import gym

env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=2)
doc.print("The action space is: ", env.action_space)
```

```
The action space is:  Box(4,)
```

## Dense Reward

The reward is computed as `r = old_dist - dist`, dist being the distance to the goal.

```python
env.seed(100)
obs = env.reset()
doc.print(obs)
act = env.action_space.sample()
obs, r, done, info = env.step(act)
doc.print('the reward is:', r)
```

```
[-0.06514147  0.09519143 -0.15679238 -0.0590579  -0.13886868 -0.0413905
  0.00762082  0.0534794   0.07483203  0.08715882]
the reward is: 0.14956679130283426
```

## Multiple Distractor Objects

Currently support up to 4.

|                        **n_objs = 1**                        |                        **n_objs = 2**                        |                        **n_objs = 3**                        |                        **n_objs = 4**                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img style="align-self:center;" src="figures/many_world-1.png?ts=243236" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-2.png?ts=446009" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-3.png?ts=588259" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-4.png?ts=725175" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |

```python
for i in range(1, 5):
    env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
    env.seed(100)
    env.reset()
    img = env.render('rgb')

    plt.imshow(img)
    row.savefig(f"figures/many_world-{i}.png?ts={doc.now('%f')}", title=f"n_objs = {i}")
```

## Friction and Dynamics

We set the friction to 0, so that the distractor object move indefinitely.
The initial velocity is sampled with `vel_high` parameter in the environment
constructor.

|                         **n_objs=1**                         |                         **n_objs=2**                         |                         **n_objs=3**                         |                         **n_objs=4**                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![figures/many_world-1.gif?ts=823593](figures/many_world-1.gif?ts=823593) | ![figures/many_world-2.gif?ts=220373](figures/many_world-2.gif?ts=220373) | ![figures/many_world-3.gif?ts=848789](figures/many_world-3.gif?ts=848789) | ![figures/many_world-4.gif?ts=835134](figures/many_world-4.gif?ts=835134) |

```python
for i in range(1, 5):
    def servo_pi(obs, env):
        act = np.zeros_like(env.action_space.high)
        act[:2] = (obs[2 * i:2 * i + 2] - obs[:2])
        return act


    env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
    env.seed(100)
    render_video("ManyWorld-v0", 15, row, env=env, pi=servo_pi,
                 title=f"n_objs={i}", filename=f"many_world-{i}.gif")
```

## Specs and Documentation

For up-to-date, live documentation, take a look at the [specs folder](specs). We use the library [`cmx-python`](https://github.com/cmx/cmx-python)  to generate markdown files like this, containing figures, automatically. 



## To Develop

Our research python modules need a proper `setup.py` file that installs the module. First clone this repository, then

```bash
cd many_world
pip install -e .
```

We also need the following packages:

```python
pip install cmx
```

This is a python modul that allows us to generate markdown readmes like this automatically, as if we are writing a jupyter notebook (but your script is actually python).

**The script that generated this README automatically could be found here:** [specs/many_world_spec.py](specs/many_world_spec.py)
