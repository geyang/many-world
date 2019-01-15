# ge_world

The name is just a placeholder b/c I didn't have a better name. ge_ is the namespace for this package.

## To Develop

All research python modules need to have a proper `setup.py` file that installs the module. 

To start using this module, install this code in evaluation
mode by
```bash
cd ge_wold
pip install -e .
```

Now you should be able to import the module, and evaluate with 
updated code when it changes.


## Environments

Right now all environments are continuous. As we discussed, we might want to change the action to discrete action at some point to make it easy to use Q learning.

### PointMass

the state space it returns is 

```python
[x, y, x_dot, y_dot, goal_x, goal_y]
```

### Inverted Pendulum

We avoid the name `Pendulum-v0` b/c there is already a gym 
environment with the same name.

The state space it returns is 
```python
[cos(theta), sin(theta), theta_dot]
```
The reward is the angular difference between the current
angle theta, and the upright position.

### CartPole

- [ ] Still under construction
