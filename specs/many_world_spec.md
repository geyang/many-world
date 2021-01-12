
# Many-world Environment

This environment contains a single table with multiple objects of 
different colors. Only one of the objects is controllable.

We also set the friction of the table to be small, to allow the 
blocks to slide, sometimes indefinitely.

| **n_objs = 1** | **n_objs = 2** | **n_objs = 3** | **n_objs = 4** |
|:--------------:|:--------------:|:--------------:|:--------------:|
| <img style="align-self:center;" src="figures/many_world-1.png?ts=006076" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-2.png?ts=296787" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-3.png?ts=443373" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/many_world-4.png?ts=573546" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |

```python
import gym

for i in range(1, 5):
    env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
    env.reset()
    img = env.render('rgb')

    plt.imshow(img)
    row.savefig(f"figures/many_world-{i}.png?ts={doc.now('%f')}", title=f"n_objs = {i}")
```
