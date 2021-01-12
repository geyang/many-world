import matplotlib.pyplot as plt
from cmx import doc

doc @ """
# Many-world Environment

This environment contains a single table with multiple objects of 
different colors. Only one of the objects is controllable.

We also set the friction of the table to be small, to allow the 
blocks to slide, sometimes indefinitely.
"""

row = doc.table().figure_row()
with doc:
    import gym

    for i in range(1, 5):
        env = gym.make('many_world.many_world:ManyWorld-v0', n_objs=i)
        env.reset()
        img = env.render('rgb')

        plt.imshow(img)
        row.savefig(f"figures/many_world-{i}.png?ts={doc.now('%f')}", title=f"n_objs = {i}")

doc.flush()
