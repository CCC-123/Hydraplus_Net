import visdom
import numpy as np

vis = visdom.Visdom()
vis.text("gg")
vis.image(np.ones(3,10,10))