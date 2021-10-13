import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from os.path import join as pjoin
import matplotlib
from scipy.special import expit

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = (
    "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"
)

n = 200
xs = np.linspace(-10, 10, n)
ys = expit(xs)

plt.figure(figsize=(17, 7))
max_val = np.max(ys)

################ STANDARD POSTERIOR

## Prior
plt.subplot(121)
plt.plot(np.linspace(-50, -30, n), ys, color="blue", linewidth=4)
plt.plot(np.linspace(30, 50, n), np.flip(ys), color="blue", linewidth=4)

flat_line = np.array([[-30, ys[-1]], [30, ys[-1]]])
plt.plot(
    flat_line[:, 0], flat_line[:, 1], color="blue", linewidth=4, label=r"$p(\theta)$"
)
# plt.arrow(x=-30, y=ys[-1] + 0.2, dx=60, dy=0, linewidth=4, head_length=0.01)
plt.arrow(
    x=-35,
    y=ys[-1] + 0.2,
    dx=70,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
arrow = plt.arrow(
    x=35,
    y=ys[-1] + 0.2,
    dx=-70,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
plt.text(x=13, y=ys[-1] + 0.4, s=r"$\Delta \theta_{prior}$")

## Posterior
ys_posterior = ys * 4
plt.plot(np.linspace(-15, -5, n), ys_posterior, color="red", linewidth=4)
plt.plot(np.linspace(5, 15, n), np.flip(ys_posterior), color="red", linewidth=4)

flat_line = np.array([[-5, ys_posterior[-1]], [5, ys_posterior[-1]]])
plt.plot(
    flat_line[:, 0], flat_line[:, 1], color="red", linewidth=4, label=r"$p(\theta | x)$"
)

plt.arrow(
    x=-5,
    y=ys_posterior[-1] + 0.2,
    dx=13,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
arrow = plt.arrow(
    x=5,
    y=ys_posterior[-1] + 0.2,
    dx=-13,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
plt.text(x=-5, y=ys_posterior[-1] + 0.5, s=r"$\Delta \theta_{MAP}$")

plt.xticks([], [])
plt.yticks([], [])

axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.ylim([0, ymax + 0.7])


plt.xlabel(r"$\theta$")
# plt.ylabel(r"$p(\theta)$ or $p(\theta | x)$"))
plt.title("Standard posterior")
plt.tight_layout()


########### POWER POSTERIOR

plt.subplot(122)

## Prior
gamma = 0.7
plt.plot(np.linspace(-50, -30, n), ys, color="blue", linewidth=4)
plt.plot(np.linspace(30, 50, n), np.flip(ys), color="blue", linewidth=4)

flat_line = np.array([[-30, ys[-1]], [30, ys[-1]]])
plt.plot(
    flat_line[:, 0], flat_line[:, 1], color="blue", linewidth=4, label=r"$p(\theta)$"
)
plt.arrow(
    x=-35,
    y=ys[-1] + 0.2,
    dx=70,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
arrow = plt.arrow(
    x=35,
    y=ys[-1] + 0.2,
    dx=-70,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
plt.text(x=13, y=ys[-1] + 0.4, s=r"$\Delta \theta_{prior}$")

## Posterior
ys_posterior *= gamma
plt.plot(np.linspace(-15, -5, n), ys_posterior, color="red", linewidth=4)
plt.plot(np.linspace(5, 15, n), np.flip(ys_posterior), color="red", linewidth=4)

flat_line = np.array([[-5, ys_posterior[-1]], [5, ys_posterior[-1]]])
plt.plot(
    flat_line[:, 0], flat_line[:, 1], color="red", linewidth=4, label=r"$p(\theta | x)$"
)

plt.arrow(
    x=-5,
    y=ys_posterior[-1] + 0.2,
    dx=13,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
arrow = plt.arrow(
    x=5,
    y=ys_posterior[-1] + 0.2,
    dx=-13,
    dy=0,
    head_width=0.2,
    head_length=3,
    linewidth=4,
    color="black",
    length_includes_head=True,
)
plt.text(x=-5, y=ys_posterior[-1] + 0.5, s=r"$\Delta \theta_{MAP}$")

plt.ylim([0, ymax])

plt.xticks([], [])
plt.yticks([], [])


plt.xlabel(r"$\theta$")
# plt.ylabel(r"$p(\theta)$ or $p(\theta | x)$")
plt.title(r"Power posterior, $\gamma={}$".format(gamma))
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()

plt.savefig(pjoin(SAVE_DIR, "power_posterior_complexity_diagram.png"))
plt.show()
# import ipdb; ipdb.set_trace()
