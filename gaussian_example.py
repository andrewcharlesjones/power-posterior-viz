import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from os.path import join as pjoin
import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = (
    "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"
)

## Set up model
likelihood_variance = 1
prior_mean = 0
prior_variance = 1
true_mean = 1.5
gamma = 0.5

## Generate data from 1D Gaussian
n = 10
X = np.random.normal(loc=true_mean, scale=1, size=n)


## Function for computing posterior mean and variance
def power_posterior(X, gamma=1):
    n = len(X)
    posterior_mean = gamma / n * np.sum(X)
    posterior_variance = 1 / (gamma * n + 1)
    return posterior_mean, posterior_variance


## Compute standard posterior
bp_mean, bp_variance = power_posterior(X, gamma=1)

## Compute coarsened posterior
pp_mean, pp_variance = power_posterior(X, gamma=gamma)


## Set up plot
plt.figure(figsize=(7, 5))
zs = np.linspace(-3, 3, 200)

axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.axvline(X[0], color="gray", ymin=ymin, ymax=ymin + 0.1, label="Data")
for ii in range(1, n):
    plt.axvline(X[ii], color="gray", ymin=ymin, ymax=ymin + 0.1)

bp_curve = norm.pdf(zs, bp_mean, bp_variance)
pp_curve = norm.pdf(zs, pp_mean, pp_variance)

plt.plot(zs, bp_curve, color="black", linewidth=3, label="Standard posterior")
plt.plot(
    zs,
    pp_curve,
    color="red",
    linewidth=3,
    label=r"Power posterior, $\gamma={}$".format(gamma),
)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$p(\mu | x_1, \dots, x_n)$")
plt.legend()
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "power_posterior_gaussian.png"))
plt.close()


import matplotlib.animation as animation
import matplotlib.image as mpimg
import os

for ii in range(n):

    bp_mean, bp_variance = power_posterior(X[: ii + 1], gamma=1)
    pp_mean, pp_variance = power_posterior(X[: ii + 1], gamma=gamma)

    bp_curve = norm.pdf(zs, bp_mean, bp_variance)
    pp_curve = norm.pdf(zs, pp_mean, pp_variance)

    plt.figure(figsize=(7, 5))
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    # import ipdb; ipdb.set_trace()
    for jj in range(ii + 1):
        plt.axvline(X[jj], color="gray", ymin=ymin, ymax=ymin + 0.1)
    plt.plot(zs, bp_curve, color="black", linewidth=3, label="Standard posterior")
    plt.plot(
        zs,
        pp_curve,
        color="red",
        linewidth=3,
        label=r"Power posterior, $\gamma={}$".format(gamma),
    )
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$p(\mu | x_1, \dots, x_n)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./tmp/tmp{}".format(ii))
    plt.close()

fig = plt.figure()
ims = []
for ii in range(n):
    fname = "./tmp/tmp{}.png".format(ii)
    img = mpimg.imread(fname)
    im = plt.imshow(img)
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    ims.append([im])
    os.remove(fname)

writervideo = animation.FFMpegWriter(fps=2)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
ani.save(
    pjoin(SAVE_DIR, "power_posterior_gaussian_animation.mp4"),
    writer=writervideo,
    dpi=1000,
)
plt.close()
