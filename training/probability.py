import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binom, norm
from statistics import NormalDist


def plot_cumwithn(p, q):
    n_values = np.arange(30, 500, 20)
    p_values = [NormalDist(mu = (p-q)*n, sigma = math.sqrt(n*(p*(1-p) + q*(1-q)))).cdf(0) for n in n_values]
    plt.plot(n_values, p_values, marker='o', linestyle='-', color='r')
    plt.title(f'Probability of incorrect ranking given p={p} and q={q}')
    plt.xlabel('Number of trials (n)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()

p = 0.75
cum_prob = 0.95
step = 0.01
#plot_cumwithn(0.75, 0.70)
#f = q + p - 2 * p * q
#r = p*(1-q)/f
#sample_sizes = [round(n*f) for n in n_values]
#b = [math.ceil(round(n*f)/2)  - 1 for n in n_values]
#p_values = [binom.cdf(math.ceil(round(n*f)/2)  - 1, round(n*f), r) for n in n_values]
#plt.plot(n_values, p_values, marker='o', linestyle='-', color='b')


'''q_values = np.arange(p - step, 0.5, -step)
q_values_inv = [p - q for q in q_values]
n_values = [norm.ppf(cum_prob)**2 * (p * (1-p) + q * (1-q)) / (p-q)**2 for q in q_values]
plt.plot(q_values_inv, n_values, marker='o', linestyle='-', color='r')
plt.title(f'Required trials for probability of misordering to be {1 - cum_prob:.2f} given p={p}')
plt.ylim((0, 1000))
plt.xlabel('Difference in probability between p and q')
plt.ylabel('Required number of trials (n)')
plt.grid(True)
plt.show()'''

p_values = np.arange(0.95 - step, 0.6, -step)
q_values = np.arange(0.1, step, -step)
p_values, q_values = np.meshgrid(p_values, q_values)
n_values = norm.ppf(cum_prob)**2 * (p_values * (1-p_values) + (p_values - q_values) * (1-p_values + q_values)) / (q_values)**2

plt.pcolormesh(q_values, p_values, n_values, vmin=10, vmax=1000, shading="gouraud")
plt.colorbar(label='Number of Trials')
contour = plt.contour(q_values, p_values, n_values, levels=[200, 500, 1000], colors='red')
plt.clabel(contour, inline=True, fontsize=8)
plt.ylabel('Probability of Accurate Model')
plt.xlabel('Difference in Probability')
plt.title("Color plot of trials needed to reach 0.05 probability of misordering")
plt.show()

n = 1000
misorder_values = norm.cdf(0, loc=q_values * n, scale=(n * (p_values * (1-p_values) + (p_values - q_values) * (1-p_values + q_values)))**0.5)
plt.contourf(q_values, p_values, misorder_values, levels=15, cmap='viridis')
plt.colorbar(label='Probability of Misordering')
contour = plt.contour(q_values, p_values, misorder_values, levels=[0.05, 0.1], colors='red')
plt.clabel(contour, inline=True, fontsize=8)
plt.ylabel('Probability of Accurate Model')
plt.xlabel('Difference in Probability')
plt.title(f"Color plot of probability of misordering given {n} trials")
plt.show()