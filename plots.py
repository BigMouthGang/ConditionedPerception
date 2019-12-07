"""
File for creating plots for the paper
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform


def plot_uniform_theta(alpha = np.pi/9):
	"""
	Plots uniform theta on [-alpha to alpha]
	"""
	rv = uniform(loc=-alpha, scale = 2*alpha)
	

	fig, ax = plt.subplots(1,1)

	x = np.linspace(rv.ppf(0.01),
                 rv.ppf(0.99), 100)
	ax.plot(x, rv.pdf(x))
	plt.xlabel('$\\theta$')
	plt.ylabel('$f(\\theta)$')
	plt.savefig('plots/uniform_theta.png')


if __name__ == '__main__':
	plot_uniform_theta()