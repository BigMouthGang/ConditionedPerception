"""
File for creating plots for the paper
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import pickle
import pdb

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


def fit_polynomial(thetas, MC_thetas, MC):

	degree = 4

	zero_index = thetas.index(0)

	th1 = thetas[:zero_index]
	th2 = thetas[zero_index + 1:]
	y1 = MC_thetas[MC][:zero_index]
	y2 = MC_thetas[MC][zero_index + 1:]

	
	p1 = np.polyfit(th1, y1, degree)
	p2 = np.polyfit(th2,y2, degree)

	x1 = np.linspace(min(th1), max(th1), 100)
	x2 = np.linspace(min(th2), max(th2), 100)

	z1 = np.polyval(p1, x1)
	z2 = np.polyval(p2, x2)

	X = list(x1) + [0] + list(x2)
	Z = list(z1) + [np.nan] + list(z2)
	return X, Z


def fit_polynomial_direction(thetas, MC_thetas, MC):

	degree = 2

	zero_index = thetas.index(0)

	th1 = thetas[:zero_index]
	th2 = thetas[zero_index + 1:]
	y1 = MC_thetas[MC][:zero_index]
	y2 = MC_thetas[MC][zero_index + 1:]

	
	p1 = np.polyfit(th1, y1, degree)
	p2 = np.polyfit(th2,y2, degree)

	x1 = np.linspace(min(th1), max(th1), 100)
	x2 = np.linspace(min(th2), max(th2), 100)

	z1 = np.polyval(p1, x1)
	z2 = np.polyval(p2, x2)

	X = list(x1) + [0] + list(x2)
	Z = list(z1) + [np.nan] + list(z2)
	return X, Z


def fit_polynomial_fraction(thetas, MC_probs, MC, degree=5):
	p = np.polyfit(thetas, MC_probs[MC], degree)
	X = np.linspace(min(thetas), max(thetas), 100)
	Z = np.polyval(p, X)
	return X,Z

def plot_smooth_motion_estimation_with_disconuity(datapoints, num_iterations):
    
	color_dic = {
	    0.03: [0, 0, 1],
	    0.06: [1, 0, 0],
	    0.12: [0, 1, 0]
	}

	max_theta_pickle_path = "pickled_data/max_thetas_%s_datapoints_%s_iterations" %(datapoints, num_iterations)
	probs_pickle_path = "pickled_data/probs_to_right_%s_datapoints_%s_iterations" %(datapoints, num_iterations)
	thetas_pickle_path = "pickled_data/thetas_linspace_%s_datapoints_%s_iterations" %(datapoints, num_iterations)

	with open(max_theta_pickle_path, 'rb') as f:
		MC_thetas = pickle.load(f)
	with open(probs_pickle_path, 'rb') as f:
		MC_probs = pickle.load(f)
	with open(thetas_pickle_path, 'rb') as f:
		thetas = pickle.load(f)

	fig, ax = plt.subplots(1, 1)
	for MC in [0.03, 0.06, 0.12]:
		X, Z =  fit_polynomial_direction(thetas, MC_thetas, MC)
		#ax.plt.plot(X,Z)
		ax.plot(X, Z, color = color_dic[MC], label="Motion Coherence of: %s"%MC)
		#ax.plot(thetas, MC_thetas[MC], color = color_dic[MC], label="Motion Coherence of: %s"%MC)
	ax.plot(thetas, thetas, '--', label='Estimated Direction=Actual Degree')

	plt.xlabel("Actual Degree")
	plt.ylabel("Estimated Degree")
	plt.title("Estimated Degree vs Actual Degree")
	ax.legend()

	plt.show()

	fraction_degree_dict = {
		0.03 : 2,
		0.06 : 2,
		0.12 : 3
	}

	fig, ax = plt.subplots(1, 1)
	for MC in [0.03, 0.06, 0.12]:
		degree = fraction_degree_dict[MC]
		X, Z = fit_polynomial_fraction(thetas, MC_probs, MC, degree)
		ax.plot(X, Z, color=color_dic[MC], label="Motion Coherence of: %s"%MC)

	plt.xlabel("Actual Degree")
	plt.ylabel("Fraction motion right of reference")
	ax.legend()
	#plt.savefig("plots/fraction_to_the_right_MC_%s_datapoints_%s_iterations_v2.png" %(num_datapoints_on_each_side*2+1, num_iterations))
	plt.show()

    #plt.savefig("plots/motion_estimation_main_MC_%s_datapoints_%s_iterations_v2.png" %(num_datapoints_on_each_side*2+1, num_iterations))
    #plt.show()   

if __name__ == '__main__':
	plot_smooth_motion_estimation_with_disconuity(15,40)
	#plot_uniform_theta()