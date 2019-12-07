import numpy as np

alpha = 20*2*np.pi/360  #20 deg in radians
theta = np.random.uniform(-alpha, alpha)


def angle_mean(thetas):
    """
    Input: array of angles
    Output: circular mean of the angles

    https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    """
    x = np.mean(np.cos(thetas))
    y = np.mean(np.sin(thetas))
    theta_mean = np.atan2(y, x)
    return theta_mean


def make_s(num_points, motion_coherence):
    motions = np.random.uniform(low=-np.pi, high=np.pi, size = num_points)
    # set N/MC of the particles to the same value
    num_coherent = int(num_points * motion_coherence)
    motions[:num_points] = theta
    return angle_mean(motions)
    

def add_noise(s, num_points, motion_coherence, mean = 0):
    var = 1/motion_coherence/100
    noise = np.random.normal(mean, num_points**2*var)
    return s + noise
