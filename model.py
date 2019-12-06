import numpy as np

alpha = 0.3490658504 #20 deg in radians
theta = np.random.uniform(-alpha, alpha)


def make_s(num_points, motion_coherence):
    motions = np.zeros((num_points, 1))
    num_coherent = int(num_points * motion_coherence)
    s = num_coherent * theta
    for i in range(num_points - num_coherent):
        rand_angle = np.random.uniform(-np.pi, np.pi)
        s += rand_angle
    return s/num_points

def add_noise(s, num_points, motion_coherence, mean = 0):
    var = 1/motion_coherence/100
    noise = np.random.normal(mean, num_points**2*var)
    return s + noise
