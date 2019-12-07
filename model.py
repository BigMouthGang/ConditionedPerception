import numpy as np
import probability_utils


alpha = 20*2*np.pi/360  #20 deg in radians
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

def main():
    print(theta)
    MC = 0.03
    num_points = 100
    prior_on_left = 0.5
    s = make_s(num_points, MC)
    m = add_noise(s, num_points, MC)
    p_h_m_left = probability_utils.p_h_given_m('left', m, MC, prior_on_left)
    p_h_m_right = probability_utils.p_h_given_m('right', m, MC, prior_on_left)
    if p_h_m_left >= p_h_m_right:
        hmap = 'left'
    else:
        hmap = 'right'
    print(hmap)

if __name__ == "__main__":
    main()
