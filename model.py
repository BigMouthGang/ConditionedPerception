import numpy as np
import probability_utils


alpha = 20*2*np.pi/360  #20 deg in radians
theta = -0.01#np.random.uniform(-alpha, alpha)


def angle_mean(thetas):
    """
    Input: array of angles
    Output: circular mean of the angles

    https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    """
    x = np.mean(np.cos(thetas))
    y = np.mean(np.sin(thetas))
    theta_mean = np.arctan2(y, x)
    return theta_mean


def make_s(num_points, motion_coherence):
    motions = np.random.uniform(low=-np.pi, high=np.pi, size = num_points)
    # set N/MC of the particles to the same value
    num_coherent = int(num_points * motion_coherence)
    motions[:num_coherent] = theta
    return angle_mean(motions)
    

def add_noise(s, num_points, motion_coherence, mean = 0):
    var = 1/motion_coherence/100
    noise = np.random.normal(mean, var)
    print("noise: ", noise)
    return s + noise

def normalize_angle(ang):
    while ang < -np.pi:
        ang += 2 * np.pi
    while ang > np.pi:
        ang -= 2*np.pi
    return ang

def main():
    MC = 0.12
    num_points = 1000
    prior_on_left = 0.5
    s = make_s(num_points, MC)
    print("s: ", s)
    m = normalize_angle(add_noise(s, num_points, MC))
    print("m: ", m)
    p_h_m_left = probability_utils.p_h_given_m('left', m, MC, prior_on_left, alpha)
    
    
    p_h_m_right = probability_utils.p_h_given_m('right', m, MC, prior_on_left, alpha)
    print("p left: ", p_h_m_left/(p_h_m_left + p_h_m_right))
    print("p right: ", p_h_m_right/(p_h_m_left + p_h_m_right))
    if p_h_m_left >= p_h_m_right:
        hmap = 'left'
    else:
        hmap = 'right'
    print(hmap)

    max_theta = None
    max_p_theta = 0
    epsilon = 0.5
    for theta in np.linspace(-alpha, alpha, 10):
        p_theta = probability_utils.p_theta_given_m(m, hmap, theta, MC, alpha)
        if p_theta > max_p_theta:
            max_theta = theta
            max_p_theta = p_theta
    print(max_theta)
if __name__ == "__main__":
    main()
