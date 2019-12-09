import numpy as np
import probability_utils
import matplotlib.pyplot as plt
import pickle

alpha = 20*2*np.pi/360  #20 deg in radians
# theta = -0.01#np.random.uniform(-alpha, alpha)


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


def make_s(num_points, motion_coherence, theta):
    # return theta
    motions = np.random.uniform(low=-np.pi, high=np.pi, size = num_points)
    # set N/MC of the particles to the same value
    num_coherent = int(num_points * motion_coherence)
    motions[:num_coherent] = theta
    return angle_mean(motions)
    

def add_noise(s, num_points, motion_coherence, mean = 0):
    var = 1/motion_coherence/200
    noise = np.random.normal(mean, var)
    print("noise: ", noise)
    return s + noise

def normalize_angle(ang):
    while ang < -np.pi:
        ang += 2 * np.pi
    while ang > np.pi:
        ang -= 2*np.pi
    return ang

def main(theta, MC):
    num_points = 10000
    prior_on_left = 0.5
    s = make_s(num_points, MC, theta)
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
    epsilon = 0.05
    #half a degree increments
    for th in np.linspace(-alpha-epsilon, alpha+epsilon, 40):#np.linspace(-alpha, alpha, 10):
        p_theta = probability_utils.p_theta_given_m(m, hmap, th, MC, alpha)
        if p_theta > max_p_theta:
            max_theta = th
            max_p_theta = p_theta
    return max_theta, p_h_m_right/(p_h_m_left + p_h_m_right)


if __name__ == "__main__":
    num_iterations = 2
    num_datapoints_on_each_side = 3
    color_dic = {
        0.03: [0, 0, 1],
        0.06: [1, 0, 0],
        0.12: [0, 1, 0]
    }
    MC_thetas = {}
    MC_probs = {}
    for MC in [0.03, 0.06, 0.12]:
        print("MC: ", MC)
        thetas = []
        max_thetas= []
        probs = []
        for theta in np.linspace(-alpha, alpha, 2*num_datapoints_on_each_side+1):
            if theta == 0:
                thetas.append(0)
                max_thetas.append(0)
                # probs.append(0)
                # continue
            tot_angle = []
            tot_probability = 0
            for i in range(num_iterations):
                max_theta, p_right = main(theta, MC)
                tot_angle.append(max_theta)
                tot_probability+=p_right
            mean_angle = angle_mean(np.array(tot_angle))
            max_thetas.append(mean_angle * 360 / (2*np.pi))
            probs.append(tot_probability/num_iterations) 
            thetas.append(theta * 360/(2*np.pi))
            print("result: ", theta, mean_angle, tot_probability/num_iterations)
        # ax.plot(thetas, max_thetas, color=color_dic[MC], label="Motion Coherence of: %s" %MC)
        # ax.plot(thetas, probs, color=color_dic[MC], label="Motion Coherence of: %s"%MC)
        MC_thetas[MC] = max_thetas
        MC_probs[MC] = probs
        print(thetas)
        print(max_thetas)
        print(probs)
    print(MC_thetas)
    print(MC_probs)
    print(thetas)
    
    max_theta_pickle_path = "pickled_data/max_thetas_%s_datapoints_%s_iterations_v2" %(num_datapoints_on_each_side*2+1, num_iterations)
    probs_pickle_path = "pickled_data/probs_to_right_%s_datapoints_%s_iterations_v2" %(num_datapoints_on_each_side*2+1, num_iterations)
    thetas_pickle_path = "pickled_data/thetas_linspace_%s_datapoints_%s_iterations_v2" %(num_datapoints_on_each_side*2+1, num_iterations)

    with open(max_theta_pickle_path, 'wb') as f:
        pickle.dump(MC_thetas, f)
    with open(probs_pickle_path, 'wb') as f:
        pickle.dump(MC_probs, f)
    with open(thetas_pickle_path, 'wb') as f:
        pickle.dump(thetas, f)


    # with open(max_theta_pickle_path, 'rb') as f:
    #     MC_thetas = pickle.load(f)
    # with open(probs_pickle_path, 'rb') as f:
    #     MC_probs = pickle.load(f)
    # with open(thetas_pickle_path, 'rb') as f:
    #     thetas = pickle.load(f)

    #plot max thetas
    fig, ax = plt.subplots(1, 1)
    for MC in [0.03, 0.06, 0.12]:
        ax.plot(thetas, MC_thetas[MC], color = color_dic[MC], label="Motion Coherence of: %s"%MC)
    plt.xlabel("Actual Degree")
    plt.ylabel("Estimated Direction")
    plt.title("Estimated Direction vs Actual Degree")
    ax.legend()
    plt.savefig("plots/motion_estimation_main_MC_%s_datapoints_%s_iterations_v2.png" %(num_datapoints_on_each_side*2+1, num_iterations))
    plt.show()

    fig, ax = plt.subplots(1, 1)
    for MC in [0.03, 0.06, 0.12]:
        ax.plot(thetas, MC_probs[MC], color=color_dic[MC], label="Motion Coherence of: %s"%MC)
    plt.xlabel("Actual Degree")
    plt.ylabel("Fraction motion right of reference")
    ax.legend()
    plt.savefig("plots/fraction_to_the_right_MC_%s_datapoints_%s_iterations_v2.png" %(num_datapoints_on_each_side*2+1, num_iterations))
    plt.show()
