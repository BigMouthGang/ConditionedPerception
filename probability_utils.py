import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, uniform, norm
from scipy.integrate import quad


def p_m_given_theta_h(theta, hypothesis, MC):
    """
    Inputs:
        m : np.array of shape (1, n) of observed angles for n particles
        theta : float, overall angle of motion
        hypothesis : function 
    Outputs:
        p : float representing p(m | theta, hypothesis)
    """
    var = 1/MC/100
    n = norm(loc = theta, scale = var)
    return n


def p_theta_given_h(h, alpha):
    if h == 'left':
        u = uniform(loc=-alpha, scale=  alpha)
    elif h == 'right':
        u = uniform(loc=0, scale = alpha)
    else:
        print("h should be left or right")
    return u

def p_m_given_h(m, h, MC, alpha):
    func = lambda theta: p_theta_given_h(h, alpha).pdf(theta)*p_m_given_theta_h(theta, h, MC).pdf(m)
    prob = quad(func, -np.pi, np.pi)
    return prob[0]

def p_h(h, prior_on_left):
    if h == 'left':
        return prior_on_left
    else:
        return 1 - prior_on_left

def p_h_given_m(h, m, MC, prior_on_left, alpha):
    num = p_m_given_h(m, h, MC, alpha)
    return num*p_h(h, prior_on_left)

def p_theta_given_m(m, hmap, theta, MC, alpha):
    print(m, theta)
    num = p_m_given_theta_h(theta, hmap, MC).pdf(m)*p_theta_given_h(hmap, alpha).pdf(theta)
    denom = p_m_given_h(m, hmap, MC, alpha)
    print(num)
    print(denom)
    return num/denom


# func = lambda m: p_m_given_h(m, 'left', 0.01)[0]
# prob = quad(func, -np.pi, np.pi)
# print(prob)


# u = p_theta_given_h("right")
# x = np.linspace(u.ppf(0.01), u.ppf(0.99), 100)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, u.pdf(x))
# plt.show()

# u = p_m_given_h(0.2,'right', 0.03, 0.359)
# print(u)
# x = np.linspace(u.ppf(0.01), u.ppf(0.99), 100)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, u.pdf(x))
# plt.show()
# print(p_m_given_h(-0.5, 'left', .1))