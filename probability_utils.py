import numpy as numpy


def p_m_given_theta_h(m, theta, hypothesis, MC):
    """
    Inputs:
        m : np.array of shape (1, n) of observed angles for n particles
        theta : float, overall angle of motion
        hypothesis : function 
    Outputs:
        p : float representing p(m | theta, hypothesis)
    """
    var = 1/MC