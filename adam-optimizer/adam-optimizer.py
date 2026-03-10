import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param_np = np.asarray(param, dtype=float) 
    grad_np = np.asarray(grad, dtype=float)
    m_np = np.asarray(m, dtype=float)
    v_np = np.asarray(v, dtype=float)
    print("m_np", m_np)
    print("v_np", v_np)
    
    m_new = beta1 * m_np + (1 - beta1) * grad_np
    v_new = beta2 * v_np + (1 - beta2) * np.square(grad_np)
    print("m_new", m_new)
    print("v_new", v_new)

    m_bias = m_new / (1 - np.power(beta1, t))
    v_bias = v_new / (1 - np.power(beta2, t))
    print("m_bias", m_bias)
    print("v_bias", v_bias)

    param_new = param_np - m_bias / (np.sqrt(v_bias) + eps) * lr

    return param_new, m_new, v_new