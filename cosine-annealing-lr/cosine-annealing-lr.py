import math as m

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + m.cos(m.pi * current_step / total_steps))

    return lr