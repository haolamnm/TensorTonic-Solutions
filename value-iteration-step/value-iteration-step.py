import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    num_actions = len(transitions[0])
    new_values = np.zeros(num_states)
    
    rewards = np.asarray(rewards)
    transitions = np.asarray(transitions)

    for s in range(num_states):
        action_values = []
        for a in range(num_actions):
            expected_v = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
            action_values.append(expected_v)

        new_values[s] = max(action_values)

    return new_values.tolist()