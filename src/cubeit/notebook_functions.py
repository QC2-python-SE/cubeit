
import pytest
import numpy as np
from cubeit.gates_DMs import(
    x, y, z, had, s, t, cnot, cnot_10, phase, rotation_x, rotation_y, rotation_z, swap, cz, cphase
)
from cubeit.visualisation import simulate_measurements

from cubeit.DMs import DensityMatrix2Qubit as DM2
from cubeit.DMs import DensityMatrix1Qubit as DM1
import matplotlib.pyplot as plt



def plot_measure_DMs(state, meas_shots=1000):
    fig, ax = plt.subplots(figsize=(3, 5))
    basis_states = {
        '|00⟩': 0,
        '|01⟩': 0,
        '|10⟩': 0,
        '|11⟩': 0
    }
    
    ideal_probs = dict.fromkeys(basis_states, 0)
    noisy_probs = dict.fromkeys(basis_states, 0)
    
    meast, ideal = state.measure_shots(shots=[meas_shots])

    for k in basis_states:
        k_clean = k.replace('|', '').replace('⟩', '')
        if k_clean in ideal:
            ideal_probs[k] = ideal[k_clean]
        if k_clean in meast[0]:
            noisy_probs[k] = meast[0][k_clean]

    # Positions for bars
    x = np.arange(len(basis_states))
    width = 0.35  # bar width

    # Plot bars side by side
    ax.bar(x - width/2, list(ideal_probs.values()), width, label='Ideal', color='dodgerblue')
    ax.bar(x + width/2, list(noisy_probs.values()), width, label=f'Shots', color='darkorange', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(list(basis_states.keys()))
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
    ax.set_title(f'Measurement over {meas_shots} shots')
    ax.set_ylabel('Probability')
    plt.show()
    return fig

def get_gates_targets(string):
    """
    Parses user input string to extract quantum gates and their targets.
    
    Args:
        string (str): Input string containing Gate(target) commands separated by semicolons. For rotations, the format is Gate(arg)(target).

    Returns:
        tuple: A tuple containing two lists - one for gates (with arguments if applicable) and another for targets.
    """
    gates = []
    targets = []
    commands = string.split(';')        #Iterate over each command separated by semicolon
    for cmd in commands:
        if len(cmd)>1:      
            cmd = cmd.strip()
            if not cmd or "(" not in cmd or not cmd.endswith(")"):
                print("Invalid command:", cmd)
                continue
            if ")(" in cmd:         #Check for rotation gates with arguments
                cmd = cmd.split('(')
                gate = cmd[0]
                arg = cmd[1][:-1]
                target = cmd[2][:-1]
                if ',' in target:
                    sub_targets = target.split(',')
                    target = [int(sub_targets[0]), int(sub_targets[1])]
                else:
                    target = int(target)
                gates.append((gate,arg))
                targets.append(target)
            else:                   #Standard gates without arguments
                cmd = cmd.split('(')
                gate = cmd[0]
                target = cmd[1][:-1]
                if ',' in target:
                    sub_targets = target.split(',')
                    target = [int(sub_targets[0]), int(sub_targets[1])]
                else:
                    target = int(target)
                gates.append(gate)
                targets.append(target)
    return gates, targets

def gates_lookup(gates):
    """
    Maps gate names to their corresponding gate functions.

    Args:
        gates (list): List of gate names (str) or tuples for rotation gates with arguments.

    Returns:
        list: List of gate functions corresponding to the input gate names.
    """
    gate_functions = []
    for gate in gates:
        if isinstance(gate, tuple):
            gate_name, arg = gate   #Rotation gates with arguments
            if gate_name == 'Rotation_x':
                gate_functions.append(rotation_x(float(arg)))
            elif gate_name == 'Rotation_y':
                gate_functions.append(rotation_y(float(arg)))
            elif gate_name == 'Rotation_z':
                gate_functions.append(rotation_z(float(arg)))
            elif gate_name == 'Phase':
                gate_functions.append(phase(float(arg)))
            elif gate_name == 'Cphase':
                gate_functions.append(cphase(float(arg)))
            else:
                print(f"Unknown gate with argument: {gate_name}")
        else:               #Standard gates without arguments
            if gate == 'X':
                gate_functions.append(x())
            elif gate == 'Y':
                gate_functions.append(y())
            elif gate == 'Z':
                gate_functions.append(z())
            elif gate == 'Had':
                gate_functions.append(had())
            elif gate == 'S':
                gate_functions.append(s())
            elif gate == 'T':
                gate_functions.append(t())
            elif gate == 'CNOT':
                gate_functions.append(cnot())
            elif gate == 'CNOT':
                gate_functions.append(cnot_10())
            elif gate == 'SWAP':
                gate_functions.append(swap())
            elif gate == 'CZ':
                gate_functions.append(cz())
            else:
                print(f"Unknown gate: {gate}")
    return gate_functions

def plot_measure_DM(state, shots=1000):
    fig, ax = plt.subplots(figsize=(3, 5))
    basis_states = {'|00>': 0, '|01>': 0, '|10>': 0, '|11>': 0}
    meast = simulate_measurements(state, num_samples=int(shots))
    for k in basis_states:
        k_ = k.replace('|', '').replace('>', '')
        if k_ in meast:
            basis_states[k] = float(meast[k_])/float(shots)
    ax.bar(list(basis_states.keys()), list(basis_states.values()), 0.25)
    ax.set_title(f'Measurement over {shots} shots')
    ax.set_ylabel('Probability')
    plt.show()
    return fig
