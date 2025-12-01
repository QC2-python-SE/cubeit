
import pytest
import numpy as np
from cubeit import (
    QuantumState,
    quantumregister,
    get_state,
    measure,
    h, s, t, x, y, z,
    cnot, cnot_10, swap, cz,
)
from cubeit.visualisation import plot_bloch_sphere, plot_circuit, simulate_measurements

from cubeit.DMs import DensityMatrix2Qubit as DM2
from cubeit.DMs import DensityMatrix1Qubit as DM1

import os

#Import plotting and visualisation libraries

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual, Text, SelectMultiple, Layout, VBox, HBox, Button, Dropdown, Output, Label, HTML, GridBox, FloatSlider
from IPython.display import display
import datetime


#Function to apply gates - replace this with a more complete function from register or DMs etc.
def apply_gate(state, gate, qubits):
    if gate == 'Had':
        state.had(qubits)
    elif gate == 'S':
        state.s(qubits)
    elif gate == 'T':
        state.t(qubits)
    elif gate == 'X':
        state.x(qubits)
    elif gate == 'Y':
        state.y(qubits)
    elif gate == 'Z':
        state.z(qubits)
    elif gate == 'CNOT':
        state.cnot(qubits[0], qubits[1])
    elif gate == 'SWAP':
        state.swap(qubits[0], qubits[1])
    elif gate == 'CZ':
        state.cz(qubits[0], qubits[1])
    return state

#Function to run gates from string input
def run_gates(gate_string):
    global state
    commands = gate_string.split(";")
    for cmd in commands:
        if len(cmd) > 1:
            cmd = cmd.strip()
            if not cmd or "(" not in cmd or not cmd.endswith(")"):
                print(f"Invalid format: {cmd}")
                continue
            name, args = cmd[:-1].split("(", 1)
            name = name.strip()
            args = args.strip()
            try:
                qubits = [int(q) for q in args.split(",")] if "," in args else int(args)
            except:
                print(f"Invalid qubit input: {cmd}")
                continue
            state = apply_gate(state, name, qubits)
            print(f"Applied {name} on qubit {qubits}.")

def plot_measure(state, shots=1000):
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
    
    ax.legend(frameon=0)
    ax.set_title(f'Measurement over {meas_shots} shots')
    ax.set_ylabel('Probability')
    plt.show()
    return fig
