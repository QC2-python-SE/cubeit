
import pytest
import numpy as np
from cubeit import (
    QuantumState,
    quantumregister,
    get_state,
    measure,
    had, s, t, x, y, z,
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
