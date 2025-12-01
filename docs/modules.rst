Module Index
============

.. toctree::
   :maxdepth: 2

   cubeit
   cubeit.register
   cubeit.gates


Gates Module
============

Universal gate set and additional quantum gates for two-qubit systems.

This module implements:

* The universal gate set: ``{H, S, T, CNOT}``
* Pauli gates ``X, Y, Z``
* Parameterised rotation and phase gates
* Two-qubit entangling gates (SWAP, CZ, CPHASE)

The universal gate set can approximate any unitary operation to arbitrary precision.

.. automodule:: gates
   :members:
   :undoc-members:
   :show-inheritance:


Universal Gate Set
------------------

.. autofunction:: gates.had
.. autofunction:: gates.s
.. autofunction:: gates.t
.. autofunction:: gates.cnot
.. autofunction:: gates.cnot_10


Pauli Gates
-----------

.. autofunction:: gates.x
.. autofunction:: gates.y
.. autofunction:: gates.z


Parameterized Gates
-------------------

.. autofunction:: gates.phase
.. autofunction:: gates.rotation_x
.. autofunction:: gates.rotation_y
.. autofunction:: gates.rotation_z


Two-Qubit Gates
---------------

.. autofunction:: gates.swap
.. autofunction:: gates.cz
.. autofunction:: gates.cphase


Noise Module
============

This module provides a collection of quantum noise models that act on
density matrices. These noise channels are commonly used in simulations
of open quantum systems and noisy quantum devices.

The functions in this module operate directly on NumPy arrays
representing density matrices.

.. contents::
   :local:
   :depth: 2


Depolarising Noise
------------------

.. function:: depolarising_noise(rho, p)

   Apply depolarising noise to a density matrix.

   :param rho: Density matrix of the quantum state.
   :type rho: numpy.ndarray
   :param p: Probability of depolarisation (0 ≤ p ≤ 1).
   :type p: float

   :returns: Density matrix after applying depolarising noise.
   :rtype: numpy.ndarray


Dephasing Noise
---------------

.. function:: dephasing_noise(rho, p)

   Apply dephasing noise to a density matrix.

   :param rho: Density matrix of the quantum state.
   :type rho: numpy.ndarray
   :param p: Probability of dephasing (0 ≤ p ≤ 1).
   :type p: float

   :returns: Density matrix after applying dephasing noise.
   :rtype: numpy.ndarray


Amplitude Damping Noise
-----------------------

.. function:: amplitude_damping_noise(rho, gamma)

   Apply amplitude damping noise to a density matrix.

   :param rho: Density matrix of the quantum state.
   :type rho: numpy.ndarray
   :param gamma: Damping probability (0 ≤ γ ≤ 1).
   :type gamma: float

   :returns: Density matrix after applying amplitude damping noise.
   :rtype: numpy.ndarray


Bit-Flip Noise
--------------

.. function:: bit_flip_noise(rho, p)

   Apply bit-flip noise to a density matrix.

   :param rho: Density matrix of the quantum state.
   :type rho: numpy.ndarray
   :param p: Probability of bit-flip (0 ≤ p ≤ 1).
   :type p: float

   :returns: Density matrix after applying bit-flip noise.
   :rtype: numpy.ndarray


Density Matrix Module
=====================

This module provides functionality for constructing, manipulating, and
measuring density matrices (DMs) within the ``cubeit`` package. It
supports single– and two–qubit systems, ideal and noisy measurement
models, and application of quantum gates with optional noise channels.


Functions
---------

create_density_matrix
~~~~~~~~~~~~~~~~~~~~~

.. function:: create_density_matrix(state_vector)

   Create a density matrix from a state vector.

   :param np.ndarray state_vector: State vector of the quantum state.
   :returns: Density matrix :math:`|\psi\rangle \langle \psi|`.
   :rtype: np.ndarray


DM_measurement_ideal
~~~~~~~~~~~~~~~~~~~~

.. function:: DM_measurement_ideal(rho, basis='Z')

   Measure a density matrix in a given basis.

   :param np.ndarray rho: Density matrix of the quantum state.
   :param str basis: Measurement basis (one of ``'X'``, ``'Y'``, ``'Z'``).
   :returns: Dictionary mapping bitstring outcomes to probabilities.
   :rtype: dict


DM_measurement_shots
~~~~~~~~~~~~~~~~~~~~

.. function:: DM_measurement_shots(rho, shots, basis='Z')

   Simulate measurement with a finite number of shots.

   :param np.ndarray rho: Density matrix of the quantum state.
   :param np.ndarray shots: Numbers of measurement shots.
   :returns: 
      - list of dicts – empirical outcome frequencies  
      - dict – ideal probabilities
   :rtype: (list, dict)


DM_measurement_shots_noise
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: DM_measurement_shots_noise(rho, shots, basis='Z', p_flip=None)

   Simulate measurement with readout noise.

   :param dict p_flip: Bit-flip error rates (``p01`` and ``p10``).
   :returns:
      - list of dicts – noisy outcome frequencies  
      - dict – ideal probabilities
   :rtype: (list, dict)


Classes
-------

DensityMatrix1Qubit
~~~~~~~~~~~~~~~~~~~

.. class:: DensityMatrix1Qubit(rho)

   Represents a one–qubit density matrix.

   .. attribute:: rho
   .. attribute:: history

   .. method:: apply_gate(gate, target)
   .. method:: apply_sequence(gates, targets)
   .. method:: apply_sequence_noise(gates, targets, noise_channels)
   .. method:: measure_ideal(basis='Z')
   .. method:: measure_shots(shots, basis='Z', pdict={'p01':0.02,'p10':0.05})


DensityMatrix2Qubit
~~~~~~~~~~~~~~~~~~~

.. class:: DensityMatrix2Qubit(rho)

   Represents a two–qubit density matrix.

   .. attribute:: rho
   .. attribute:: history

   .. method:: apply_single_qubit_gate(gate, target)
   .. method:: apply_sequence(gates, targets)
   .. method:: apply_sequence_noise(gates, targets, noise_channels)
   .. method:: partial_trace(keep)
   .. method:: measure_ideal(basis='Z')
   .. method:: measure_shots(shots, basis='Z', pdict={'p01':0.02,'p10':0.05})
   .. method:: clean(tol=1e-12)