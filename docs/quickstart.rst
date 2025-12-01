Quick Start
===========

Launching the CubeIt graphical interface
--------------------------------------

.. code-block:: bash

   python run_gui.python

Creating a Quantum Register
---------------------------

Create a quantum register with any number of qubits:

.. code-block:: python

   from cubeit import quantumregister, get_state, measure

   # Create a 4-qubit register (|0000⟩)
   qr = quantumregister(4)

Building Circuits
-----------------

Use fluent helper methods to build quantum circuits:

.. code-block:: python

   qr.had(0)          # Hadamard on qubit 0
   qr.cnot(0, 1)    # Entangle qubit 0 and 1
   qr.rx(2, 0.5)    # Rotate qubit 2 around X by 0.5 radians
   qr.cz(1, 3)      # Controlled-Z between qubit 1 and 3

Inspecting States
-----------------

Inspect the statevector (pretty printed):

.. code-block:: python

   get_state(qr)

Measuring
---------

Measure the register – collapses the state and returns a bitstring:

.. code-block:: python

   result = measure(qr)
   print("Measurement:", result)

Gate Helpers
------------

Available gats:

+--------------------------+----------------------------------+
| Helper                   | Description                      |
+==========================+==================================+
| ``qr.had(i)``              | Hadamard on qubit *i*            |
+--------------------------+----------------------------------+
| ``qr.x(i)``, ``qr.y(i)``,| Pauli gates                      |
| ``qr.z(i)``              |                                  |
+--------------------------+----------------------------------+
| ``qr.s(i)``, ``qr.t(i)`` | Phase / π/8 gates                |
+--------------------------+----------------------------------+
| ``qr.rx(i, θ)``,         | Rotations                        |
| ``qr.ry(i, θ)``,         |                                  |
| ``qr.rz(i, θ)``          |                                  |
+--------------------------+----------------------------------+
| ``qr.cnot(control, target)`` | Controlled-NOT              |
+--------------------------+----------------------------------+
| ``qr.cz(control, target)`` | Controlled-Z                   |
+--------------------------+----------------------------------+
| ``qr.cphase(control, target, φ)`` | Controlled-phase        |
+--------------------------+----------------------------------+
| ``qr.swap(a, b)``        | Swap two qubits                  |
+--------------------------+----------------------------------+

Functional Style
----------------

Prefer something functional? Import the factories directly:

.. code-block:: python

   from cubeit import h, cnot, quantumregister

   qr = quantumregister(2)
   qr.apply_single_qubit_gate(h(), 0)
   qr.apply_two_qubit_gate(cnot(), 0, 1)

Measurement & Probabilities
---------------------------

.. code-block:: python

   from cubeit import quantumregister, get_state, measure
   from cubeit.visualization import print_probabilities

   qr = quantumregister(2).had(0).cnot(0, 1)

   print_probabilities(qr)
   # Measurement Probabilities:
   #   |00⟩: 0.5000 (50.00%)
   #   |11⟩: 0.5000 (50.00%)

   measure(qr)  # collapses the register and prints the classical outcome

Bell States & Visualisation
---------------------------

.. code-block:: python

   from cubeit.visualization import create_bell_state, print_state, print_measurement_stats

   bell = create_bell_state("phi_plus")
   print_state(bell)                 # 0.707|00⟩ + 0.707|11⟩
   print_measurement_stats(bell)     # Monte-Carlo sampling

