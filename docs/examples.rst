Examples
========

Creating a Bell State
---------------------

.. code-block:: python

   from cubeit import quantumregister

   qr = quantumregister(2)
   qr.had(0).cnot(0, 1)
   print(qr)
   # QuantumRegister(2 qubits):
   #   0.707|00⟩ + 0.707|11⟩

Measurement Statistics
----------------------

.. code-block:: python

   from cubeit import quantumregister
   from cubeit.visualisation import print_measurement_stats

   qr = quantumregister(2).had(0).cnot(0, 1)

   print_measurement_stats(qr, num_samples=1000)

GHZ State
---------

.. code-block:: python

   from cubeit import quantumregister

   # Create a 3-qubit GHZ state
   qr = quantumregister(3)
   qr.had(0)
   qr.cnot(0, 1)
   qr.cnot(0, 2)
   get_state(qr)

