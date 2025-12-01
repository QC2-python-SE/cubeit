"""Example usage of the cubeit quantum register API."""

from cubeit import quantumregister, get_state, measure


def main() -> None:
    # Create a 4-qubit register initialised in |0000⟩
    qr = quantumregister(4)

    # Apply a small circuit
    qr.h(0)          # Put qubit 0 into superposition
    qr.cnot(0, 1)    # Entangle qubit 0 and 1
    qr.rx(2, 0.5)    # Rotate qubit 2 around X by 0.5 radians
    qr.cz(1, 3)      # Controlled-Z between qubit 1 and 3

    print("State before measurement:")
    get_state(qr)    # Pretty-print the state vector

    print("Measurement result:")
    outcome = measure(qr)  # Collapses the register and prints outcome
    print("Collapsed state:", outcome)


if __name__ == "__main__":
    main()



