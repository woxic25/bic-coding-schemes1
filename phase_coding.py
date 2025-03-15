"""
Module for Phase Coding implementation.

This module implements the phase coding scheme as described in:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.
and as proposed by Kim et al. (2018c).

Phase Coding Overview:
----------------------
1. Each input pixel (0–255) is converted into its 8-bit binary representation.
2. Each bit in the binary representation corresponds to one phase (total of 8 phases).
   A bit of "1" indicates that a spike occurs in that phase.
3. The spike weight changes with time (phase) periodically. The weight for a spike at phase t is given by:
       w_s(t) = 2^-[1 + mod(t-1, 8)]
   For a single 8-phase cycle, this yields weights:
       [2^-1, 2^-2, ..., 2^-8] (i.e., 0.5, 0.25, 0.125, …, ~0.0039)
4. During decoding, these weighted spikes are summed (along with synaptic weights) to compute the post-synaptic potential.

Usage:
    from phase_coding import PhaseCoding
    coder = PhaseCoding()
    spike_train = coder.encode(image)
    phase_weights = coder.get_phase_weights()
"""

import numpy as np
import matplotlib.pyplot as plt


class PhaseCoding:
    """
    Implements phase coding for a 2D image.

    Each pixel is converted into its 8-bit binary representation. Each bit (phase) is
    interpreted as a spike (True if bit==1, else False). A helper function provides
    the weights for each phase according to:
         w_s(t) = 2^-[1 + mod(t-1, 8)]
    """

    def __init__(self, num_phases=8):
        """
        Initialize PhaseCoding.

        Parameters:
            num_phases (int): Number of phases to use (default 8, corresponding to 8-bit representation).
        """
        self.num_phases = num_phases

    def encode(self, image):
        """
        Encode an input image into a phase-coded spike train.

        The input image is expected to be a 2D NumPy array with pixel values in the range [0, 255].
        Each pixel is converted into its 8-bit binary representation (most significant bit first).

        Parameters:
            image (numpy.ndarray): 2D array representing the input image.

        Returns:
            numpy.ndarray: A boolean 3D array of shape (height, width, num_phases), where True indicates a spike.
        """
        # Ensure image is a NumPy array of type uint8
        image = image.astype(np.uint8)
        height, width = image.shape

        # Prepare an output spike train: shape (height, width, num_phases)
        spike_train = np.zeros((height, width, self.num_phases), dtype=bool)

        # For each pixel, convert to an 8-bit binary string and then to a boolean array
        # We use format(pixel, '08b') to get an 8-character string with leading zeros.
        for i in range(height):
            for j in range(width):
                # Convert pixel value to 8-bit binary string (MSB first)
                binary_str = format(image[i, j], "08b")
                # Convert each character ('0' or '1') to boolean (1 -> True, 0 -> False)
                bits = np.array([char == "1" for char in binary_str], dtype=bool)
                spike_train[i, j, :] = bits

        return spike_train

    def get_phase_weights(self):
        """
        Compute the weights for each phase according to:
            w_s(t) = 2^-[1 + mod(t-1, num_phases)]
        For a single cycle with t = 1, 2, ..., num_phases, this simplifies to:
            weights = [2^-1, 2^-2, ..., 2^-(num_phases)]

        Returns:
            numpy.ndarray: A 1D array of phase weights.
        """
        # For phases 1 to num_phases, compute 2^-(phase)
        weights = np.array([2 ** (-(phase)) for phase in range(1, self.num_phases + 1)])
        return weights


# Example usage / quick test
if __name__ == "__main__":
    # Create a dummy image (e.g., 28x28) with random values between 0 and 255
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Initialize the PhaseCoding instance
    coder = PhaseCoding()

    # Encode the dummy image into a spike train
    spike_train = coder.encode(dummy_image)

    # Retrieve phase weights
    phase_weights = coder.get_phase_weights()
    print("Phase weights:", phase_weights)

    # For demonstration, visualize the spike train for a single pixel (e.g., at row 14, column 14)
    pixel_row, pixel_col = 14, 14
    pixel_spikes = spike_train[pixel_row, pixel_col, :].astype(int)

    plt.figure()
    plt.stem(np.arange(1, coder.num_phases + 1), pixel_spikes, basefmt=" ")
    plt.xlabel("Phase")
    plt.ylabel("Spike (1: spike, 0: no spike)")
    plt.title(f"Phase-Coded Spike Pattern for Pixel ({pixel_row}, {pixel_col})")
    plt.show()
