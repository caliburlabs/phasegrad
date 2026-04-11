"""Phase-Gradient Duality: backpropagation-free learning in coupled oscillator networks.

In a Kuramoto oscillator network at equilibrium, the physical phase response
to weak output clamping IS the gradient of the loss with respect to the
natural frequencies. This enables a learning rule that requires no separate
backward pass — the gradient is computed by physics.

    lim_{β→0} (θ^β - θ*) / β = -∂L/∂ω
"""

from phasegrad.kuramoto import KuramotoNetwork
from phasegrad.gradient import two_phase_gradient, analytical_gradient, finite_difference_gradient
from phasegrad.training import train
from phasegrad.verification import run_verification
from phasegrad.seeding import spectral_seed

__version__ = "0.1.0"
