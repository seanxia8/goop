"""ADC digitization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DigitizationConfig:
    """ADC digitization parameters (post-convolution).

    Applied after convolution: adds a pedestal offset, rounds to integers,
    and clamps to the ADC bit range [0, 2^n_bits - 1].
    """

    n_bits: int = 14          # 14-bit ADC → [0, 16383]
    pedestal: float = 1500.0  # baseline offset in ADC counts


def digitize(data: torch.Tensor, pedestal: float, n_bits: int) -> torch.Tensor:
    """Add pedestal, round to integer, clamp to ADC range.

    Returns float32 tensor with integer-valued entries in [0, 2^n_bits - 1].
    """
    adc_max = (1 << n_bits) - 1
    return (data + pedestal).round().clamp(0, adc_max)


def digitize_ste(data: torch.Tensor, pedestal: float, n_bits: int) -> torch.Tensor:
    """Straight-through-estimator (STE) digitization.

    Forward: same as ``digitize`` — ``(data + pedestal).round().clamp(0, max)``.
    Backward: gradient passes through to ``data`` as identity (the round and
    clamp are bypassed in the autograd graph).
    """
    x = data + pedestal
    x_q = digitize(data, pedestal, n_bits)  # round + clamp, no_grad
    return x_q + (x - x.detach())
