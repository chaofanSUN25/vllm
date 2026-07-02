# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LayerContext module - Track current Transformer layer during forward pass.

This provides thread-local storage for tracking the current layer index,
which is essential for computing RLI (Resource Load Index) = total_layers - layer_idx.

RLI decreases as we progress through layers, representing remaining load.
When RLI is low, slack is tight, triggering priority promotion in RMLQ.

Key feature: At each layer boundary, trigger RMLQ promotion based on RLI.
"""

import contextlib
import logging
import threading

logger = logging.getLogger(__name__)


class LayerContext:
    """
    Thread-local context for tracking current Transformer layer.
    
    Attributes:
        _thread_local: Thread-local storage for layer info
    """
    
    _thread_local = threading.local()

    @classmethod
    def set_layer(cls, layer_idx: int) -> None:
        """
        Set current layer index.
        
        Args:
            layer_idx: Current layer index (-1 means no layer active)
        """
        cls._thread_local.layer_idx = layer_idx

    @classmethod
    def get_layer(cls) -> int:
        """
        Get current layer index.
        
        Returns:
            Current layer index, -1 if not set
        """
        return getattr(cls._thread_local, "layer_idx", -1)

    @classmethod
    def set_total_layers(cls, total_layers: int) -> None:
        """
        Set total number of layers in the model.
        
        Args:
            total_layers: Total number of layers
        """
        cls._thread_local.total_layers = total_layers

    @classmethod
    def get_total_layers(cls) -> int:
        """
        Get total number of layers.
        
        Returns:
            Total layers, 0 if not set
        """
        return getattr(cls._thread_local, "total_layers", 0)

    @classmethod
    def compute_rli(cls) -> float:
        """
        Compute RLI (Resource Load Index).
        
        RLI = total_layers - layer_idx
        RLI decreases as we progress through layers.
        When RLI is low (near end), slack is tight.
        
        Returns:
            RLI value, 0.0 if not enough info
        """
        layer_idx = cls.get_layer()
        total_layers = cls.get_total_layers()
        if total_layers > 0 and layer_idx >= 0:
            return total_layers - layer_idx
        return 0.0


@contextlib.contextmanager
def track_layer(layer_idx: int, total_layers: int = 0) -> None:
    """
    Context manager to track current layer.
    
    At each layer boundary, triggers RMLQ promotion based on RLI.
    
    Args:
        layer_idx: Current layer index
        total_layers: Total number of layers (optional)
    """
    # Set layer context
    LayerContext.set_layer(layer_idx)
    if total_layers > 0:
        LayerContext.set_total_layers(total_layers)
    
    # Trigger promotion at layer boundary (before executing layer)
    # This ensures flows get promoted based on current RLI
    try:
        _trigger_rmlq_promotion(layer_idx, total_layers)
        yield
    finally:
        LayerContext.set_layer(-1)


def _trigger_rmlq_promotion(layer_idx: int, total_layers: int) -> None:
    """
    Trigger RMLQ promotion at layer boundary.
    
    This is called when entering a new layer.
    Flows in the queue are promoted based on current RLI.
    
    Args:
        layer_idx: Current layer index
        total_layers: Total number of layers
    """
    try:
        from vllm.msflow.rmlq import get_rmlq
        
        rmlq = get_rmlq()
        rmlq.promote_flows_at_layer_boundary(layer_idx)
        
        rli = total_layers - layer_idx
        logger.debug(f"📍 Layer boundary: layer={layer_idx}, RLI={rli:.1f}")
    except (ImportError, RuntimeError):
        # RMLQ not initialized - skip promotion
        pass