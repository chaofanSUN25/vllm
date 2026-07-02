# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MsFlow module - Communication flow tracking and scheduling.

MsFlow objects represent communication flows in vLLM's disaggregated mode:
- Stage 1: KV cache reuse
- Stage 2: Collective communication (TP/EP all-reduce)
- Stage 3: P2D transfer (prefill to decode)

RLI (Resource Load Index) = total_layers - layer_idx
RLI decreases as layers progress, representing remaining load.
When RLI is low, slack is tight, triggering priority promotion.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class MsFlowStage:
    """MsFlow communication stages."""
    KV_REUSE = 1
    COLLECTIVE_COMM = 2
    KV_TRANSFER = 3


@dataclass
class MsFlow:
    """
    Communication flow object for tracking and scheduling.
    
    Attributes:
        stage: Communication stage (1=KV reuse, 2=collective, 3=P2D transfer)
        layer_idx: Current Transformer layer index
        bytes: Size of the data being transferred
        rli: Resource Load Index = total_layers - layer_idx
        total_layers: Total number of layers in the model
        request_id: Optional request identifier
        tensor_id: Optional tensor identifier
        remote_address: Remote address for P2D transfers
        send_func: Original send function for Stage 3
        args: Arguments for send_func
        kwargs: Keyword arguments for send_func
        priority: Current priority level (P1=highest, PK=lowest)
        timestamp: Creation timestamp
    """
    stage: int
    layer_idx: int
    bytes: int
    
    # Derived attributes
    total_layers: int = 32
    rli: float = field(init=False)
    
    # Optional attributes
    request_id: str = ""
    tensor_id: str = ""
    remote_address: str = ""
    send_func: Optional[Callable[..., Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Scheduling attributes
    priority: int = field(default=5)  # Default to lowest priority
    timestamp: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Compute RLI after initialization."""
        self.rli = self.total_layers - self.layer_idx

    def is_collective(self) -> bool:
        """Check if this is a Stage 2 collective communication flow."""
        return self.stage == MsFlowStage.COLLECTIVE_COMM

    def is_kv_transfer(self) -> bool:
        """Check if this is a Stage 3 KV transfer flow."""
        return self.stage == MsFlowStage.KV_TRANSFER

    def is_kv_reuse(self) -> bool:
        """Check if this is a Stage 1 KV reuse flow."""
        return self.stage == MsFlowStage.KV_REUSE

    def promote(self, new_priority: int) -> bool:
        """
        Promote to a higher priority. Priority only increases, never decreases.
        
        Args:
            new_priority: New priority level (lower number = higher priority)
            
        Returns:
            True if promotion occurred, False if already at or above new priority
        """
        if new_priority < self.priority:
            self.priority = new_priority
            return True
        return False

    def get_slack(self, deadline_ms: float, current_time_ms: float) -> float:
        """
        Compute slack time based on RLI.
        Slack = deadline - (current_time + estimated_remaining_time)
        
        Args:
            deadline_ms: Deadline in milliseconds
            current_time_ms: Current time in milliseconds
            
        Returns:
            Slack time in milliseconds. Negative value means already missed.
        """
        # Estimate remaining time based on RLI (higher RLI = more work left)
        # This is a simplified estimation - can be improved with profiling
        estimated_remaining_ms = self.rli * 1.0  # Assume ~1ms per layer
        return deadline_ms - (current_time_ms + estimated_remaining_ms)