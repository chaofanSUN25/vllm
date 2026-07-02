# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
RMLQ (Reverse Multi-Level Queue) - Communication flow scheduler.

This implements a K-level priority queue where:
- P1 is highest priority, PK is lowest priority
- Flows start at low priority (Defer)
- Flows get promoted to higher priority when slack drops below threshold (Promote)
- Priority only increases, never decreases (avoid priority oscillation)
- Promotions only happen at layer boundaries to avoid packet reordering

The goal is to maximize TTFT (Time To First Token) SLO compliance.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from vllm.msflow.layer_tracker import LayerContext
from vllm.msflow.msflow import MsFlow, MsFlowStage

logger = logging.getLogger(__name__)

# Global RMLQ instance
_rmlq_instance: Optional["RMLQ"] = None


class RMLQ:
    """
    Reverse Multi-Level Queue scheduler.
    
    Key characteristics:
    1. K discrete priority levels (P1 = highest, PK = lowest)
    2. Stage 2 (collective comm) gets higher initial priority (P2)
    3. Stage 3 (P2D transfer) starts at lowest priority (Defer to PK)
    4. Stage 1 (KV reuse) starts at medium priority (P3)
    5. Promotion only occurs at layer boundaries
    6. Priority only increases, never decreases
    7. Promotion triggered when RLI drops below threshold
    """

    def __init__(self, num_priorities: int = 5, total_layers: int = 32) -> None:
        """
        Initialize RMLQ.
        
        Args:
            num_priorities: Number of priority levels (K)
            total_layers: Total number of Transformer layers
        """
        self.num_priorities = num_priorities
        self.total_layers = total_layers
        
        # Priority queues: P1 is highest (index 0), PK is lowest (index K-1)
        self.queues: Dict[int, deque[MsFlow]] = {
            i: deque() for i in range(1, num_priorities + 1)
        }
        
        # Statistics
        self.stats: Dict[str, int] = {
            "total_enqueued": 0,
            "promotions": 0,
        }
        for i in range(1, num_priorities + 1):
            self.stats[f"p{i}_size"] = 0
            self.stats[f"p{i}_processed"] = 0
        
        # Threading
        self.lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Promotion thresholds based on RLI (remaining load index)
        # When RLI drops below threshold, promote to next higher priority
        # RLI = total_layers - layer_idx, decreases as we go deeper
        self.promotion_thresholds = {
            5: 24,  # RLI <= 24: P5 -> P4 (entered layer 8+)
            4: 16,  # RLI <= 16: P4 -> P3 (entered layer 16+)
            3: 8,   # RLI <= 8: P3 -> P2 (entered layer 24+)
            2: 4,   # RLI <= 4: P2 -> P1 (entered layer 28+)
        }
        
        # Default deadline for TTFT (500ms - typical SLO target)
        self.default_deadline_ms = 500.0

    def start(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="RMLQ-Scheduler"
        )
        self._scheduler_thread.start()
        logger.info(f"✅ RMLQ scheduler started with {self.num_priorities} priority levels")

    def stop(self) -> None:
        """Stop the scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("⏹️ RMLQ scheduler stopped")

    def enqueue(self, flow: MsFlow) -> None:
        """
        Enqueue a MsFlow into the appropriate priority queue.
        
        Initial priority assignment (Defer):
        - Stage 1 (KV reuse): P3 (medium)
        - Stage 2 (collective comm): P2 (higher)
        - Stage 3 (P2D transfer): PK (lowest - defer until slack is tight)
        
        Args:
            flow: MsFlow object to enqueue
        """
        with self.lock:
            # Set initial priority based on stage (Defer)
            if flow.is_collective():
                # Stage 2: Collective comm - start at P2 (higher priority)
                flow.priority = 2
            elif flow.is_kv_transfer():
                # Stage 3: P2D transfer - start at lowest priority (Defer)
                flow.priority = self.num_priorities
            elif flow.is_kv_reuse():
                # Stage 1: KV reuse - start at P3 (medium)
                flow.priority = 3
            
            # Set timestamp if not already set
            if flow.timestamp == 0.0:
                flow.timestamp = time.time()
            
            self.queues[flow.priority].append(flow)
            self.stats["total_enqueued"] += 1
            self._update_stats()
            
        logger.debug(f"📥 Enqueued flow: stage={flow.stage}, layer={flow.layer_idx}, "
                    f"rli={flow.rli:.1f}, priority=P{flow.priority}, bytes={flow.bytes}")

    def _update_stats(self) -> None:
        """Update queue size statistics."""
        for i in range(1, self.num_priorities + 1):
            self.stats[f"p{i}_size"] = len(self.queues[i])

    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop.
        Process flows from highest priority to lowest.
        Always process P1 completely before moving to P2, etc.
        """
        while self._running:
            processed = False
            
            # Process from highest priority to lowest
            for priority in range(1, self.num_priorities + 1):
                with self.lock:
                    while self.queues[priority]:
                        flow = self.queues[priority].popleft()
                        self._process_flow(flow)
                        self.stats[f"p{priority}_processed"] += 1
                        self._update_stats()
                        processed = True
            
            if not processed:
                time.sleep(0.001)  # Yield if no work

    def _process_flow(self, flow: MsFlow) -> None:
        """
        Process a single MsFlow.
        
        Args:
            flow: MsFlow object to process
        """
        try:
            if flow.is_collective():
                # Stage 2: Collective comm - execute original function
                if flow.send_func:
                    flow.send_func(*flow.args, **flow.kwargs)
                logger.debug(f"🔄 Processed collective flow: layer={flow.layer_idx}, "
                            f"rli={flow.rli:.1f}, priority=P{flow.priority}")
            
            elif flow.is_kv_transfer():
                # Stage 3: P2D transfer - execute send function
                if flow.send_func:
                    flow.send_func(*flow.args, **flow.kwargs)
                logger.debug(f"📤 Processed KV transfer: layer={flow.layer_idx}, "
                            f"rli={flow.rli:.1f}, priority=P{flow.priority}, "
                            f"bytes={flow.bytes}")
            
            elif flow.is_kv_reuse():
                # Stage 1: KV reuse - just log (no action needed)
                logger.debug(f"♻️ Processed KV reuse: layer={flow.layer_idx}, "
                            f"rli={flow.rli:.1f}, priority=P{flow.priority}")
        except Exception as e:
            logger.error(f"❌ Failed to process flow: {e}")

    def promote_flows_at_layer_boundary(self, current_layer: int) -> None:
        """
        Promote flows at layer boundary based on RLI.
        This is the key promotion point - only at layer boundaries.
        
        Promotion logic:
        - Calculate current RLI = total_layers - current_layer
        - Check each queue from lowest to highest priority
        - If flow's RLI <= threshold, promote to next higher priority
        - Priority only increases, never decreases
        
        Args:
            current_layer: Current layer index being entered
        """
        with self.lock:
            current_rli = self.total_layers - current_layer
            
            # Check queues from lowest to highest priority
            # This ensures lower priority flows get promoted first
            for priority in range(self.num_priorities, 1, -1):
                threshold = self.promotion_thresholds.get(priority)
                if threshold is None:
                    continue
                
                # Collect flows that need promotion
                flows_to_promote: List[MsFlow] = []
                for flow in self.queues[priority]:
                    # RLI is already computed at enqueue time, but we use
                    # current RLI for promotion decision since we're at
                    # a layer boundary
                    if current_rli <= threshold:
                        flows_to_promote.append(flow)
                
                # Promote flows to next higher priority
                for flow in flows_to_promote:
                    old_priority = flow.priority
                    if flow.promote(priority - 1):
                        self.queues[old_priority].remove(flow)
                        self.queues[priority - 1].append(flow)
                        self.stats["promotions"] += 1
                        logger.debug(f"⬆️ Promoted flow: stage={flow.stage}, "
                                    f"layer={current_layer}, "
                                    f"P{old_priority}->P{priority-1}, "
                                    f"rli={current_rli:.1f}")
            
            self._update_stats()

    def compute_slack(self, flow: MsFlow, current_time_ms: float) -> float:
        """
        Compute slack for a flow based on RLI.
        
        Args:
            flow: MsFlow object
            current_time_ms: Current time in milliseconds
            
        Returns:
            Slack time in milliseconds
        """
        return flow.get_slack(
            deadline_ms=self.default_deadline_ms,
            current_time_ms=current_time_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics.
        
        Returns:
            Dictionary containing queue sizes and processed counts
        """
        with self.lock:
            return dict(self.stats)

    def wait_for_empty(self, timeout: float = 10.0) -> bool:
        """
        Wait for all queues to become empty.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if all queues emptied, False on timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                total_size = sum(len(q) for q in self.queues.values())
            if total_size == 0:
                return True
            time.sleep(0.01)
        return False


def init_rmlq(num_priorities: int = 5, total_layers: int = 32) -> "RMLQ":
    """
    Initialize global RMLQ instance.
    
    Args:
        num_priorities: Number of priority levels (K)
        total_layers: Total number of Transformer layers
        
    Returns:
        RMLQ instance
    """
    global _rmlq_instance
    if _rmlq_instance is None:
        _rmlq_instance = RMLQ(num_priorities, total_layers)
        _rmlq_instance.start()
    return _rmlq_instance


def get_rmlq() -> "RMLQ":
    """
    Get global RMLQ instance.
    
    Returns:
        RMLQ instance
    
    Raises:
        RuntimeError: If RMLQ has not been initialized
    """
    if _rmlq_instance is None:
        raise RuntimeError("RMLQ not initialized. Call init_rmlq() first.")
    return _rmlq_instance