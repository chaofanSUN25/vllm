# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MsFlow module - Communication flow interception and scheduling.

Key components:
- MsFlow: Represents communication flows (Stage 1/2/3)
- LayerContext: Tracks current Transformer layer during forward pass
- RMLQ: Reverse Multi-Level Queue for flow scheduling
- Interceptors: Monkey-patch wrappers for communication functions

Three Stages of Communication Flow:
- Stage 1 (KV_REUSE=1): KV cache reuse (decode loads prefill's KV cache)
- Stage 2 (COLLECTIVE_COMM=2): NCCL collective communication (TP all-reduce)
- Stage 3 (KV_TRANSFER=3): P2D transfer (prefill sends KV cache to decode)

RLI (Resource Load Index) = total_layers - layer_idx
RLI decreases as layers progress, representing remaining load.
When RLI is low (near end), slack is tight → trigger priority promotion.

RMLQ (Reverse Multi-Level Queue) Scheduling:
- K discrete priority levels (P1=highest, PK=lowest)
- Stage 1 (KV reuse): P3 initial priority (medium)
- Stage 2 (collective comm): P2 initial priority (higher)
- Stage 3 (P2D transfer): PK initial priority (Defer - lowest)
- Promotion only at layer boundaries (avoid packet reordering)
- Promotion triggered when RLI drops below threshold
- Priority only increases, never decreases (avoid oscillation)

Promotion Thresholds (for K=5):
- P5 → P4: RLI ≤ 24 (entered layer 8+)
- P4 → P3: RLI ≤ 16 (entered layer 16+)
- P3 → P2: RLI ≤ 8 (entered layer 24+)
- P2 → P1: RLI ≤ 4 (entered layer 28+)

Goal: Maximize TTFT (Time To First Token) SLO compliance.
"""

from .layer_tracker import LayerContext, track_layer
from .msflow import MsFlow, MsFlowStage
from .rmlq import RMLQ, get_rmlq, init_rmlq
from .interceptor import setup_interceptors, teardown_interceptors

__all__ = [
    "LayerContext",
    "MsFlow",
    "MsFlowStage",
    "RMLQ",
    "get_rmlq",
    "init_rmlq",
    "setup_interceptors",
    "teardown_interceptors",
    "track_layer",
]