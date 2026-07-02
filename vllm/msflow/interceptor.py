# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Communication flow interceptors for MsFlow system.

This module provides monkey-patch wrappers to intercept:
- Stage 1: KV cache reuse (start_load_kv)
- Stage 2: NCCL collective communication (all_reduce)
- Stage 3: P2D KV transfer (send_tensor)

All intercepted flows are wrapped in MsFlow objects and enqueued to RMLQ.

RLI (Resource Load Index) = total_layers - layer_idx
RLI decreases as we progress through layers.
When RLI is low (near end), slack is tight → trigger priority promotion.
"""

import functools
import logging
import time
from typing import Any, Callable

import torch

from vllm.msflow.layer_tracker import LayerContext
from vllm.msflow.msflow import MsFlow, MsFlowStage
from vllm.msflow.rmlq import get_rmlq

logger = logging.getLogger(__name__)

# Store original functions for restoration
_original_functions: dict[str, Callable[..., Any]] = {}


def _compute_tensor_bytes(tensor: Any) -> int:
    """
    Compute the size of a tensor in bytes.
    
    Args:
        tensor: Tensor object
        
    Returns:
        Size in bytes, 0 if not a tensor
    """
    if not hasattr(tensor, "numel") or not hasattr(tensor, "dtype"):
        return 0
    
    dtype_size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int32: 4,
        torch.int64: 8,
    }
    dtype_size = dtype_size_map.get(tensor.dtype, 2)  # Default to fp16
    return tensor.numel() * dtype_size


def intercept_pynccl_all_reduce(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Interceptor for PyNcclCommunicator.all_reduce.
    Generates Stage 2 (COLLECTIVE_COMM) MsFlow.
    
    Stage 2 flows start at P2 priority (higher) since they're synchronous
    and must complete for forward pass to proceed.
    
    Args:
        func: Original all_reduce function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        layer_idx = LayerContext.get_layer()
        total_layers = LayerContext.get_total_layers()
        
        # Compute bytes from input tensor (first tensor arg)
        bytes_size = 0
        for arg in args:
            bytes_size = _compute_tensor_bytes(arg)
            if bytes_size > 0:
                break
        
        flow = MsFlow(
            stage=MsFlowStage.COLLECTIVE_COMM,
            layer_idx=layer_idx,
            bytes=bytes_size,
            total_layers=total_layers,
            timestamp=time.time(),
        )
        
        get_rmlq().enqueue(flow)
        
        # Execute original function immediately (Stage 2 is synchronous)
        return func(self, *args, **kwargs)
    
    return wrapper


def intercept_p2p_send_tensor(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Interceptor for P2pNcclEngine.send_tensor.
    Generates Stage 3 (KV_TRANSFER) MsFlow.
    
    Stage 3 flows start at lowest priority (Defer).
    They get promoted as RLI decreases (slack tightens).
    
    Args:
        func: Original send_tensor function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        tensor_id = args[0]
        tensor = args[1]
        remote_address = args[2]
        
        # Get current layer from context
        layer_idx = LayerContext.get_layer()
        total_layers = LayerContext.get_total_layers()
        
        # Extract layer index from tensor_id if available
        # Format: request_id#layer_name (e.g., req_123#layer_0)
        if "#" in tensor_id:
            layer_name = tensor_id.split("#")[1]
            if "layer_" in layer_name:
                try:
                    layer_idx = int(layer_name.split("layer_")[1])
                except ValueError:
                    pass
        
        # Compute bytes from tensor
        bytes_size = _compute_tensor_bytes(tensor)
        
        # Extract request_id from tensor_id
        request_id = ""
        if "#" in tensor_id:
            request_id = tensor_id.split("#")[0]
        
        # Create Stage 3 MsFlow with send_func for deferred execution
        flow = MsFlow(
            stage=MsFlowStage.KV_TRANSFER,
            layer_idx=layer_idx,
            bytes=bytes_size,
            total_layers=total_layers,
            request_id=request_id,
            tensor_id=tensor_id,
            remote_address=remote_address,
            send_func=func,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
        )
        
        # Enqueue to RMLQ (will be deferred to lowest priority)
        get_rmlq().enqueue(flow)
        
        # Return immediately (async - actual send handled by RMLQ)
        return True
    
    return wrapper


def intercept_start_load_kv(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Interceptor for P2pNcclConnector.start_load_kv.
    Generates Stage 1 (KV_REUSE) MsFlow.
    
    Stage 1 flows start at P3 priority (medium).
    KV reuse is important but can be deferred slightly.
    
    Args:
        func: Original start_load_kv function
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        forward_context = args[0] if args else None
        
        layer_idx = LayerContext.get_layer()
        total_layers = LayerContext.get_total_layers()
        
        # Get request_ids from forward_context
        request_ids = []
        if forward_context is not None:
            if hasattr(forward_context, "attn_metadata"):
                attn_metadata = forward_context.attn_metadata
                if attn_metadata is not None:
                    if hasattr(attn_metadata, "request_ids"):
                        request_ids = attn_metadata.request_ids
                    elif hasattr(attn_metadata, "seq_groups"):
                        # Try alternative access pattern
                        seq_groups = attn_metadata.seq_groups
                        if seq_groups:
                            request_ids = [sg.request_id for sg in seq_groups]
        
        # Generate Stage 1 MsFlow for each request
        for request_id in request_ids:
            flow = MsFlow(
                stage=MsFlowStage.KV_REUSE,
                layer_idx=layer_idx,
                bytes=0,  # Size will be determined later
                total_layers=total_layers,
                request_id=request_id,
                timestamp=time.time(),
            )
            get_rmlq().enqueue(flow)
        
        # Execute original function
        return func(self, *args, **kwargs)
    
    return wrapper


def setup_interceptors() -> None:
    """
    Setup all communication interceptors.
    Monkey-patches the following methods:
    
    Stage 1: P2pNcclConnector.start_load_kv (KV cache reuse)
    Stage 2: PyNcclCommunicator.all_reduce (NCCL collective comm)
    Stage 3: P2pNcclEngine.send_tensor (P2D KV transfer)
    """
    global _original_functions
    
    try:
        # Stage 2: Intercept NCCL all_reduce
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        
        if "PyNcclCommunicator.all_reduce" not in _original_functions:
            _original_functions["PyNcclCommunicator.all_reduce"] = (
                PyNcclCommunicator.all_reduce
            )
            PyNcclCommunicator.all_reduce = intercept_pynccl_all_reduce(
                PyNcclCommunicator.all_reduce
            )
            logger.info("✅ Intercepted Stage 2: PyNcclCommunicator.all_reduce")
        
        # Stage 3: Intercept P2pNcclEngine.send_tensor
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            P2pNcclEngine,
        )
        
        if "P2pNcclEngine.send_tensor" not in _original_functions:
            _original_functions["P2pNcclEngine.send_tensor"] = (
                P2pNcclEngine.send_tensor
            )
            P2pNcclEngine.send_tensor = intercept_p2p_send_tensor(
                P2pNcclEngine.send_tensor
            )
            logger.info("✅ Intercepted Stage 3: P2pNcclEngine.send_tensor")
        
        # Stage 1: Intercept P2pNcclConnector.start_load_kv
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
            P2pNcclConnector,
        )
        
        if "P2pNcclConnector.start_load_kv" not in _original_functions:
            _original_functions["P2pNcclConnector.start_load_kv"] = (
                P2pNcclConnector.start_load_kv
            )
            P2pNcclConnector.start_load_kv = intercept_start_load_kv(
                P2pNcclConnector.start_load_kv
            )
            logger.info("✅ Intercepted Stage 1: P2pNcclConnector.start_load_kv")
        
    except ImportError as e:
        logger.warning(f"⚠️ Failed to setup interceptors: {e}")


def teardown_interceptors() -> None:
    """
    Restore original functions.
    """
    global _original_functions
    
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        
        if "PyNcclCommunicator.all_reduce" in _original_functions:
            PyNcclCommunicator.all_reduce = _original_functions[
                "PyNcclCommunicator.all_reduce"
            ]
            logger.info("✅ Restored PyNcclCommunicator.all_reduce")
        
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            P2pNcclEngine,
        )
        
        if "P2pNcclEngine.send_tensor" in _original_functions:
            P2pNcclEngine.send_tensor = _original_functions[
                "P2pNcclEngine.send_tensor"
            ]
            logger.info("✅ Restored P2pNcclEngine.send_tensor")
        
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
            P2pNcclConnector,
        )
        
        if "P2pNcclConnector.start_load_kv" in _original_functions:
            P2pNcclConnector.start_load_kv = _original_functions[
                "P2pNcclConnector.start_load_kv"
            ]
            logger.info("✅ Restored P2pNcclConnector.start_load_kv")
        
    except ImportError as e:
        logger.warning(f"⚠️ Failed to teardown interceptors: {e}")