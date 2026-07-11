# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import logging
import random
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class LayerContext:
    """Thread-local context for layer-level tracking."""
    layer_index: int = -1
    request_id: str = ""
    step: int = 0


class CommHookManager:
    """Singleton manager for communication hooks.
    
    This manager provides:
    1. Layer context tracking via thread-local storage
    2. NCCL communication hooks for collective operations
    3. KV transfer hooks for P2D operations
    4. Deterministic drop mechanism at layer level
    
    The drop mechanism uses deterministic randomness based on:
    - Layer index
    - Request ID
    - Step
    - Communication type
    
    This ensures P and D workers make consistent drop decisions without coordination.
    
    Important TP Consistency:
    - For NCCL collective operations (all_reduce), we NEVER skip the call
      to avoid TP consistency issues. Instead, we zero out the tensor.
    - For KV transfers, P and D must use unified comm_type for consistent
      drop decisions.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Thread-local storage for layer context
        self._thread_local = threading.local()
        self._thread_local.context = LayerContext()
        
        # NCCL hooks: list of (hook_func, priority)
        self._nccl_hooks: list[tuple[Callable, int]] = []
        
        # KV transfer hooks: list of (hook_func, priority)
        self._kv_hooks: list[tuple[Callable, int]] = []
        
        # Drop configuration
        self._enabled = False
        self._drop_prob = 0.0
        self._seed = 42  # Fixed seed for deterministic drop
        
        # Track dropped layers for D-side fallback
        # Key: layer_index, Value: set of comm_types dropped for this layer
        self._dropped_layers: dict[int, set[str]] = {}
        
        # Stats
        self._total_nccl_ops = 0
        self._dropped_nccl_ops = 0
        self._total_kv_ops = 0
        self._dropped_kv_ops = 0
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    @property
    def drop_prob(self) -> float:
        return self._drop_prob
    
    @drop_prob.setter
    def drop_prob(self, value: float):
        self._drop_prob = max(0.0, min(1.0, value))
    
    @property
    def context(self) -> LayerContext:
        """Get the current thread-local layer context."""
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = LayerContext()
        return self._thread_local.context
    
    @context.setter
    def context(self, ctx: LayerContext):
        """Set the current thread-local layer context."""
        self._thread_local.context = ctx
    
    def set_layer_context(
        self,
        layer_index: int,
        request_id: str = "",
        step: int = 0,
    ):
        """Set the layer context for the current thread."""
        self._thread_local.context = LayerContext(
            layer_index=layer_index,
            request_id=request_id,
            step=step,
        )
    
    def clear_layer_context(self):
        """Clear the layer context for the current thread."""
        self._thread_local.context = LayerContext()
    
    def track_layer(
        self,
        layer_index: int,
        request_id: str = "",
        step: int = 0,
    ):
        """Context manager to track layer context.
        
        Usage:
            with comm_hook_manager.track_layer(layer_index=0, request_id="req_1"):
                # NCCL operations and KV transfers here will have layer context
                pass
        """
        class LayerContextManager:
            def __enter__(self):
                self._old_context = comm_hook_manager.context
                comm_hook_manager.set_layer_context(layer_index, request_id, step)
                return comm_hook_manager.context
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                comm_hook_manager.context = self._old_context
        
        return LayerContextManager()
    
    def _get_unified_comm_type(self, comm_type: str) -> str:
        """Get a unified comm_type for P/D consistency.
        
        For KV transfer operations, P and D use different method names:
        - P: save_kv_layer, send_blocks
        - D: start_load_kv
        
        This maps them to a unified type so both sides make consistent drop decisions.
        """
        # Map KV operations to unified type based on layer_index
        ctx = self.context
        if ctx.layer_index >= 0:
            if comm_type in ["save_kv_layer", "start_load_kv", "send_blocks"]:
                return f"kv_transfer_layer_{ctx.layer_index}"
        return comm_type
    
    def should_drop(self, comm_type: str) -> bool:
        """Determine if a communication operation should be dropped.
        
        Uses deterministic randomness based on layer context to ensure
        P and D workers make consistent drop decisions.
        
        Args:
            comm_type: Type of communication (e.g., "all_reduce", "send", "save_kv_layer")
        
        Returns:
            True if the operation should be dropped, False otherwise.
        """
        if not self.enabled or self.drop_prob <= 0:
            return False
        
        ctx = self.context
        if ctx.layer_index < 0:
            # For NCCL collective operations, require layer context to ensure consistency
            if comm_type in ["all_reduce"]:
                logging.warning(
                    "NCCL all_reduce called without layer context - "
                    "this may cause TP consistency issues!"
                )
            return False  # Not in a layer context
        
        # Use unified comm_type for P/D consistency
        unified_comm_type = self._get_unified_comm_type(comm_type)
        
        # Create deterministic seed from context
        seed_str = f"{ctx.layer_index}:{ctx.request_id}:{ctx.step}:{unified_comm_type}"
        seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) & 0xFFFFFFFF
        
        # Use deterministic randomness
        rng = random.Random(self._seed + seed_hash)
        return rng.random() < self.drop_prob
    
    def register_nccl_hook(self, hook_func: Callable, priority: int = 0):
        """Register an NCCL communication hook.
        
        Hooks are executed in priority order (higher priority first).
        
        Args:
            hook_func: Hook function with signature:
                func(comm_type: str, *args, **kwargs) -> tuple[bool, Any, Any]
                Returns (should_proceed, result, metadata)
            priority: Hook priority (higher = executed first)
        """
        self._nccl_hooks.append((hook_func, priority))
        self._nccl_hooks.sort(key=lambda x: -x[1])
    
    def register_kv_hook(self, hook_func: Callable, priority: int = 0):
        """Register a KV transfer hook.
        
        Hooks are executed in priority order (higher priority first).
        
        Args:
            hook_func: Hook function with signature:
                func(comm_type: str, *args, **kwargs) -> tuple[bool, Any, Any]
                Returns (should_proceed, result, metadata)
            priority: Hook priority (higher = executed first)
        """
        self._kv_hooks.append((hook_func, priority))
        self._kv_hooks.sort(key=lambda x: -x[1])
    
    def run_nccl_hooks(
        self,
        comm_type: str,
        *args,
        **kwargs,
    ) -> tuple[bool, Any, dict]:
        """Run all registered NCCL hooks for a communication operation.
        
        For NCCL collective operations (like all_reduce), we NEVER skip the call
        to avoid TP consistency issues. Instead, we zero out the tensor if dropped.
        
        Args:
            comm_type: Type of NCCL operation (e.g., "all_reduce", "send", "recv")
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
        
        Returns:
            tuple[bool, Any, dict]: (should_proceed, result, metadata)
                - should_proceed: If False, the operation should be skipped/dropped
                - result: Result from the last hook (if any)
                - metadata: Aggregated metadata from all hooks
                    "zero_tensor_before_call": True if tensor should be zeroed
        """
        self._total_nccl_ops += 1
        should_proceed = True
        result = None
        metadata: dict[str, Any] = {}
        
        # Check deterministic drop first
        if self.should_drop(comm_type):
            self._dropped_nccl_ops += 1
            metadata["dropped"] = True
            
            # For NCCL collective operations, we MUST NOT skip the call
            # Instead, we return a special flag to indicate the caller should zero out
            # the out_tensor AFTER calling NCCL. This protects the original in_tensor
            # (which may be a view of paged KV cache) from being modified.
            if comm_type == "all_reduce":
                metadata["zero_tensor_after_call"] = True
                should_proceed = True  # Always proceed with NCCL call for collectives
            else:
                should_proceed = False
            return should_proceed, result, metadata
        
        # Run registered hooks
        for hook_func, _ in self._nccl_hooks:
            try:
                hook_result = hook_func(comm_type, *args, **kwargs)
                if isinstance(hook_result, tuple) and len(hook_result) >= 2:
                    should_proceed, result = hook_result[:2]
                    if len(hook_result) >= 3:
                        metadata.update(hook_result[2])
                
                # For NCCL collective operations, never skip even if hook says to drop
                if comm_type == "all_reduce" and not should_proceed:
                    should_proceed = True
                    metadata["zero_tensor_after_call"] = True
                    metadata["dropped"] = True
                
                if not should_proceed:
                    self._dropped_nccl_ops += 1
                    break
            except Exception as e:
                # Don't let hooks break the communication pipeline
                pass
        
        return should_proceed, result, metadata
    
    def run_kv_hooks(
        self,
        comm_type: str,
        *args,
        **kwargs,
    ) -> tuple[bool, Any, dict]:
        """Run all registered KV transfer hooks for a communication operation.
        
        Args:
            comm_type: Type of KV operation (e.g., "save_kv_layer", "start_load_kv")
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
        
        Returns:
            tuple[bool, Any, dict]: (should_proceed, result, metadata)
                - should_proceed: If False, the operation should be skipped/dropped
                - result: Result from the last hook (if any)
                - metadata: Aggregated metadata from all hooks
        """
        self._total_kv_ops += 1
        should_proceed = True
        result = None
        metadata: dict[str, Any] = {}
        
        # Check deterministic drop first
        if self.should_drop(comm_type):
            self._dropped_kv_ops += 1
            should_proceed = False
            metadata["dropped"] = True
            
            # Record dropped layer for D-side fallback
            ctx = self.context
            if ctx.layer_index >= 0:
                if ctx.layer_index not in self._dropped_layers:
                    self._dropped_layers[ctx.layer_index] = set()
                self._dropped_layers[ctx.layer_index].add(comm_type)
            
            return should_proceed, result, metadata
        
        # Run registered hooks
        for hook_func, _ in self._kv_hooks:
            try:
                hook_result = hook_func(comm_type, *args, **kwargs)
                if isinstance(hook_result, tuple) and len(hook_result) >= 2:
                    should_proceed, result = hook_result[:2]
                    if len(hook_result) >= 3:
                        metadata.update(hook_result[2])
                if not should_proceed:
                    self._dropped_kv_ops += 1
                    break
            except Exception as e:
                # Don't let hooks break the communication pipeline
                pass
        
        return should_proceed, result, metadata
    
    def get_stats(self) -> dict[str, int]:
        """Get statistics about communication operations."""
        return {
            "total_nccl_ops": self._total_nccl_ops,
            "dropped_nccl_ops": self._dropped_nccl_ops,
            "total_kv_ops": self._total_kv_ops,
            "dropped_kv_ops": self._dropped_kv_ops,
        }
    
    def is_layer_dropped(self, layer_index: int) -> bool:
        """Check if a layer has been dropped in the current context."""
        return layer_index in self._dropped_layers
    
    def get_dropped_layers(self) -> set[int]:
        """Get the set of dropped layer indices."""
        return set(self._dropped_layers.keys())
    
    def reset_stats(self):
        """Reset statistics and dropped layers tracking."""
        self._total_nccl_ops = 0
        self._dropped_nccl_ops = 0
        self._total_kv_ops = 0
        self._dropped_kv_ops = 0
        self._dropped_layers = {}


# Global singleton instance
comm_hook_manager = CommHookManager()

# Convenience function for track_layer context manager
def track_layer(
    layer_index: int,
    request_id: str = "",
    step: int = 0,
):
    """Convenience function for track_layer context manager."""
    return comm_hook_manager.track_layer(layer_index, request_id, step)