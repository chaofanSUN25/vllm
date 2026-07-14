# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM TP/EP Communication Hijack Demo

This module intercepts TP and EP communication operations for research purposes.
The core idea is to simulate communication overload by dropping tokens:

- **TP AllReduce**: Mask dropped tokens to 0 before all_reduce
- **EP Dispatch/Combine**: Prevent specific tokens from being sent/received

All dropped token indices are recorded in DropRegistry for scheduler integration.

Supported drop strategies:
- RANDOM: Randomly drop a fixed ratio of tokens
- LOAD_BASED: Dynamically adjust drop ratio based on communication frequency
- TOKEN_MASK: Apply user-specified token mask
- BYPASS: Completely skip the communication (extreme test)
"""

import random
import threading
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Callable

import torch
import torch.distributed as dist

from vllm.distributed import (
    get_tp_group,
    get_ep_group,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


class DropStrategy(Enum):
    """
    Strategy for determining which tokens to drop.
    
    RANDOM: Randomly drop a fixed ratio of tokens
    LOAD_BASED: Dynamically adjust drop ratio based on communication frequency
    TOKEN_MASK: Apply user-specified token mask (via set_token_mask)
    BYPASS: Completely skip the communication operation
    """
    RANDOM = "random"
    LOAD_BASED = "load_based"
    TOKEN_MASK = "token_mask"
    BYPASS = "bypass"


class DropRegistry:
    """
    Registry to track dropped tokens across communication operations.
    
    This provides a centralized store for all token drops, which can be
    queried by the scheduler for final request dropping decisions.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        # Map: request_id -> set of dropped token indices
        self.dropped_tokens: Dict[int, Set[int]] = defaultdict(set)
        # Map: seq_group_id -> set of dropped token indices
        self.seq_group_drops: Dict[int, Set[int]] = defaultdict(set)
        # Map: request_id -> set of dropped positions (global)
        self.dropped_positions: Dict[int, Set[int]] = defaultdict(set)
        # Stats
        self.total_drops = 0
        self.total_tp_drops = 0
        self.total_ep_drops = 0
        self.lock = threading.RLock()
    
    def register_drop(self, request_id: int, token_idx: int, drop_type: str = "tp") -> None:
        """Register a dropped token."""
        with self.lock:
            self.dropped_tokens[request_id].add(token_idx)
            self.total_drops += 1
            if drop_type == "tp":
                self.total_tp_drops += 1
            elif drop_type == "ep":
                self.total_ep_drops += 1
    
    def register_drops(self, request_id: int, token_indices: List[int], drop_type: str = "tp") -> None:
        """Register multiple dropped tokens."""
        with self.lock:
            for idx in token_indices:
                self.dropped_tokens[request_id].add(idx)
            self.total_drops += len(token_indices)
            if drop_type == "tp":
                self.total_tp_drops += len(token_indices)
            elif drop_type == "ep":
                self.total_ep_drops += len(token_indices)
    
    def register_seq_group_drop(self, seq_group_id: int, token_idx: int) -> None:
        """Register a dropped token for a specific seq_group."""
        with self.lock:
            self.seq_group_drops[seq_group_id].add(token_idx)
    
    def get_dropped_tokens(self, request_id: int) -> Set[int]:
        """Get all dropped tokens for a request."""
        with self.lock:
            return self.dropped_tokens.get(request_id, set())
    
    def get_seq_group_drops(self, seq_group_id: int) -> Set[int]:
        """Get all dropped tokens for a seq_group."""
        with self.lock:
            return self.seq_group_drops.get(seq_group_id, set())
    
    def has_drops(self, request_id: int) -> bool:
        """Check if a request has any dropped tokens."""
        with self.lock:
            return len(self.dropped_tokens.get(request_id, set())) > 0
    
    def has_seq_group_drops(self, seq_group_id: int) -> bool:
        """Check if a seq_group has any dropped tokens."""
        with self.lock:
            return len(self.seq_group_drops.get(seq_group_id, set())) > 0
    
    def reset(self) -> None:
        """Reset all drop records."""
        with self.lock:
            self.dropped_tokens.clear()
            self.seq_group_drops.clear()
            self.dropped_positions.clear()
            self.total_drops = 0
            self.total_tp_drops = 0
            self.total_ep_drops = 0
    
    def get_stats(self) -> Dict:
        """Get statistics about drops."""
        with self.lock:
            return {
                "total_drops": self.total_drops,
                "total_tp_drops": self.total_tp_drops,
                "total_ep_drops": self.total_ep_drops,
                "num_requests_with_drops": len(self.dropped_tokens),
                "num_seq_groups_with_drops": len(self.seq_group_drops),
            }


class CommHijackConfig:
    """
    Configuration for communication hijacking.
    
    Attributes:
        enable: Whether to enable hijacking
        tp_strategy: Drop strategy for TP communication
        ep_strategy: Drop strategy for EP communication
        tp_drop_ratio: Base drop ratio for TP (0.0-1.0)
        ep_drop_ratio: Base drop ratio for EP (0.0-1.0)
        seed: Random seed for reproducibility
        verbose: Whether to print debug information
        load_based_threshold: Communication frequency threshold for LOAD_BASED
        load_based_max_ratio: Maximum drop ratio for LOAD_BASED
        request_id: Current request ID (for token tracking)
    """
    
    def __init__(self):
        self.enable = False
        self.tp_strategy: DropStrategy = DropStrategy.RANDOM
        self.ep_strategy: DropStrategy = DropStrategy.RANDOM
        self.tp_drop_ratio = 0.0
        self.ep_drop_ratio = 0.0
        self.seed = 42
        self.verbose = False
        self.load_based_threshold = 100  # ops per second
        self.load_based_max_ratio = 0.5
        self.request_id = 0
        self._token_mask: Optional[torch.Tensor] = None
        self._seq_group_id = 0


# Global config and registry
_hijack_config = CommHijackConfig()
_drop_registry = DropRegistry()

# Communication frequency tracking for LOAD_BASED strategy
_comm_freq_counter = {"tp": 0, "ep": 0}
_comm_freq_lock = threading.RLock()
_comm_freq_last_reset = time.time()

# Original functions storage
_original_tp_all_reduce = None
_original_group_all_reduce = None
_original_group_dispatch = None
_original_group_combine = None
_original_torch_all_reduce = None
_original_torch_all_to_all_single = None


def get_hijack_config() -> CommHijackConfig:
    """Get the global hijack configuration."""
    return _hijack_config


def get_drop_registry() -> DropRegistry:
    """Get the global drop registry."""
    return _drop_registry


def set_token_mask(mask: torch.Tensor) -> None:
    """
    Set a custom token mask for TOKEN_MASK strategy.
    
    Args:
        mask: Boolean tensor where True means keep token, False means drop
    """
    _hijack_config._token_mask = mask


def get_token_mask() -> Optional[torch.Tensor]:
    """Get the current token mask."""
    return _hijack_config._token_mask


def _update_comm_freq(op_type: str) -> None:
    """Update communication frequency counter."""
    global _comm_freq_last_reset
    
    with _comm_freq_lock:
        _comm_freq_counter[op_type] += 1
        
        # Reset every second
        now = time.time()
        if now - _comm_freq_last_reset >= 1.0:
            _comm_freq_counter["tp"] = 0
            _comm_freq_counter["ep"] = 0
            _comm_freq_last_reset = now


def _get_load_based_ratio(op_type: str) -> float:
    """Calculate drop ratio based on communication load."""
    with _comm_freq_lock:
        freq = _comm_freq_counter[op_type]
    
    threshold = _hijack_config.load_based_threshold
    max_ratio = _hijack_config.load_based_max_ratio
    
    if freq <= threshold:
        return 0.0
    
    # Linear increase from 0 to max_ratio based on frequency
    ratio = min(max_ratio, (freq - threshold) / threshold * max_ratio)
    return ratio


def _generate_drop_indices(num_tokens: int, drop_ratio: float) -> List[int]:
    """
    Generate indices of tokens to drop based on current strategy.
    
    Args:
        num_tokens: Total number of tokens
        drop_ratio: Base drop ratio
        
    Returns:
        List of token indices to drop
    """
    if num_tokens == 0:
        return []
    
    strategy = _hijack_config.tp_strategy
    
    if strategy == DropStrategy.BYPASS:
        # Drop all tokens
        return list(range(num_tokens))
    
    if strategy == DropStrategy.TOKEN_MASK:
        # Use user-specified mask
        mask = _hijack_config._token_mask
        if mask is not None and mask.numel() >= num_tokens:
            mask = mask[:num_tokens]
            return torch.where(~mask)[0].tolist()
        return []
    
    # Calculate actual drop ratio
    if strategy == DropStrategy.LOAD_BASED:
        actual_ratio = _get_load_based_ratio("tp")
    else:  # RANDOM
        actual_ratio = drop_ratio
    
    if actual_ratio <= 0.0:
        return []
    
    num_drop = max(1, int(num_tokens * actual_ratio))
    return random.sample(range(num_tokens), num_drop)


def _generate_ep_drop_indices(num_tokens: int, drop_ratio: float) -> List[int]:
    """
    Generate indices of tokens to drop for EP operations.
    
    Args:
        num_tokens: Total number of tokens
        drop_ratio: Base drop ratio
        
    Returns:
        List of token indices to drop
    """
    if num_tokens == 0:
        return []
    
    strategy = _hijack_config.ep_strategy
    
    if strategy == DropStrategy.BYPASS:
        return list(range(num_tokens))
    
    if strategy == DropStrategy.TOKEN_MASK:
        mask = _hijack_config._token_mask
        if mask is not None and mask.numel() >= num_tokens:
            mask = mask[:num_tokens]
            return torch.where(~mask)[0].tolist()
        return []
    
    if strategy == DropStrategy.LOAD_BASED:
        actual_ratio = _get_load_based_ratio("ep")
    else:  # RANDOM
        actual_ratio = drop_ratio
    
    if actual_ratio <= 0.0:
        return []
    
    num_drop = max(1, int(num_tokens * actual_ratio))
    return random.sample(range(num_tokens), num_drop)


# ==================== TP Communication Hijacking ====================

def _hijacked_tp_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """
    Hijacked TP all_reduce implementation (top-level).
    
    Randomly masks out tokens before all_reduce, effectively dropping them
    from the communication.
    """
    if not _hijack_config.enable:
        return _original_tp_all_reduce(input_)
    
    _update_comm_freq("tp")
    
    if _hijack_config.verbose:
        print(f"[TP Hijack] tensor_model_parallel_all_reduce called on shape: {input_.shape}")
    
    # Generate drop indices
    drop_indices = _generate_drop_indices(input_.shape[0], _hijack_config.tp_drop_ratio)
    
    if len(drop_indices) > 0:
        if _hijack_config.verbose:
            print(f"[TP Hijack] Dropping {len(drop_indices)} tokens out of {input_.shape[0]}")
        
        # Mask the input tensor (set dropped tokens to 0)
        masked_input = input_.clone()
        masked_input[drop_indices] = 0.0
        
        # Register dropped tokens
        _drop_registry.register_drops(
            request_id=_hijack_config.request_id,
            token_indices=drop_indices,
            drop_type="tp"
        )
        
        return _original_tp_all_reduce(masked_input)
    
    return _original_tp_all_reduce(input_)


def _hijacked_group_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """
    Hijacked GroupCoordinator.all_reduce (lower-level).
    
    This provides a second layer of interception for TP group communication.
    """
    if not _hijack_config.enable:
        return _original_group_all_reduce(self, input_)
    
    _update_comm_freq("tp")
    
    if self.unique_name.startswith("tp") and _hijack_config.verbose:
        print(f"[TP Hijack] GroupCoordinator.all_reduce called on shape: {input_.shape}")
    
    # For BYPASS strategy, return input without communication
    if _hijack_config.tp_strategy == DropStrategy.BYPASS:
        if _hijack_config.verbose:
            print(f"[TP Hijack] BYPASS mode: skipping all_reduce")
        return input_
    
    # Generate drop indices
    drop_indices = _generate_drop_indices(input_.shape[0], _hijack_config.tp_drop_ratio)
    
    if len(drop_indices) > 0:
        # Mask the input tensor
        masked_input = input_.clone()
        masked_input[drop_indices] = 0.0
        
        _drop_registry.register_drops(
            request_id=_hijack_config.request_id,
            token_indices=drop_indices,
            drop_type="tp"
        )
        
        return _original_group_all_reduce(self, masked_input)
    
    return _original_group_all_reduce(self, input_)


# ==================== EP Communication Hijacking ====================

def _hijacked_group_dispatch(
    self,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    is_sequence_parallel: bool = False,
    extra_tensors: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Hijacked GroupCoordinator.dispatch for EP.
    
    Prevent specific tokens from being sent to other ranks during EP dispatch.
    """
    if not _hijack_config.enable:
        return _original_group_dispatch(
            self, hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors
        )
    
    _update_comm_freq("ep")
    
    if self.unique_name.startswith("ep") and _hijack_config.verbose:
        print(f"[EP Hijack] dispatch called on hidden_states shape: {hidden_states.shape}")
    
    # For BYPASS strategy, return inputs without communication
    if _hijack_config.ep_strategy == DropStrategy.BYPASS:
        if _hijack_config.verbose:
            print(f"[EP Hijack] BYPASS mode: skipping dispatch")
        if extra_tensors is not None:
            return hidden_states, topk_weights, topk_ids, extra_tensors
        return hidden_states, topk_weights, topk_ids
    
    # Generate drop indices
    drop_indices = _generate_ep_drop_indices(hidden_states.shape[0], _hijack_config.ep_drop_ratio)
    
    if len(drop_indices) > 0:
        if _hijack_config.verbose:
            print(f"[EP Hijack] Dropping {len(drop_indices)} tokens during dispatch")
        
        # Mask dropped tokens by setting their topk_weights to 0
        # This effectively prevents them from being routed to experts
        masked_topk_weights = topk_weights.clone()
        if len(masked_topk_weights.shape) == 1:
            masked_topk_weights[drop_indices] = 0.0
        else:
            masked_topk_weights[drop_indices, :] = 0.0
        
        # Also mask the hidden states for dropped tokens
        masked_hidden_states = hidden_states.clone()
        masked_hidden_states[drop_indices] = 0.0
        
        # Register dropped tokens
        _drop_registry.register_drops(
            request_id=_hijack_config.request_id,
            token_indices=drop_indices,
            drop_type="ep"
        )
        
        return _original_group_dispatch(
            self, masked_hidden_states, masked_topk_weights, topk_ids, 
            is_sequence_parallel, extra_tensors
        )
    
    return _original_group_dispatch(
        self, hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors
    )


def _hijacked_group_combine(
    self,
    hidden_states: torch.Tensor,
    is_sequence_parallel: bool = False,
) -> torch.Tensor:
    """
    Hijacked GroupCoordinator.combine for EP.
    
    Prevent receiving results for dropped tokens during EP combine.
    """
    if not _hijack_config.enable:
        return _original_group_combine(self, hidden_states, is_sequence_parallel)
    
    _update_comm_freq("ep")
    
    if self.unique_name.startswith("ep") and _hijack_config.verbose:
        print(f"[EP Hijack] combine called on hidden_states shape: {hidden_states.shape}")
    
    # For BYPASS strategy, return input without communication
    if _hijack_config.ep_strategy == DropStrategy.BYPASS:
        if _hijack_config.verbose:
            print(f"[EP Hijack] BYPASS mode: skipping combine")
        return hidden_states
    
    # Generate drop indices
    drop_indices = _generate_ep_drop_indices(hidden_states.shape[0], _hijack_config.ep_drop_ratio)
    
    if len(drop_indices) > 0:
        if _hijack_config.verbose:
            print(f"[EP Hijack] Dropping {len(drop_indices)} tokens during combine")
        
        # Perform original combine
        result = _original_group_combine(self, hidden_states, is_sequence_parallel)
        
        # Mask the result to zero out dropped tokens
        masked_result = result.clone()
        masked_result[drop_indices] = 0.0
        
        return masked_result
    
    return _original_group_combine(self, hidden_states, is_sequence_parallel)


# ==================== Torch Distributed Fallback Hijacking ====================

def _hijacked_torch_all_reduce(input_: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Hijacked torch.distributed.all_reduce (fallback).
    
    This provides a safety net for any all_reduce calls not caught by other layers.
    """
    if not _hijack_config.enable:
        return _original_torch_all_reduce(input_, **kwargs)
    
    _update_comm_freq("tp")
    
    if _hijack_config.verbose:
        print(f"[Torch Hijack] all_reduce called on shape: {input_.shape}")
    
    # For BYPASS strategy, skip communication
    if _hijack_config.tp_strategy == DropStrategy.BYPASS:
        return input_
    
    # Generate drop indices
    drop_indices = _generate_drop_indices(input_.shape[0], _hijack_config.tp_drop_ratio)
    
    if len(drop_indices) > 0:
        masked_input = input_.clone()
        masked_input[drop_indices] = 0.0
        
        _drop_registry.register_drops(
            request_id=_hijack_config.request_id,
            token_indices=drop_indices,
            drop_type="tp"
        )
        
        return _original_torch_all_reduce(masked_input, **kwargs)
    
    return _original_torch_all_reduce(input_, **kwargs)


def _hijacked_torch_all_to_all_single(output_tensor: torch.Tensor, input_tensor: torch.Tensor, **kwargs) -> None:
    """
    Hijacked torch.distributed.all_to_all_single (fallback for EP).
    
    This provides a safety net for EP all-to-all operations.
    """
    if not _hijack_config.enable:
        return _original_torch_all_to_all_single(output_tensor, input_tensor, **kwargs)
    
    _update_comm_freq("ep")
    
    if _hijack_config.verbose:
        print(f"[Torch Hijack] all_to_all_single called on input shape: {input_tensor.shape}")
    
    # For BYPASS strategy, just copy input to output without communication
    if _hijack_config.ep_strategy == DropStrategy.BYPASS:
        output_tensor.copy_(input_tensor)
        return
    
    # Generate drop indices
    drop_indices = _generate_ep_drop_indices(input_tensor.shape[0], _hijack_config.ep_drop_ratio)
    
    if len(drop_indices) > 0:
        masked_input = input_tensor.clone()
        masked_input[drop_indices] = 0.0
        
        _drop_registry.register_drops(
            request_id=_hijack_config.request_id,
            token_indices=drop_indices,
            drop_type="ep"
        )
        
        return _original_torch_all_to_all_single(output_tensor, masked_input, **kwargs)
    
    return _original_torch_all_to_all_single(output_tensor, input_tensor, **kwargs)


# ==================== Installation/Uninstallation ====================

def install_hijack(
    tp_strategy: str = "random",
    ep_strategy: str = "random",
    tp_drop_ratio: float = 0.0,
    ep_drop_ratio: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
    load_based_threshold: int = 100,
    load_based_max_ratio: float = 0.5,
) -> None:
    """
    Install communication hijacking hooks.
    
    Args:
        tp_strategy: Drop strategy for TP ('random', 'load_based', 'token_mask', 'bypass')
        ep_strategy: Drop strategy for EP ('random', 'load_based', 'token_mask', 'bypass')
        tp_drop_ratio: Base drop ratio for TP (0.0-1.0)
        ep_drop_ratio: Base drop ratio for EP (0.0-1.0)
        seed: Random seed for reproducibility
        verbose: Whether to print debug information
        load_based_threshold: Communication frequency threshold for LOAD_BASED strategy
        load_based_max_ratio: Maximum drop ratio for LOAD_BASED strategy
    """
    global _original_tp_all_reduce, _original_group_all_reduce
    global _original_group_dispatch, _original_group_combine
    global _original_torch_all_reduce, _original_torch_all_to_all_single
    global tensor_model_parallel_all_reduce
    
    # Configure hijack
    _hijack_config.enable = True
    _hijack_config.tp_strategy = DropStrategy(tp_strategy)
    _hijack_config.ep_strategy = DropStrategy(ep_strategy)
    _hijack_config.tp_drop_ratio = tp_drop_ratio
    _hijack_config.ep_drop_ratio = ep_drop_ratio
    _hijack_config.seed = seed
    _hijack_config.verbose = verbose
    _hijack_config.load_based_threshold = load_based_threshold
    _hijack_config.load_based_max_ratio = load_based_max_ratio
    
    # Set random seed
    random.seed(seed)
    
    # Reset communication frequency counter
    global _comm_freq_counter, _comm_freq_last_reset
    with _comm_freq_lock:
        _comm_freq_counter["tp"] = 0
        _comm_freq_counter["ep"] = 0
        _comm_freq_last_reset = time.time()
    
    # Monkey patch TP communication (top-level)
    if _original_tp_all_reduce is None:
        _original_tp_all_reduce = tensor_model_parallel_all_reduce
        tensor_model_parallel_all_reduce = _hijacked_tp_all_reduce
    
    # Monkey patch GroupCoordinator.all_reduce (lower-level TP)
    if _original_group_all_reduce is None:
        _original_group_all_reduce = GroupCoordinator.all_reduce
        GroupCoordinator.all_reduce = _hijacked_group_all_reduce
    
    # Monkey patch GroupCoordinator.dispatch (EP)
    if _original_group_dispatch is None:
        _original_group_dispatch = GroupCoordinator.dispatch
        GroupCoordinator.dispatch = _hijacked_group_dispatch
    
    # Monkey patch GroupCoordinator.combine (EP)
    if _original_group_combine is None:
        _original_group_combine = GroupCoordinator.combine
        GroupCoordinator.combine = _hijacked_group_combine
    
    # Monkey patch torch.distributed.all_reduce (fallback)
    if _original_torch_all_reduce is None:
        _original_torch_all_reduce = dist.all_reduce
        dist.all_reduce = _hijacked_torch_all_reduce
    
    # Monkey patch torch.distributed.all_to_all_single (fallback for EP)
    if _original_torch_all_to_all_single is None:
        _original_torch_all_to_all_single = dist.all_to_all_single
        dist.all_to_all_single = _hijacked_torch_all_to_all_single
    
    print(f"[CommHijack] Installed:")
    print(f"  TP Strategy: {tp_strategy}, Ratio: {tp_drop_ratio}")
    print(f"  EP Strategy: {ep_strategy}, Ratio: {ep_drop_ratio}")


def uninstall_hijack() -> None:
    """
    Uninstall communication hijacking hooks and restore original functions.
    """
    global _original_tp_all_reduce, _original_group_all_reduce
    global _original_group_dispatch, _original_group_combine
    global _original_torch_all_reduce, _original_torch_all_to_all_single
    global tensor_model_parallel_all_reduce
    
    # Restore TP all_reduce (top-level)
    if _original_tp_all_reduce is not None:
        tensor_model_parallel_all_reduce = _original_tp_all_reduce
        _original_tp_all_reduce = None
    
    # Restore GroupCoordinator.all_reduce
    if _original_group_all_reduce is not None:
        GroupCoordinator.all_reduce = _original_group_all_reduce
        _original_group_all_reduce = None
    
    # Restore GroupCoordinator.dispatch
    if _original_group_dispatch is not None:
        GroupCoordinator.dispatch = _original_group_dispatch
        _original_group_dispatch = None
    
    # Restore GroupCoordinator.combine
    if _original_group_combine is not None:
        GroupCoordinator.combine = _original_group_combine
        _original_group_combine = None
    
    # Restore torch.distributed.all_reduce
    if _original_torch_all_reduce is not None:
        dist.all_reduce = _original_torch_all_reduce
        _original_torch_all_reduce = None
    
    # Restore torch.distributed.all_to_all_single
    if _original_torch_all_to_all_single is not None:
        dist.all_to_all_single = _original_torch_all_to_all_single
        _original_torch_all_to_all_single = None
    
    # Reset config
    _hijack_config.enable = False
    _hijack_config.tp_strategy = DropStrategy.RANDOM
    _hijack_config.ep_strategy = DropStrategy.RANDOM
    _hijack_config.tp_drop_ratio = 0.0
    _hijack_config.ep_drop_ratio = 0.0
    
    # Reset registry
    _drop_registry.reset()
    
    print("[CommHijack] Uninstalled")


def set_strategy(tp_strategy: str, ep_strategy: str) -> None:
    """
    Update drop strategies dynamically.
    
    Args:
        tp_strategy: New TP drop strategy ('random', 'load_based', 'token_mask', 'bypass')
        ep_strategy: New EP drop strategy ('random', 'load_based', 'token_mask', 'bypass')
    """
    _hijack_config.tp_strategy = DropStrategy(tp_strategy)
    _hijack_config.ep_strategy = DropStrategy(ep_strategy)
    
    if _hijack_config.verbose:
        print(f"[CommHijack] Updated strategies: TP={tp_strategy}, EP={ep_strategy}")


def set_drop_ratios(tp_ratio: float, ep_ratio: float) -> None:
    """
    Update drop ratios dynamically without reinstalling.
    
    Args:
        tp_ratio: New TP drop ratio (0.0-1.0)
        ep_ratio: New EP drop ratio (0.0-1.0)
    """
    _hijack_config.tp_drop_ratio = tp_ratio
    _hijack_config.ep_drop_ratio = ep_ratio
    
    if _hijack_config.verbose:
        print(f"[CommHijack] Updated ratios: TP={tp_ratio}, EP={ep_ratio}")


def set_request_id(request_id: int) -> None:
    """
    Set the current request ID for token tracking.
    
    Args:
        request_id: Current request ID
    """
    _hijack_config.request_id = request_id


def set_seq_group_id(seq_group_id: int) -> None:
    """
    Set the current seq_group ID for token tracking.
    
    Args:
        seq_group_id: Current seq_group ID
    """
    _hijack_config._seq_group_id = seq_group_id


def enable_hijack() -> None:
    """Enable hijacking without changing configuration."""
    _hijack_config.enable = True
    if _hijack_config.verbose:
        print("[CommHijack] Enabled")


def disable_hijack() -> None:
    """Disable hijacking without uninstalling."""
    _hijack_config.enable = False
    if _hijack_config.verbose:
        print("[CommHijack] Disabled")


def get_comm_freq() -> Dict[str, int]:
    """Get current communication frequency."""
    with _comm_freq_lock:
        return dict(_comm_freq_counter)