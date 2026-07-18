# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer drop implementation for vLLM.

This module implements layer-level request dropping to reduce computation
and communication overhead in tensor parallel inference.
"""

from dataclasses import replace
from typing import Any

import torch
from torch import nn

from vllm.distributed import (
    get_tp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


class LayerDropManager:
    """Manager for layer-level request dropping.
    
    Implements two drop strategies:
    1. Length-based: Drop requests with significantly longer lengths to prevent
       stragglers from dragging down batch SLO.
    2. Score-based: Calculate drop scores based on priority and prompt length,
       then drop top-K requests. K is layer-dependent (highest at layer 0).
    
    Both strategies are fused together for the final decision.
    """

    def __init__(
        self,
        max_drop_ratio: float = 0.3,
        length_ratio_threshold: float = 2.0,
        priority_weight: float = 1.0,
        length_weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Initialize the layer drop manager.
        
        Args:
            max_drop_ratio: Maximum fraction of requests that can be dropped
                at any layer.
            length_ratio_threshold: If a request's length is more than this
                ratio times the median length, it's considered a straggler.
            priority_weight: Weight for priority factor in drop score.
            length_weight: Weight for prompt length factor in drop score.
            enabled: Whether layer drop is enabled.
        """
        self.max_drop_ratio = max_drop_ratio
        self.length_ratio_threshold = length_ratio_threshold
        self.priority_weight = priority_weight
        self.length_weight = length_weight
        self.enabled = enabled

        # Track dropped requests across layers for consistency
        self.dropped_req_ids: set[int] = set()
        # Per-layer precomputed drop masks: layer_idx -> bool tensor [num_reqs]
        self.layer_drop_masks: dict[int, torch.Tensor] = {}
        # Cumulative drop mask across all layers processed so far
        self.final_drop_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset the dropped request tracking for a new batch."""
        self.dropped_req_ids.clear()
        self.layer_drop_masks.clear()
        self.final_drop_mask = None

    def precompute_layer_drop_masks(
        self,
        seq_lens: torch.Tensor,
        total_layers: int,
        priorities: torch.Tensor | None = None,
        req_ids: torch.Tensor | None = None,
        is_prefilling: torch.Tensor | None = None,
    ) -> None:
        """Precompute drop masks for all layers.
        
        This is called by the runner before model execution to ensure
        consistent drop decisions across the entire pipeline.
        
        Args:
            seq_lens: Tensor of sequence lengths for each request.
            total_layers: Total number of layers in the model.
            priorities: Optional tensor of request priorities (0-1, higher=better).
            req_ids: Optional tensor of request IDs for tracking.
            is_prefilling: Optional bool tensor [num_reqs], True for requests
                still in prefill phase. When provided, only prefill requests
                are eligible for dropping; decode requests are never dropped.
                When None, all requests are eligible (legacy behavior).
        """
        with open("/tmp/layer_drop_debug.txt", "a") as _f:
            _f.write(f"[LAYER_DROP] precompute called: enabled={self.enabled}, "
                     f"num_reqs={seq_lens.shape[0]}, total_layers={total_layers}, "
                     f"is_prefilling={is_prefilling}\n")
        logger.warning("[LAYER_DROP] precompute called: enabled=%s, "
                       "num_reqs=%s, total_layers=%s, is_prefilling=%s",
                       self.enabled, seq_lens.shape[0], total_layers,
                       is_prefilling)
        if not self.enabled:
            return
        
        num_reqs = seq_lens.shape[0]
        if num_reqs <= 1:
            logger.warning("[LAYER_DROP] skipped: num_reqs=%s <= 1", num_reqs)
            return
        
        # When is_prefilling is provided, skip entirely for decode-only
        # batches so the forward path stays CUDA-graph compatible.
        if is_prefilling is not None and not bool(is_prefilling.any()):
            return
        
        # Initialize final drop mask (starts as all False)
        self.final_drop_mask = torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device)
        
        # Precompute drop mask for each layer
        for layer_idx in range(total_layers):
            k = self._calculate_k(num_reqs, layer_idx, total_layers)
            if k <= 0:
                self.layer_drop_masks[layer_idx] = torch.zeros(
                    num_reqs, dtype=torch.bool, device=seq_lens.device
                )
                continue
            
            # Strategy 1: Detect stragglers
            is_straggler = self._detect_stragglers(seq_lens, num_reqs)
            
            # Strategy 2: Compute drop scores
            drop_scores = self._compute_drop_scores(seq_lens, priorities, num_reqs)
            
            # Combine strategies: boost straggler scores
            boosted_scores = drop_scores.clone()
            boosted_scores[is_straggler] += 100.0
            
            # Exclude decode requests from drop candidates so only prefill
            # requests can be dropped.
            if is_prefilling is not None:
                boosted_scores[~is_prefilling] = float('-inf')
            
            # Exclude already dropped requests from consideration
            available_scores = boosted_scores.clone()
            available_scores[self.final_drop_mask] = float('-inf')

            # Clamp k to the number of still-available requests (keep >= 1)
            num_available = num_reqs - int(self.final_drop_mask.sum().item())
            k = min(k, max(num_available - 1, 0))
            if k <= 0:
                # No new drops at this layer, but requests dropped earlier
                # must still be skipped by subsequent layers.
                self.layer_drop_masks[layer_idx] = self.final_drop_mask.clone()
                continue

            # Select top-K requests to drop
            _, top_k_indices = torch.topk(available_scores, k, largest=True)
            drop_mask = torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device)
            drop_mask[top_k_indices] = True

            # DEBUG: force drop request 0 at layer 0 to verify end-to-end path
            if layer_idx == 0 and num_reqs > 1:
                logger.warning("[LAYER_DROP] forcing drop of request 0 at layer 0")
                drop_mask[0] = True
            
            # Synchronize across TP ranks
            drop_mask = self._sync_drop_mask(drop_mask)
            
            # Update final drop mask and store the cumulative mask for this
            # layer so that every later layer skips all already-dropped
            # requests instead of re-processing them.
            self.final_drop_mask |= drop_mask
            self.layer_drop_masks[layer_idx] = self.final_drop_mask.clone()

    def get_drop_mask_for_layer(
        self,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """Get the precomputed drop mask for a specific layer.
        
        Args:
            layer_idx: The layer index to get the drop mask for.
            
        Returns:
            The drop mask for the specified layer, or None if not precomputed.
        """
        return self.layer_drop_masks.get(layer_idx)

    def get_dropped_req_indices(self) -> list[int]:
        """Return indices of requests dropped at any layer this step.

        The runner uses this to map dropped request indices back to
        request IDs and report them to the scheduler, which then frees
        their KV cache blocks and skips sampling.

        Returns:
            List of request indices (into the seq_lens tensor passed to
            precompute_layer_drop_masks) that were dropped. Empty when no
            requests were dropped or layer drop is disabled.
        """
        if self.final_drop_mask is None or not bool(self.final_drop_mask.any()):
            return []
        return self.final_drop_mask.nonzero(as_tuple=False).flatten().tolist()

    def _calculate_k(self, num_reqs: int, layer_idx: int, total_layers: int) -> int:
        """Calculate K (number of requests to drop) based on layer position.
        
        K is highest at layer 0 (more aggressive dropping early) and 
        decreases linearly to 0 at the last layer.
        
        Args:
            num_reqs: Number of requests in the batch.
            layer_idx: Current layer index (0-based).
            total_layers: Total number of layers in the model.
            
        Returns:
            Number of requests to drop at this layer.
        """
        if total_layers <= 1 or num_reqs <= 1:
            return 0
        
        # Linearly decrease ratio from max_ratio at layer 0 to 0 at last layer
        ratio = 1.0 - (layer_idx / (total_layers - 1))
        k = int(num_reqs * self.max_drop_ratio * ratio)
        
        # Ensure at least 1 request remains after dropping
        return min(k, num_reqs - 1)

    def _detect_stragglers(
        self,
        seq_lens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Detect straggler requests based on length differences.
        
        A request is considered a straggler if its length is significantly
        longer than the median length of the batch.
        
        Args:
            seq_lens: Tensor of sequence lengths for each request.
            num_reqs: Number of requests in the batch.
            
        Returns:
            Boolean tensor indicating which requests are stragglers.
        """
        if num_reqs <= 1:
            return torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device)
        
        # Calculate median length
        sorted_lens, _ = torch.sort(seq_lens)
        mid = num_reqs // 2
        if num_reqs % 2 == 0:
            median_len = (sorted_lens[mid - 1] + sorted_lens[mid]) / 2
        else:
            median_len = sorted_lens[mid]
        
        # Detect stragglers
        is_straggler = seq_lens > (median_len * self.length_ratio_threshold)
        return is_straggler

    def _compute_drop_scores(
        self,
        seq_lens: torch.Tensor,
        priorities: torch.Tensor | None = None,
        num_reqs: int = 0,
    ) -> torch.Tensor:
        """Compute drop scores for each request.
        
        Higher scores mean higher probability of being dropped.
        Score = priority_weight * (1 - priority) + length_weight * normalized_length
        
        Args:
            seq_lens: Tensor of sequence lengths for each request.
            priorities: Optional tensor of request priorities (0-1, higher=better).
            num_reqs: Number of requests in the batch.
            
        Returns:
            Tensor of drop scores for each request.
        """
        scores = torch.zeros(num_reqs, dtype=torch.float32, device=seq_lens.device)
        
        # Normalize sequence lengths
        if num_reqs > 0 and seq_lens.max() > 0:
            normalized_lengths = seq_lens.float() / seq_lens.max()
            scores += self.length_weight * normalized_lengths
        
        # Priority factor: lower priority = higher drop score
        if priorities is not None and num_reqs > 0:
            # Assuming priorities are 0-1, invert so lower priority = higher score
            scores += self.priority_weight * (1.0 - priorities.float())
        
        return scores

    def _sync_drop_mask(self, drop_mask: torch.Tensor) -> torch.Tensor:
        """Synchronize drop mask across TP ranks.
        
        Ensures all TP ranks agree on which requests to drop for consistency.
        
        Args:
            drop_mask: Boolean tensor indicating which requests to drop.
            
        Returns:
            Synchronized drop mask (same across all TP ranks).
        """
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return drop_mask

        tp_group = get_tp_group()
        # Convert boolean to float for all_reduce
        drop_mask_float = drop_mask.float()

        # All reduce to get the majority vote
        reduced_mask = tp_group.all_reduce(drop_mask_float)

        # Round to get binary decision
        synchronized_mask = reduced_mask >= (tp_world_size / 2)
        return synchronized_mask.to(torch.bool)

    def decide_drop(
        self,
        seq_lens: torch.Tensor,
        layer_idx: int,
        total_layers: int,
        priorities: torch.Tensor | None = None,
        req_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make drop decisions for the current layer.
        
        Combines two strategies:
        1. Length-based straggler detection
        2. Score-based ranking
        
        Args:
            seq_lens: Tensor of sequence lengths for each request.
            layer_idx: Current layer index (0-based).
            total_layers: Total number of layers in the model.
            priorities: Optional tensor of request priorities.
            req_ids: Optional tensor of request IDs for tracking.
            
        Returns:
            Tuple of (drop_mask, keep_mask):
                drop_mask: Boolean tensor indicating which requests to drop.
                keep_mask: Boolean tensor indicating which requests to keep.
        """
        if not self.enabled:
            num_reqs = seq_lens.shape[0]
            return (
                torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device),
                torch.ones(num_reqs, dtype=torch.bool, device=seq_lens.device),
            )
        
        num_reqs = seq_lens.shape[0]
        if num_reqs <= 1:
            return (
                torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device),
                torch.ones(num_reqs, dtype=torch.bool, device=seq_lens.device),
            )
        
        # Calculate K for this layer
        k = self._calculate_k(num_reqs, layer_idx, total_layers)
        if k <= 0:
            return (
                torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device),
                torch.ones(num_reqs, dtype=torch.bool, device=seq_lens.device),
            )
        
        # Strategy 1: Detect stragglers based on length
        is_straggler = self._detect_stragglers(seq_lens, num_reqs)
        
        # Strategy 2: Compute drop scores
        drop_scores = self._compute_drop_scores(seq_lens, priorities, num_reqs)
        
        # Combine strategies:
        # - Stragglers get a boost to their drop score
        # - Then select top-K requests with highest scores
        boosted_scores = drop_scores.clone()
        boosted_scores[is_straggler] += 100.0  # Large boost for stragglers
        
        # Select top-K requests to drop (k is already clamped to <= num_reqs-1
        # by _calculate_k, so topk is always safe)
        _, top_k_indices = torch.topk(boosted_scores, k, largest=True)
        drop_mask = torch.zeros(num_reqs, dtype=torch.bool, device=seq_lens.device)
        drop_mask[top_k_indices] = True
        
        # Ensure consistency across already dropped requests
        # (once dropped, stay dropped)
        if req_ids is not None:
            for i in range(num_reqs):
                if int(req_ids[i].item()) in self.dropped_req_ids:
                    drop_mask[i] = True
        
        # Synchronize across TP ranks
        drop_mask = self._sync_drop_mask(drop_mask)
        
        # Update tracking
        if req_ids is not None:
            dropped_ids = req_ids[drop_mask].tolist()
            self.dropped_req_ids.update(int(id_) for id_ in dropped_ids)
        
        keep_mask = ~drop_mask
        return drop_mask, keep_mask

    def compact_hidden_states(
        self,
        hidden_states: torch.Tensor,
        keep_mask: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact hidden_states by removing dropped tokens.
        
        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_size].
            keep_mask: Boolean tensor indicating which requests to keep.
            query_start_loc: Tensor of shape [num_reqs + 1] indicating start
                positions of each request in the hidden_states tensor.
                
        Returns:
            Tuple of (compacted_hidden_states, index_map, keep_indices):
                compacted_hidden_states: Hidden states with dropped requests
                    removed.
                index_map: Mapping from original token indices to new indices,
                    shape [num_tokens]. Used for reindexing slot_mapping etc.
                keep_indices: Original token indices that were kept, shape
                    [num_kept_tokens]. Used to scatter outputs back.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Compute per-request token counts and start/end positions for kept
        # requests, then build a token-level keep mask using only GPU ops.
        req_lens = query_start_loc[1:] - query_start_loc[:-1]
        kept_req_lens = req_lens[keep_mask]
        kept_starts = query_start_loc[:-1][keep_mask]
        num_kept_tokens = int(kept_req_lens.sum().item())

        if num_kept_tokens == 0:
            # All requests are dropped
            return (
                torch.empty(0, hidden_states.shape[1], dtype=dtype, device=device),
                torch.zeros(num_tokens, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
            )

        # Mark +1 at the start of each kept request and -1 right after its end,
        # then cumsum to obtain a per-token boolean keep mask. This avoids
        # per-request .item() syncs and Python loops.
        starts = kept_starts.long()
        ends = starts + kept_req_lens.long()
        delta = torch.zeros(
            num_tokens + 1, dtype=torch.int64, device=device
        )
        ones = torch.ones(starts.shape[0], dtype=torch.int64, device=device)
        delta.scatter_add_(0, starts, ones)
        delta.scatter_add_(0, ends, -ones)
        token_keep_mask = delta[:-1].cumsum(0).bool()

        keep_indices = token_keep_mask.nonzero(as_tuple=False).flatten()
        compacted_hidden_states = hidden_states[keep_indices]

        # Build index map (original index -> new index)
        index_map = torch.zeros(
            num_tokens, dtype=torch.int64, device=device
        )
        index_map[keep_indices] = torch.arange(
            num_kept_tokens, dtype=torch.int64, device=device
        )

        return compacted_hidden_states, index_map, keep_indices

    def update_metadata(
        self,
        metadata: CommonAttentionMetadata,
        keep_mask: torch.Tensor,
        index_map: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> CommonAttentionMetadata:
        """Update attention metadata after dropping requests.
        
        Args:
            metadata: Original attention metadata.
            keep_mask: Boolean tensor indicating which requests to keep.
            index_map: Mapping from original token indices to new indices,
                shape [num_tokens]. Used for reindexing token-axis tensors.
            keep_indices: Original token indices that were kept, shape
                [num_kept_tokens]. Used to gather per-token fields.
            
        Returns:
            Updated attention metadata with dropped requests removed.
        """
        # V1 FlashAttention uses FlashAttentionMetadata, which has a different
        # field layout than CommonAttentionMetadata. Route to the specialized
        # updater when we detect that backend.
        if (
            hasattr(metadata, "block_table")
            and not hasattr(metadata, "block_table_tensor")
        ):
            return self._update_flash_attention_metadata(
                metadata, keep_mask, index_map, keep_indices
            )

        # Filter kept requests
        kept_seq_lens = metadata.seq_lens[keep_mask]
        
        # Get original query lens
        orig_query_lens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
        kept_query_lens = orig_query_lens[keep_mask]
        
        # Compute new query_start_loc
        num_kept_reqs = kept_seq_lens.shape[0]
        new_query_start_loc = torch.zeros(
            num_kept_reqs + 1, dtype=torch.int32, device=metadata.query_start_loc.device
        )
        if num_kept_reqs > 0:
            new_query_start_loc[0] = 0
            new_query_start_loc[1:] = torch.cumsum(kept_query_lens, dim=0)
        
        # Update CPU version
        new_query_start_loc_cpu = new_query_start_loc.cpu()
        
        # Filter block_table
        kept_block_table = metadata.block_table_tensor[keep_mask]
        
        # Update slot_mapping: gather slots for kept tokens only.
        # keep_indices has length num_kept_tokens, matching the compacted
        # token axis. Using index_map here would be wrong (length num_tokens).
        new_slot_mapping = metadata.slot_mapping[keep_indices]
        
        # Compute new num_actual_tokens
        new_num_actual_tokens = int(new_query_start_loc[-1].item()) if num_kept_reqs > 0 else 0
        
        # Compute new max_query_len and max_seq_len
        new_max_query_len = int(kept_query_lens.max().item()) if num_kept_reqs > 0 else 0
        new_max_seq_len = int(kept_seq_lens.max().item()) if num_kept_reqs > 0 else 0
        
        # Update causal mask if it's a tensor
        if isinstance(metadata.causal, torch.Tensor):
            new_causal = metadata.causal[keep_mask]
        else:
            new_causal = metadata.causal
        
        # Update is_prefilling if present
        if metadata.is_prefilling is not None:
            new_is_prefilling = metadata.is_prefilling[keep_mask]
        else:
            new_is_prefilling = None
        
        # Update rswa_prefix_lens if present
        if metadata.rswa_prefix_lens is not None:
            new_rswa_prefix_lens = metadata.rswa_prefix_lens[keep_mask]
        else:
            new_rswa_prefix_lens = None
        
        # Update encoder_seq_lens if present
        if metadata.encoder_seq_lens is not None:
            new_encoder_seq_lens = metadata.encoder_seq_lens[keep_mask]
        else:
            new_encoder_seq_lens = None
        
        # Update dcp_local_seq_lens if present
        if metadata.dcp_local_seq_lens is not None:
            new_dcp_local_seq_lens = metadata.dcp_local_seq_lens[keep_mask]
        else:
            new_dcp_local_seq_lens = None

        # Update CPU-side per-request upper bounds if present
        if metadata.seq_lens_cpu_upper_bound is not None:
            new_seq_lens_cpu_upper_bound = metadata.seq_lens_cpu_upper_bound[
                keep_mask
            ]
        else:
            new_seq_lens_cpu_upper_bound = None

        if metadata.encoder_seq_lens_cpu is not None:
            new_encoder_seq_lens_cpu = metadata.encoder_seq_lens_cpu[keep_mask]
        else:
            new_encoder_seq_lens_cpu = None

        if metadata.dcp_local_seq_lens_cpu is not None:
            new_dcp_local_seq_lens_cpu = metadata.dcp_local_seq_lens_cpu[
                keep_mask
            ]
        else:
            new_dcp_local_seq_lens_cpu = None

        # Update token-level positions if present
        if metadata.positions is not None:
            new_positions = metadata.positions[keep_indices]
        else:
            new_positions = None

        # Update logits indices for kv-sharing fast prefill if present.
        # Dropped requests must be removed and surviving indices remapped to
        # the compacted token layout.
        new_logits_indices_padded = metadata.logits_indices_padded
        new_num_logits_indices = metadata.num_logits_indices
        if (
            metadata.logits_indices_padded is not None
            and metadata.num_logits_indices is not None
            and metadata.num_logits_indices > 0
        ):
            logits_indices = metadata.logits_indices_padded[:metadata.num_logits_indices]
            valid_token_mask = torch.zeros(
                metadata.num_actual_tokens,
                dtype=torch.bool,
                device=logits_indices.device,
            )
            valid_token_mask[keep_indices] = True
            kept_logits_mask = valid_token_mask[logits_indices]
            if kept_logits_mask.any():
                new_logits_indices = index_map[logits_indices[kept_logits_mask]]
                new_num_logits_indices = int(new_logits_indices.shape[0])
                new_logits_indices_padded = metadata.logits_indices_padded.clone()
                new_logits_indices_padded[:new_num_logits_indices].copy_(
                    new_logits_indices
                )
                if new_num_logits_indices < metadata.logits_indices_padded.shape[0]:
                    new_logits_indices_padded[new_num_logits_indices:] = (
                        new_logits_indices[-1]
                    )
            else:
                new_num_logits_indices = 0
                new_logits_indices_padded = None

        # Update multimodal PrefixLM ranges if present. Keys are request
        # indices and must be renumbered after dropping.
        if metadata.mm_req_doc_ranges is not None:
            keep_mask_list = keep_mask.tolist()
            new_mm_req_doc_ranges: dict[int, list[tuple[int, int]]] = {}
            new_req_idx = 0
            for old_req_idx, kept in enumerate(keep_mask_list):
                if kept:
                    if old_req_idx in metadata.mm_req_doc_ranges:
                        new_mm_req_doc_ranges[new_req_idx] = (
                            metadata.mm_req_doc_ranges[old_req_idx]
                        )
                    new_req_idx += 1
        else:
            new_mm_req_doc_ranges = None

        # Create updated metadata
        updated_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=kept_seq_lens,
            num_reqs=num_kept_reqs,
            num_actual_tokens=new_num_actual_tokens,
            max_query_len=new_max_query_len,
            max_seq_len=new_max_seq_len,
            block_table_tensor=kept_block_table,
            slot_mapping=new_slot_mapping,
            causal=new_causal,
            logits_indices_padded=new_logits_indices_padded,
            num_logits_indices=new_num_logits_indices,
            encoder_seq_lens=new_encoder_seq_lens,
            encoder_seq_lens_cpu=new_encoder_seq_lens_cpu,
            dcp_local_seq_lens=new_dcp_local_seq_lens,
            dcp_local_seq_lens_cpu=new_dcp_local_seq_lens_cpu,
            positions=new_positions,
            is_prefilling=new_is_prefilling,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            mm_req_doc_ranges=new_mm_req_doc_ranges,
            rswa_prefix_lens=new_rswa_prefix_lens,
        )
        
        return updated_metadata

    def _update_flash_attention_metadata(
        self,
        metadata: Any,
        keep_mask: torch.Tensor,
        index_map: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> Any:
        """Return a compacted copy of FlashAttentionMetadata after dropping.

        FlashAttentionMetadata has a different shape and field set than
        CommonAttentionMetadata. We update the per-request and per-token
        fields on a shallow copy and conservatively disable cascade attention
        / DCP / scheduler metadata for the compacted batch because recomputing
        those structures is backend-specific and not needed for correctness.
        """
        num_kept_reqs = int(keep_mask.sum().item())
        if num_kept_reqs == 0:
            # All requests dropped; leave caller to handle empty batch.
            return metadata

        # Work on a copy so the original attention metadata object shared
        # across layers is not mutated.
        metadata = replace(metadata)

        # Per-request lengths
        kept_seq_lens = metadata.seq_lens[keep_mask]
        query_lens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
        kept_query_lens = query_lens[keep_mask]

        # Rebuild query_start_loc
        new_query_start_loc = torch.zeros(
            num_kept_reqs + 1,
            dtype=metadata.query_start_loc.dtype,
            device=metadata.query_start_loc.device,
        )
        new_query_start_loc[0] = 0
        new_query_start_loc[1:] = torch.cumsum(kept_query_lens, dim=0)
        metadata.query_start_loc = new_query_start_loc

        metadata.seq_lens = kept_seq_lens
        metadata.num_actual_tokens = int(new_query_start_loc[-1].item())
        metadata.max_query_len = int(kept_query_lens.max().item())
        metadata.max_seq_len = int(kept_seq_lens.max().item())

        # Block table and slot mapping
        metadata.block_table = metadata.block_table[keep_mask]
        metadata.slot_mapping = metadata.slot_mapping[keep_indices]

        # Causal mask: backend supports both bool and per-request tensor.
        if isinstance(metadata.causal, torch.Tensor):
            metadata.causal = metadata.causal[keep_mask]

        # Multimodal PrefixLM ranges: renumber request indices.
        if metadata.mm_prefix_range_tensor is not None:
            keep_mask_cpu = keep_mask.cpu()
            old_to_new = torch.cumsum(keep_mask_cpu, dim=0) - 1
            kept_old_indices = old_to_new[keep_mask_cpu].long()
            metadata.mm_prefix_range_tensor = (
                metadata.mm_prefix_range_tensor[keep_mask]
            )
            # The tensor is [num_seqs, max_ranges, 2]; request axis is already
            # filtered above, no further index rewrite needed.
            _ = kept_old_indices  # silence unused variable

        # R-SWA prefix lengths
        if metadata.rswa_prefix_lens is not None:
            metadata.rswa_prefix_lens = metadata.rswa_prefix_lens[keep_mask]

        # Cascade attention structures depend on the exact prefix shared across
        # the batch. Dropping requests can change the common prefix, so disable
        # cascade for the compacted batch to keep correctness.
        metadata.use_cascade = False
        metadata.common_prefix_len = 0
        metadata.cu_prefix_query_lens = None
        metadata.prefix_kv_lens = None
        metadata.suffix_kv_lens = None

        # GQA DCP and AOT scheduler metadata are also batch-shape dependent;
        # clear them to force the backend to recompute or fall back.
        metadata.max_dcp_context_kv_len = None
        metadata.dcp_context_kv_lens = None
        metadata.scheduler_metadata = None
        metadata.prefix_scheduler_metadata = None
        metadata.max_num_splits = 0

        return metadata

    def update_positions(
        self,
        positions: torch.Tensor,
        keep_mask: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Update positions tensor after dropping requests.
        
        Args:
            positions: Original positions tensor of shape [num_tokens].
            keep_mask: Boolean tensor indicating which requests to keep.
            query_start_loc: Tensor of shape [num_reqs + 1] indicating start
                positions of each request in the positions tensor.
                
        Returns:
            Updated positions tensor with dropped requests removed.
        """
        num_reqs = keep_mask.shape[0]
        
        # Build indices of tokens to keep
        indices_to_keep = []
        for i in range(num_reqs):
            if keep_mask[i]:
                start = query_start_loc[i].item()
                end = query_start_loc[i + 1].item()
                indices_to_keep.extend(range(start, end))
        
        if not indices_to_keep:
            return torch.empty(0, dtype=positions.dtype, device=positions.device)
        
        indices_to_keep_tensor = torch.tensor(
            indices_to_keep, dtype=torch.int64, device=positions.device
        )
        
        return positions[indices_to_keep_tensor]


# Global layer drop manager instance
_layer_drop_manager: LayerDropManager | None = None


def get_layer_drop_manager() -> LayerDropManager:
    """Get the global layer drop manager instance."""
    global _layer_drop_manager
    if _layer_drop_manager is None:
        _layer_drop_manager = LayerDropManager()
    return _layer_drop_manager


def set_layer_drop_manager(manager: LayerDropManager) -> None:
    """Set the global layer drop manager instance."""
    global _layer_drop_manager
    _layer_drop_manager = manager


def initialize_layer_drop(
    max_drop_ratio: float = 0.3,
    length_ratio_threshold: float = 2.0,
    priority_weight: float = 1.0,
    length_weight: float = 1.0,
    enabled: bool = True,
) -> LayerDropManager:
    """Initialize the layer drop manager with specified parameters."""
    global _layer_drop_manager
    _layer_drop_manager = LayerDropManager(
        max_drop_ratio=max_drop_ratio,
        length_ratio_threshold=length_ratio_threshold,
        priority_weight=priority_weight,
        length_weight=length_weight,
        enabled=enabled,
    )
    return _layer_drop_manager