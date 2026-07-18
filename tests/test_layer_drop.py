# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for layer drop functionality."""

import torch
import pytest

from vllm.model_executor.layers.layer_drop import (
    LayerDropManager,
    initialize_layer_drop,
    get_layer_drop_manager,
)


class TestLayerDropManager:
    """Test cases for LayerDropManager."""

    def test_init(self):
        """Test initialization of LayerDropManager."""
        manager = LayerDropManager(
            max_drop_ratio=0.5,
            length_ratio_threshold=1.5,
            priority_weight=2.0,
            length_weight=0.5,
            enabled=True,
        )
        assert manager.max_drop_ratio == 0.5
        assert manager.length_ratio_threshold == 1.5
        assert manager.priority_weight == 2.0
        assert manager.length_weight == 0.5
        assert manager.enabled is True

    def test_reset(self):
        """Test resetting dropped request tracking."""
        manager = LayerDropManager()
        manager.dropped_req_ids.add(1)
        manager.dropped_req_ids.add(2)
        assert len(manager.dropped_req_ids) == 2
        
        manager.reset()
        assert len(manager.dropped_req_ids) == 0

    def test_calculate_k(self):
        """Test K calculation based on layer position."""
        manager = LayerDropManager(max_drop_ratio=0.3)
        
        # Test with 10 requests and 10 layers
        num_reqs = 10
        total_layers = 10
        
        # Layer 0: K should be highest
        k0 = manager._calculate_k(num_reqs, 0, total_layers)
        assert k0 == int(10 * 0.3 * 1.0)
        
        # Middle layer: K should be around half
        k5 = manager._calculate_k(num_reqs, 5, total_layers)
        assert k5 == int(10 * 0.3 * 0.555)
        
        # Last layer: K should be 0
        k9 = manager._calculate_k(num_reqs, 9, total_layers)
        assert k9 == 0
        
        # Ensure at least 1 request remains
        k_max = manager._calculate_k(2, 0, 10)
        assert k_max == 1  # Only drop 1, keep 1

    def test_detect_stragglers(self):
        """Test straggler detection based on length."""
        manager = LayerDropManager(length_ratio_threshold=2.0)
        
        # Create sequence lengths with one straggler
        seq_lens = torch.tensor([10, 12, 11, 30, 9])  # 30 is a straggler
        num_reqs = 5
        
        is_straggler = manager._detect_stragglers(seq_lens, num_reqs)
        
        # Only the request with length 30 should be detected as straggler
        expected = torch.tensor([False, False, False, True, False])
        assert torch.equal(is_straggler, expected)

    def test_compute_drop_scores(self):
        """Test drop score computation."""
        manager = LayerDropManager(priority_weight=1.0, length_weight=1.0)
        
        seq_lens = torch.tensor([10, 20, 30])
        priorities = torch.tensor([0.9, 0.5, 0.1])
        
        scores = manager._compute_drop_scores(seq_lens, priorities, num_reqs=3)
        
        # Higher length + lower priority = higher score
        assert scores[0] < scores[1] < scores[2]

    def test_decide_drop_straggler_priority(self):
        """Test drop decision with both straggler and priority."""
        manager = LayerDropManager(max_drop_ratio=1.0, length_ratio_threshold=1.5)
        
        seq_lens = torch.tensor([10, 12, 50])  # 50 is a straggler
        priorities = torch.tensor([0.9, 0.1, 0.5])  # Request 1 has lowest priority
        
        drop_mask, keep_mask = manager.decide_drop(
            seq_lens,
            layer_idx=0,
            total_layers=10,
            priorities=priorities,
        )
        
        # Straggler (50) should be dropped
        assert drop_mask[2] is True

    def test_compact_hidden_states(self):
        """Test compacting hidden states after dropping."""
        manager = LayerDropManager()
        
        # Create hidden states for 3 requests with different lengths
        hidden_states = torch.randn(6, 4)  # 6 tokens, 4 hidden size
        keep_mask = torch.tensor([True, False, True])  # Drop request 1
        
        # Query start locations: request 0 has 2 tokens, request 1 has 2 tokens, request 2 has 2 tokens
        query_start_loc = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        
        compacted_hidden, index_map, keep_indices = manager.compact_hidden_states(
            hidden_states, keep_mask, query_start_loc
        )
        
        # Should have 4 tokens (requests 0 and 2)
        assert compacted_hidden.shape[0] == 4
        assert compacted_hidden.shape[1] == 4
        
        # Index map should map: 0->0, 1->1, 4->2, 5->3
        assert index_map[0] == 0
        assert index_map[1] == 1
        assert index_map[4] == 2
        assert index_map[5] == 3
        
        # keep_indices should be the original token positions kept: 0,1,4,5
        assert torch.equal(keep_indices, torch.tensor([0, 1, 4, 5]))
        
        # Scattering compacted back via keep_indices should recover original
        # rows for kept requests.
        scattered = torch.zeros_like(hidden_states)
        scattered[keep_indices] = compacted_hidden
        assert torch.equal(scattered[0], hidden_states[0])
        assert torch.equal(scattered[1], hidden_states[1])
        assert torch.equal(scattered[4], hidden_states[4])
        assert torch.equal(scattered[5], hidden_states[5])
        # Dropped rows should remain zero.
        assert torch.equal(scattered[2], torch.zeros(4))
        assert torch.equal(scattered[3], torch.zeros(4))

    def test_update_positions(self):
        """Test updating positions after dropping."""
        manager = LayerDropManager()
        
        positions = torch.tensor([0, 1, 10, 11, 20, 21])
        keep_mask = torch.tensor([True, False, True])
        query_start_loc = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        
        updated_positions = manager.update_positions(positions, keep_mask, query_start_loc)
        
        # Should keep positions for requests 0 and 2
        expected = torch.tensor([0, 1, 20, 21])
        assert torch.equal(updated_positions, expected)

    def test_decide_drop_disabled(self):
        """Test that no drops occur when layer drop is disabled."""
        manager = LayerDropManager(enabled=False)
        
        seq_lens = torch.tensor([10, 20, 30])
        drop_mask, keep_mask = manager.decide_drop(
            seq_lens,
            layer_idx=0,
            total_layers=10,
        )
        
        assert not drop_mask.any()
        assert keep_mask.all()

    def test_decide_drop_single_request(self):
        """Test that no drops occur when there's only one request."""
        manager = LayerDropManager(enabled=True)
        
        seq_lens = torch.tensor([10])
        drop_mask, keep_mask = manager.decide_drop(
            seq_lens,
            layer_idx=0,
            total_layers=10,
        )
        
        assert not drop_mask.any()
        assert keep_mask.all()

    def test_decide_drop_all_kept(self):
        """Test that all requests are kept when K is 0."""
        manager = LayerDropManager(max_drop_ratio=0.0)
        
        seq_lens = torch.tensor([10, 20, 30])
        drop_mask, keep_mask = manager.decide_drop(
            seq_lens,
            layer_idx=0,
            total_layers=10,
        )
        
        assert not drop_mask.any()
        assert keep_mask.all()


class TestLayerDropIntegration:
    """Integration tests for layer drop."""

    def test_initialize_layer_drop(self):
        """Test global initialization of layer drop."""
        manager = initialize_layer_drop(
            max_drop_ratio=0.4,
            length_ratio_threshold=2.5,
            priority_weight=1.5,
            length_weight=0.8,
            enabled=True,
        )
        
        retrieved = get_layer_drop_manager()
        assert retrieved is manager
        assert manager.max_drop_ratio == 0.4
        assert manager.length_ratio_threshold == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])