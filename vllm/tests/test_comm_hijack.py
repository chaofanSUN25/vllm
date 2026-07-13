# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test script for TP/EP communication hijacking demo.
"""

import torch

from vllm.distributed.comm_hijack import (
    install_hijack,
    uninstall_hijack,
    get_drop_registry,
    get_hijack_config,
    set_drop_ratios,
    set_strategy,
    set_token_mask,
    get_comm_freq,
    DropStrategy,
)


def test_drop_registry():
    """Test DropRegistry operations."""
    print("\n=== Testing DropRegistry ===")
    
    registry = get_drop_registry()
    registry.reset()
    
    # Register drops
    registry.register_drop(request_id=1, token_idx=10, drop_type="tp")
    registry.register_drop(request_id=1, token_idx=20, drop_type="tp")
    registry.register_drops(request_id=2, token_indices=[5, 15, 25], drop_type="ep")
    registry.register_seq_group_drop(seq_group_id=100, token_idx=5)
    
    # Query drops
    drops_1 = registry.get_dropped_tokens(request_id=1)
    drops_2 = registry.get_dropped_tokens(request_id=2)
    print(f"Drops for request 1: {drops_1}")
    print(f"Drops for request 2: {drops_2}")
    
    # Check has_drops
    print(f"Request 1 has drops: {registry.has_drops(request_id=1)}")
    print(f"Request 3 has drops: {registry.has_drops(request_id=3)}")
    
    # Check seq_group drops
    print(f"SeqGroup 100 has drops: {registry.has_seq_group_drops(seq_group_id=100)}")
    
    # Get stats
    stats = registry.get_stats()
    print(f"Registry stats: {stats}")
    
    # Reset again
    registry.reset()
    stats_after = registry.get_stats()
    print(f"Stats after reset: {stats_after}")
    
    assert stats_after["total_drops"] == 0
    assert stats_after["total_tp_drops"] == 0
    assert stats_after["total_ep_drops"] == 0
    print("✅ DropRegistry test passed")


def test_install_uninstall():
    """Test installation and uninstallation."""
    print("\n=== Testing Install/Uninstall ===")
    
    # Install with default settings
    install_hijack(tp_drop_ratio=0.1, ep_drop_ratio=0.1, verbose=False)
    
    config = get_hijack_config()
    assert config.enable == True
    assert config.tp_drop_ratio == 0.1
    assert config.ep_drop_ratio == 0.1
    
    print("✅ Install successful")
    
    # Uninstall
    uninstall_hijack()
    
    config = get_hijack_config()
    assert config.enable == False
    assert config.tp_drop_ratio == 0.0
    assert config.ep_drop_ratio == 0.0
    
    print("✅ Uninstall successful")


def test_strategy_config():
    """Test different drop strategies."""
    print("\n=== Testing Drop Strategies ===")
    
    install_hijack(tp_strategy="random", ep_strategy="load_based", verbose=False)
    
    config = get_hijack_config()
    assert config.tp_strategy == DropStrategy.RANDOM
    assert config.ep_strategy == DropStrategy.LOAD_BASED
    
    # Update strategy
    set_strategy(tp_strategy="bypass", ep_strategy="token_mask")
    
    config = get_hijack_config()
    assert config.tp_strategy == DropStrategy.BYPASS
    assert config.ep_strategy == DropStrategy.TOKEN_MASK
    
    # Test token mask
    mask = torch.tensor([True, True, False, True, False])
    set_token_mask(mask)
    assert get_hijack_config()._token_mask is not None
    
    uninstall_hijack()
    print("✅ Strategy configuration test passed")


def test_dynamic_ratio_update():
    """Test dynamically updating drop ratios."""
    print("\n=== Testing Dynamic Ratio Update ===")
    
    install_hijack(tp_drop_ratio=0.1, ep_drop_ratio=0.1, verbose=False)
    
    # Update ratios
    set_drop_ratios(tp_ratio=0.5, ep_ratio=0.3)
    
    config = get_hijack_config()
    assert config.tp_drop_ratio == 0.5
    assert config.ep_drop_ratio == 0.3
    
    uninstall_hijack()
    print("✅ Dynamic ratio update test passed")


def test_comm_freq_tracking():
    """Test communication frequency tracking."""
    print("\n=== Testing Communication Frequency Tracking ===")
    
    install_hijack(tp_drop_ratio=0.0, ep_drop_ratio=0.0, verbose=False)
    
    # Simulate communication
    from vllm.distributed.comm_hijack import _update_comm_freq
    
    for _ in range(50):
        _update_comm_freq("tp")
    
    freq = get_comm_freq()
    assert freq["tp"] == 50
    assert freq["ep"] == 0
    
    uninstall_hijack()
    print("✅ Communication frequency tracking test passed")


def test_request_id_tracking():
    """Test request ID tracking."""
    print("\n=== Testing Request ID Tracking ===")
    
    install_hijack(tp_drop_ratio=0.0, ep_drop_ratio=0.0, verbose=False)
    
    from vllm.distributed.comm_hijack import set_request_id, set_seq_group_id
    
    set_request_id(123)
    set_seq_group_id(456)
    
    config = get_hijack_config()
    assert config.request_id == 123
    assert config._seq_group_id == 456
    
    uninstall_hijack()
    print("✅ Request ID tracking test passed")


if __name__ == "__main__":
    print("Running CommHijack tests...")
    
    test_drop_registry()
    test_install_uninstall()
    test_strategy_config()
    test_dynamic_ratio_update()
    test_comm_freq_tracking()
    test_request_id_tracking()
    
    print("\n✅ All tests completed!")