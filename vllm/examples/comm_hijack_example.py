# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example usage of TP/EP communication hijacking.

To use this in a real vLLM deployment:
1. Import the comm_hijack module
2. Call install_hijack() before model initialization
3. Set current request ID before each inference
4. Run inference as normal
5. Query DropRegistry for dropped tokens
6. Use the information to make scheduling decisions
"""

from vllm.distributed.comm_hijack import (
    install_hijack,
    uninstall_hijack,
    get_drop_registry,
    set_drop_ratios,
    set_strategy,
    set_request_id,
    set_seq_group_id,
    enable_hijack,
    disable_hijack,
    get_comm_freq,
)


def example_workflow():
    """Example workflow showing how to use comm hijack."""
    print("=== Comm Hijack Example Workflow ===\n")
    
    # Step 1: Install hijack with initial settings
    print("Step 1: Installing comm hijack...")
    install_hijack(
        tp_strategy="random",
        ep_strategy="load_based",
        tp_drop_ratio=0.1,
        ep_drop_ratio=0.1,
        seed=42,
        verbose=True,
        load_based_threshold=100,
        load_based_max_ratio=0.5,
    )
    
    # Step 2: For each request, set the request ID
    print("\nStep 2: Processing requests...")
    
    for request_id in range(3):
        set_request_id(request_id=request_id)
        set_seq_group_id(seq_group_id=request_id * 100)
        
        print(f"\nProcessing request {request_id}...")
        
        # Run vLLM inference here
        # The hijack will automatically intercept TP/EP communication
        
        # Check if this request has dropped tokens
        registry = get_drop_registry()
        if registry.has_drops(request_id):
            dropped = registry.get_dropped_tokens(request_id)
            print(f"Request {request_id} has {len(dropped)} dropped tokens")
        
        # Check communication frequency
        freq = get_comm_freq()
        print(f"Current comm frequency: TP={freq['tp']}/s, EP={freq['ep']}/s")
    
    # Step 3: Adjust strategies based on load
    print("\nStep 3: Adjusting strategies based on system load...")
    set_strategy(tp_strategy="load_based", ep_strategy="random")
    set_drop_ratios(tp_ratio=0.2, ep_ratio=0.3)
    
    # Step 4: Temporarily disable for critical requests
    print("\nStep 4: Disabling hijack for critical request...")
    disable_hijack()
    
    # Process critical request
    set_request_id(request_id=999)
    print("Processing critical request (hijack disabled)...")
    
    enable_hijack()
    
    # Step 5: Query final statistics
    print("\nStep 5: Final statistics...")
    registry = get_drop_registry()
    stats = registry.get_stats()
    print(f"Total drops: {stats['total_drops']}")
    print(f"TP drops: {stats['total_tp_drops']}")
    print(f"EP drops: {stats['total_ep_drops']}")
    print(f"Requests with drops: {stats['num_requests_with_drops']}")
    
    # Step 6: Cleanup
    print("\nStep 6: Cleaning up...")
    uninstall_hijack()
    
    print("\nExample workflow completed!")


def example_with_token_mask():
    """Example using TOKEN_MASK strategy."""
    print("\n=== TOKEN_MASK Strategy Example ===\n")
    
    install_hijack(tp_strategy="token_mask", ep_strategy="token_mask", verbose=True)
    
    # Create a token mask where tokens at indices 2, 5, 7 are dropped
    import torch
    mask = torch.tensor([
        True, True, False, True, True, 
        False, True, False, True, True
    ])
    
    from vllm.distributed.comm_hijack import set_token_mask
    set_token_mask(mask)
    
    print(f"Token mask set: {mask.tolist()}")
    print("Tokens at indices [2, 5, 7] will be dropped")
    
    # Run inference...
    
    uninstall_hijack()
    print("\nTOKEN_MASK example completed!")


if __name__ == "__main__":
    example_workflow()
    example_with_token_mask()