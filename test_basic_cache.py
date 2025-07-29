#!/usr/bin/env python3

import asyncio
from datetime import datetime, timedelta

# Simple test to verify cache behavior
async def test_basic_cache():
    from aiocache import SimpleMemoryCache
    
    print("Testing basic aiocache behavior...")
    
    cache = SimpleMemoryCache()
    
    # Test 1: Set with TTL and verify it expires
    print("Test 1: TTL expiration")
    await cache.set("key1", "value1", ttl=1)  # 1 second TTL
    
    # Should be there immediately
    val = await cache.get("key1")
    assert val == "value1", "Value should be available immediately"
    print("✓ Value available immediately")
    
    # Wait for expiration
    await asyncio.sleep(1.1)
    val = await cache.get("key1")
    assert val is None, "Value should be expired"
    print("✓ Value expired after TTL")
    
    # Test 2: Manual deletion
    print("Test 2: Manual deletion")
    await cache.set("key2", "value2", ttl=60)
    val = await cache.get("key2")
    assert val == "value2"
    
    await cache.delete("key2")
    val = await cache.get("key2")
    assert val is None, "Value should be deleted"
    print("✓ Manual deletion works")
    
    print("Basic cache tests passed!")

if __name__ == "__main__":
    asyncio.run(test_basic_cache())
