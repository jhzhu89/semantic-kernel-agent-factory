#!/usr/bin/env python3

import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

# Add the project root to Python path
sys.path.insert(0, '/home/jiahzhu/ws_ai/semantic-kernel-agent-factory')

from agent_factory.mcp_server.auth.app_credential_cache import AppCredentialCache, CachedCredential


async def test_app_cache_ttl_logic():
    """Test App credential cache TTL calculation and application"""
    print("Testing App credential cache TTL logic...")
    
    # Mock factory
    factory = MagicMock()
    factory.create_app_credential = AsyncMock(return_value="app_credential")
    
    # Create cache
    cache = AppCredentialCache(factory)
    
    # Verify initial configuration
    assert cache._ttl == 43_200  # 12 hours
    assert cache._buffer_seconds == 300  # 5 minutes
    print("✓ Cache configuration correct")
    
    # Test credential creation and caching
    result = await cache.get_credential("tenant", "client")
    assert result == "app_credential"
    factory.create_app_credential.assert_called_once()
    print("✓ Credential created successfully")
    
    # Verify it's cached with correct TTL calculation
    key = cache._k("tenant", "client")
    cached_entry = await cache._cache.get(key)
    assert isinstance(cached_entry, CachedCredential)
    assert cached_entry.credential == "app_credential"
    print("✓ Credential properly cached")
    
    # Test cache hit
    factory.create_app_credential.reset_mock()
    result2 = await cache.get_credential("tenant", "client")
    assert result2 == "app_credential"
    factory.create_app_credential.assert_not_called()
    print("✓ Cache hit works correctly")
    
    # Test different tenant/client creates separate cache entry
    result3 = await cache.get_credential("tenant2", "client2")
    assert result3 == "app_credential"
    factory.create_app_credential.assert_called_once()
    
    key2 = cache._k("tenant2", "client2")
    cached_entry2 = await cache._cache.get(key2)
    assert isinstance(cached_entry2, CachedCredential)
    print("✓ Separate cache entries for different tenant/client")


async def test_app_cache_exception_handling():
    """Test App cache exception caching and retrieval"""
    print("\nTesting App cache exception handling...")
    
    factory = MagicMock()
    test_exception = Exception("Credential creation failed")
    factory.create_app_credential = AsyncMock(side_effect=test_exception)
    
    cache = AppCredentialCache(factory)
    
    # First call should raise exception and cache it
    try:
        await cache.get_credential("tenant", "client")
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "Credential creation failed"
    
    print("✓ Exception raised correctly")
    
    # Verify exception is cached
    key = cache._k("tenant", "client")
    cached_entry = await cache._cache.get(key)
    assert isinstance(cached_entry, Exception)
    assert str(cached_entry) == "Credential creation failed"
    print("✓ Exception cached correctly")
    
    # Second call should retrieve cached exception
    factory.create_app_credential.reset_mock()
    try:
        await cache.get_credential("tenant", "client")
        assert False, "Should have raised cached exception"
    except Exception as e:
        assert str(e) == "Credential creation failed"
    
    # Verify no new credential creation attempt
    factory.create_app_credential.assert_not_called()
    print("✓ Cached exception retrieved correctly")


async def test_app_cache_concurrent_requests():
    """Test App cache handles concurrent requests correctly"""
    print("\nTesting App cache concurrent request handling...")
    
    factory = MagicMock()
    
    creation_called = asyncio.Event()
    creation_proceed = asyncio.Event()
    call_count = 0
    
    async def slow_create():
        nonlocal call_count
        call_count += 1
        creation_called.set()
        await creation_proceed.wait()
        return f"credential_{call_count}"
    
    factory.create_app_credential = AsyncMock(side_effect=slow_create)
    
    cache = AppCredentialCache(factory)
    
    # Start first request
    task1 = asyncio.create_task(cache.get_credential("tenant", "client"))
    await creation_called.wait()
    
    # Start second request while first is still running
    task2 = asyncio.create_task(cache.get_credential("tenant", "client"))
    
    # Allow creation to complete
    creation_proceed.set()
    
    result1 = await task1
    result2 = await task2
    
    # Both should get the same credential
    assert result1 == result2
    assert call_count == 1  # Only one creation call
    print("✓ Concurrent requests handled correctly")


async def test_app_cache_invalidation():
    """Test App cache invalidation functionality"""
    print("\nTesting App cache invalidation...")
    
    factory = MagicMock()
    factory.create_app_credential = AsyncMock(return_value="app_credential")
    
    cache = AppCredentialCache(factory)
    
    # Create and cache credential
    await cache.get_credential("tenant", "client")
    key = cache._k("tenant", "client")
    
    # Verify it's cached
    cached_entry = await cache._cache.get(key)
    assert cached_entry is not None
    print("✓ Credential cached")
    
    # Invalidate
    await cache.invalidate("tenant", "client")
    
    # Verify it's removed
    cached_entry = await cache._cache.get(key)
    assert cached_entry is None
    print("✓ Credential invalidated")
    
    # Next request should create new credential
    factory.create_app_credential.reset_mock()
    result = await cache.get_credential("tenant", "client")
    assert result == "app_credential"
    factory.create_app_credential.assert_called_once()
    print("✓ New credential created after invalidation")


async def test_app_cache_clear():
    """Test App cache clear functionality"""
    print("\nTesting App cache clear...")
    
    factory = MagicMock()
    factory.create_app_credential = AsyncMock(return_value="app_credential")
    
    cache = AppCredentialCache(factory)
    
    # Create multiple cached credentials
    await cache.get_credential("tenant1", "client1")
    await cache.get_credential("tenant2", "client2")
    
    key1 = cache._k("tenant1", "client1")
    key2 = cache._k("tenant2", "client2")
    
    # Verify both are cached
    assert await cache._cache.get(key1) is not None
    assert await cache._cache.get(key2) is not None
    print("✓ Multiple credentials cached")
    
    # Clear cache
    await cache.clear()
    
    # Verify all are removed
    assert await cache._cache.get(key1) is None
    assert await cache._cache.get(key2) is None
    print("✓ All credentials cleared")


async def main():
    """Run all App cache tests"""
    print("=" * 60)
    print("Testing App Credential Cache")
    print("=" * 60)
    
    try:
        await test_app_cache_ttl_logic()
        await test_app_cache_exception_handling()
        await test_app_cache_concurrent_requests()
        await test_app_cache_invalidation()
        await test_app_cache_clear()
        
        print("\n" + "=" * 60)
        print("✅ ALL APP CACHE TESTS PASSED!")
        print("App credential cache is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
