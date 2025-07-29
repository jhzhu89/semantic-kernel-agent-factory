#!/usr/bin/env python3

import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

# Add the project root to Python path
sys.path.insert(0, '/home/jiahzhu/ws_ai/semantic-kernel-agent-factory')

from agent_factory.mcp_server.auth.obo_credential_cache import OboCredentialCache, CachedCredential
from agent_factory.mcp_server.auth.app_credential_cache import AppCredentialCache
from agent_factory.mcp_server.auth.token_parser import TokenInfo


async def test_obo_expired_token_logic():
    """Test that expired tokens are properly handled in OBO cache"""
    print("Testing OBO expired token logic...")
    
    # Mock factory
    factory = MagicMock()
    factory.create_obo_credential = AsyncMock(return_value="new_credential")
    
    # Create cache
    cache = OboCredentialCache(factory)
    
    # Create expired token info
    expired_token = TokenInfo(
        tenant_id="test-tenant",
        client_id="test-client", 
        user_id="test-user",
        expiry=datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
    )
    
    # Manually add expired credential to cache
    expired_cred = CachedCredential("old_credential", expired_token)
    key = cache._k(expired_token)
    await cache._cache.set(key, expired_cred, ttl=60)  # Cache for 60 seconds
    
    # Verify it's in cache
    cached = await cache._cache.get(key)
    assert cached is not None, "Expired credential should be in cache initially"
    print("✓ Expired credential added to cache")
    
    # Mock parser to return expired token
    cache._parser.parse_token = MagicMock(return_value=expired_token)
    
    # Try to get credential - should detect expiry and recreate
    result = await cache.get_credential("fake_assertion")
    
    # Verify new credential was created
    assert result == "new_credential", "Should return new credential"
    factory.create_obo_credential.assert_called_once()
    
    # Verify expired credential was removed from cache
    cached_after = await cache._cache.get(key)
    assert cached_after is None, "Expired credential should be removed from cache"
    
    print("✓ Expired token properly detected and removed")
    print("✓ New credential created successfully")


async def test_obo_valid_token_caching():
    """Test that valid tokens are properly cached with correct TTL"""
    print("\nTesting OBO valid token caching...")
    
    factory = MagicMock()
    factory.create_obo_credential = AsyncMock(return_value="valid_credential")
    
    cache = OboCredentialCache(factory)
    
    # Create valid token (expires in 1 hour)
    valid_token = TokenInfo(
        tenant_id="test-tenant",
        client_id="test-client",
        user_id="test-user", 
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    
    cache._parser.parse_token = MagicMock(return_value=valid_token)
    
    # Get credential
    result = await cache.get_credential("fake_assertion")
    assert result == "valid_credential"
    factory.create_obo_credential.assert_called_once()
    
    # Verify it's cached
    key = cache._k(valid_token)
    cached = await cache._cache.get(key)
    assert isinstance(cached, CachedCredential)
    
    print("✓ Valid token properly cached")
    
    # Get again - should use cache
    factory.create_obo_credential.reset_mock()
    result2 = await cache.get_credential("fake_assertion")
    assert result2 == "valid_credential"
    factory.create_obo_credential.assert_not_called()
    
    print("✓ Cached credential reused on second call")


async def test_app_cache_ttl():
    """Test app credential cache TTL logic"""
    print("\nTesting App credential cache TTL...")
    
    factory = MagicMock()
    factory.create_app_credential = AsyncMock(return_value="app_credential")
    
    cache = AppCredentialCache(factory)
    
    # Get credential
    result = await cache.get_credential("tenant", "client")
    assert result == "app_credential"
    factory.create_app_credential.assert_called_once()
    
    # Verify it's cached
    key = cache._k("tenant", "client")
    cached = await cache._cache.get(key)
    assert cached is not None
    
    print("✓ App credential properly cached")
    
    # Get again - should use cache
    factory.create_app_credential.reset_mock()
    result2 = await cache.get_credential("tenant", "client")
    assert result2 == "app_credential"
    factory.create_app_credential.assert_not_called()
    
    print("✓ Cached app credential reused")


async def test_obo_expired_not_cached():
    """Test that expired tokens are not cached"""
    print("\nTesting that expired tokens are not cached...")
    
    factory = MagicMock()
    factory.create_obo_credential = AsyncMock(return_value="credential")
    
    cache = OboCredentialCache(factory)
    
    # Create already expired token
    expired_token = TokenInfo(
        tenant_id="test-tenant",
        client_id="test-client",
        user_id="test-user",
        expiry=datetime.utcnow() - timedelta(seconds=10)  # Expired 10 seconds ago
    )
    
    cache._parser.parse_token = MagicMock(return_value=expired_token)
    
    # Get credential
    result = await cache.get_credential("fake_assertion")
    assert result == "credential"
    
    # Verify it's NOT cached (because it was already expired)
    key = cache._k(expired_token)
    cached = await cache._cache.get(key)
    assert cached is None, "Expired token should not be cached"
    
    print("✓ Expired token was not cached")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Credential Cache TTL Logic")
    print("=" * 60)
    
    try:
        await test_obo_expired_token_logic()
        await test_obo_valid_token_caching()
        await test_app_cache_ttl()
        await test_obo_expired_not_cached()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Cache TTL logic is working correctly.")
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
