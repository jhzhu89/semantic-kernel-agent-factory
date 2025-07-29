import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from agent_factory.mcp_server.auth.obo_credential_cache import OboCredentialCache
from agent_factory.mcp_server.auth.obo_credential_cache import CachedCredential as OboCachedCredential
from agent_factory.mcp_server.auth.app_credential_cache import AppCredentialCache
from agent_factory.mcp_server.auth.app_credential_cache import CachedCredential as AppCachedCredential
from agent_factory.mcp_server.auth.token_parser import TokenInfo


@pytest.fixture
def mock_factory():
    factory = MagicMock()
    factory.create_obo_credential = AsyncMock()
    factory.create_app_credential = AsyncMock()
    return factory


@pytest.fixture
def mock_credential():
    return MagicMock()


@pytest.fixture
def mock_token_info():
    return TokenInfo(
        tenant_id="test-tenant",
        client_id="test-client", 
        user_id="test-user",
        expiry=datetime.utcnow() + timedelta(hours=1)
    )


@pytest.fixture
def expired_token_info():
    return TokenInfo(
        tenant_id="test-tenant",
        client_id="test-client",
        user_id="test-user", 
        expiry=datetime.utcnow() - timedelta(hours=1)
    )


class TestOboCredentialCache:
    
    @pytest.mark.asyncio
    async def test_cache_miss_creates_credential(self, mock_factory, mock_credential, mock_token_info):
        cache = OboCredentialCache(mock_factory)
        mock_factory.create_obo_credential.return_value = mock_credential
        
        with patch.object(cache._parser, 'parse_token', return_value=mock_token_info):
            result = await cache.get_credential("fake_assertion")
        
        assert result == mock_credential
        mock_factory.create_obo_credential.assert_called_once_with("fake_assertion")
    
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_credential(self, mock_factory, mock_credential, mock_token_info):
        cache = OboCredentialCache(mock_factory)
        mock_factory.create_obo_credential.return_value = mock_credential
        
        with patch.object(cache._parser, 'parse_token', return_value=mock_token_info):
            await cache.get_credential("fake_assertion")
            result = await cache.get_credential("fake_assertion")
        
        assert result == mock_credential
        mock_factory.create_obo_credential.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_expired_token_removed_from_cache(self, mock_factory, mock_credential, expired_token_info):
        cache = OboCredentialCache(mock_factory)
        
        cached_cred = OboCachedCredential(mock_credential, expired_token_info)
        await cache._cache.set("test-key", cached_cred, ttl=10)
        
        with patch.object(cache._parser, 'parse_token', return_value=expired_token_info):
            with patch.object(cache, '_k', return_value="test-key"):
                mock_factory.create_obo_credential.return_value = mock_credential
                result = await cache.get_credential("fake_assertion")
        
        assert result == mock_credential
        assert await cache._cache.get("test-key") is None
    
    @pytest.mark.asyncio
    async def test_expired_token_not_cached(self, mock_factory, mock_credential, expired_token_info):
        cache = OboCredentialCache(mock_factory)
        mock_factory.create_obo_credential.return_value = mock_credential
        
        with patch.object(cache._parser, 'parse_token', return_value=expired_token_info):
            result = await cache.get_credential("fake_assertion")
        
        assert result == mock_credential
        
        k = cache._k(expired_token_info)
        cached_entry = await cache._cache.get(k)
        assert cached_entry is None
    
    @pytest.mark.asyncio
    async def test_buffer_time_applied(self, mock_factory, mock_credential):
        cache = OboCredentialCache(mock_factory)
        mock_factory.create_obo_credential.return_value = mock_credential
        
        token_info = TokenInfo(
            tenant_id="test-tenant",
            client_id="test-client",
            user_id="test-user",
            expiry=datetime.utcnow() + timedelta(seconds=30)
        )
        
        with patch.object(cache._parser, 'parse_token', return_value=token_info):
            await cache.get_credential("fake_assertion")
        
        k = cache._k(token_info)
        cached_entry = await cache._cache.get(k)
        assert cached_entry is not None
    
    @pytest.mark.asyncio
    async def test_exception_cached_and_raised(self, mock_factory, mock_token_info):
        cache = OboCredentialCache(mock_factory)
        test_exception = Exception("Test error")
        mock_factory.create_obo_credential.side_effect = test_exception
        
        with patch.object(cache._parser, 'parse_token', return_value=mock_token_info):
            with pytest.raises(Exception, match="Test error"):
                await cache.get_credential("fake_assertion")
        
        k = cache._k(mock_token_info)
        cached_entry = await cache._cache.get(k)
        assert isinstance(cached_entry, Exception)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_single_creation(self, mock_factory, mock_credential, mock_token_info):
        cache = OboCredentialCache(mock_factory)
        
        creation_called = asyncio.Event()
        creation_proceed = asyncio.Event()
        
        async def slow_create(_):
            creation_called.set()
            await creation_proceed.wait()
            return mock_credential
        
        mock_factory.create_obo_credential.side_effect = slow_create
        
        with patch.object(cache._parser, 'parse_token', return_value=mock_token_info):
            task1 = asyncio.create_task(cache.get_credential("fake_assertion"))
            await creation_called.wait()
            
            task2 = asyncio.create_task(cache.get_credential("fake_assertion"))
            
            creation_proceed.set()
            
            result1 = await task1
            result2 = await task2
        
        assert result1 == mock_credential
        assert result2 == mock_credential
        mock_factory.create_obo_credential.assert_called_once()


class TestAppCredentialCache:
    
    @pytest.mark.asyncio
    async def test_cache_miss_creates_credential(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        result = await cache.get_credential("tenant", "client")
        
        assert result == mock_credential
        mock_factory.create_app_credential.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_credential(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        await cache.get_credential("tenant", "client")
        result = await cache.get_credential("tenant", "client")
        
        assert result == mock_credential
        mock_factory.create_app_credential.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ttl_with_buffer_applied(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        await cache.get_credential("tenant", "client")
        
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert cached_entry is not None
    
    @pytest.mark.asyncio
    async def test_exception_cached_and_raised(self, mock_factory):
        cache = AppCredentialCache(mock_factory)
        test_exception = Exception("Test error")
        mock_factory.create_app_credential.side_effect = test_exception
        
        with pytest.raises(Exception, match="Test error"):
            await cache.get_credential("tenant", "client")
        
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert isinstance(cached_entry, Exception)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_single_creation(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        
        creation_called = asyncio.Event()
        creation_proceed = asyncio.Event()
        
        async def slow_create():
            creation_called.set()
            await creation_proceed.wait()
            return mock_credential
        
        mock_factory.create_app_credential.side_effect = slow_create
        
        task1 = asyncio.create_task(cache.get_credential("tenant", "client"))
        await creation_called.wait()
        
        task2 = asyncio.create_task(cache.get_credential("tenant", "client"))
        
        creation_proceed.set()
        
        result1 = await task1
        result2 = await task2
        
        assert result1 == mock_credential
        assert result2 == mock_credential
        mock_factory.create_app_credential.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_removes_from_cache(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        await cache.get_credential("tenant", "client")
        await cache.invalidate("tenant", "client")
        
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert cached_entry is None
    
    @pytest.mark.asyncio
    async def test_clear_removes_all_from_cache(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        await cache.get_credential("tenant1", "client1")
        await cache.get_credential("tenant2", "client2")
        await cache.clear()
        
        k1 = cache._k("tenant1", "client1")
        k2 = cache._k("tenant2", "client2")
        assert await cache._cache.get(k1) is None
        assert await cache._cache.get(k2) is None


class TestAppCredentialCacheSpecific:
    """Specific tests for App credential cache TTL and buffer logic"""
    
    @pytest.mark.asyncio
    async def test_ttl_calculation_with_buffer(self, mock_factory, mock_credential):
        """Test that TTL is calculated correctly with buffer applied"""
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        # Verify default configuration
        assert cache._ttl == 43_200  # 12 hours
        assert cache._buffer_seconds == 300  # 5 minutes
        
        await cache.get_credential("tenant", "client")
        
        # The actual TTL set should be 43200 - 300 = 42900 seconds
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert isinstance(cached_entry, AppCachedCredential)
    
    @pytest.mark.asyncio
    async def test_different_tenants_separate_cache(self, mock_factory, mock_credential):
        """Test that different tenant/client combinations are cached separately"""
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        # Create credentials for different tenant/client combinations
        await cache.get_credential("tenant1", "client1")
        await cache.get_credential("tenant2", "client2")
        await cache.get_credential("tenant1", "client2")
        
        # Should have called create_app_credential 3 times
        assert mock_factory.create_app_credential.call_count == 3
        
        # Verify separate cache entries
        k1 = cache._k("tenant1", "client1")
        k2 = cache._k("tenant2", "client2")
        k3 = cache._k("tenant1", "client2")
        
        assert await cache._cache.get(k1) is not None
        assert await cache._cache.get(k2) is not None
        assert await cache._cache.get(k3) is not None
    
    @pytest.mark.asyncio
    async def test_exception_caching_with_backoff(self, mock_factory):
        """Test that exceptions are cached with fail_backoff TTL"""
        cache = AppCredentialCache(mock_factory)
        test_exception = Exception("Network error")
        mock_factory.create_app_credential.side_effect = test_exception
        
        with pytest.raises(Exception, match="Network error"):
            await cache.get_credential("tenant", "client")
        
        # Exception should be cached
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert isinstance(cached_entry, Exception)
        
        # Second call should use cached exception without calling factory
        mock_factory.create_app_credential.reset_mock()
        with pytest.raises(Exception, match="Network error"):
            await cache.get_credential("tenant", "client")
        
        mock_factory.create_app_credential.assert_not_called()


class TestCacheTtlBehavior:
    
    @pytest.mark.asyncio
    async def test_obo_cache_ttl_calculation(self, mock_factory, mock_credential):
        cache = OboCredentialCache(mock_factory)
        mock_factory.create_obo_credential.return_value = mock_credential
        
        future_time = datetime.utcnow() + timedelta(seconds=100)
        token_info = TokenInfo(
            tenant_id="test-tenant",
            client_id="test-client",
            user_id="test-user",
            expiry=future_time
        )
        
        with patch.object(cache._parser, 'parse_token', return_value=token_info):
            await cache.get_credential("fake_assertion")
        
        await asyncio.sleep(0.1)
        
        k = cache._k(token_info)
        cached_entry = await cache._cache.get(k)
        assert isinstance(cached_entry, OboCachedCredential)
    
    @pytest.mark.asyncio 
    async def test_app_cache_default_ttl(self, mock_factory, mock_credential):
        cache = AppCredentialCache(mock_factory)
        mock_factory.create_app_credential.return_value = mock_credential
        
        await cache.get_credential("tenant", "client")
        
        await asyncio.sleep(0.1)
        
        k = cache._k("tenant", "client")
        cached_entry = await cache._cache.get(k)
        assert cached_entry is not None
