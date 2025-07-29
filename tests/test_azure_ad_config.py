import pytest
from agent_factory.mcp_server.config import AzureAdConfig


class TestAzureAdConfig:
    
    def test_azure_ad_config_defaults(self):
        config = AzureAdConfig(
            tenant_id="test-tenant-id",
            client_id="test-client-id"
        )
        
        assert config.certificate_pem is None
        assert config.client_secret is None

    def test_azure_ad_config_with_certificate_pem_basic(self):
        cert_pem = "-----BEGIN CERTIFICATE-----\nMIICert...\n-----END CERTIFICATE-----"
        config = AzureAdConfig(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            certificate_pem=cert_pem
        )
        
        assert config.certificate_pem == cert_pem
        assert config.client_secret is None

    def test_azure_ad_config_with_certificate_pem(self):
        cert_pem = "-----BEGIN CERTIFICATE-----\nMIICert...\n-----END CERTIFICATE-----"
        
        config = AzureAdConfig(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            certificate_pem=cert_pem
        )
        
        assert config.certificate_pem == cert_pem
        assert config.client_secret is None

    def test_azure_ad_config_with_client_secret(self):
        config = AzureAdConfig(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="secret-789"
        )
        
        assert config.certificate_pem is None
        assert config.client_secret == "secret-789"

    def test_azure_ad_config_serialization(self):
        config = AzureAdConfig(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            certificate_pem="-----BEGIN CERTIFICATE-----\ntest-cert\n-----END CERTIFICATE-----"
        )
        
        config_dict = config.model_dump()
        
        assert config_dict["certificate_pem"] == "-----BEGIN CERTIFICATE-----\ntest-cert\n-----END CERTIFICATE-----"
        assert config_dict["client_secret"] is None

    def test_azure_ad_config_from_dict(self):
        data = {
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
            "certificate_pem": "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----"
        }
        
        config = AzureAdConfig(**data)
        
        assert config.certificate_pem == "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----"
        assert config.client_secret is None
