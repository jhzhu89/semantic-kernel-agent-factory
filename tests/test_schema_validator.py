import pytest
from agent_factory.utils import OpenAISchemaValidator, OpenAISchemaValidationError


class TestOpenAISchemaValidator:
    
    @pytest.fixture
    def validator(self):
        return OpenAISchemaValidator()

    def test_valid_simple_schema(self, validator):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        validator.validate_schema(schema)

    def test_valid_nested_schema(self, validator):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        validator.validate_schema(schema)

    def test_invalid_non_dict_schema(self, validator):
        with pytest.raises(OpenAISchemaValidationError, match="Schema must be a dictionary"):
            validator.validate_schema("not a dict")

    def test_invalid_unsupported_type(self, validator):
        schema = {
            "type": "unknown_type",
            "properties": {}
        }
        
        with pytest.raises(OpenAISchemaValidationError):
            validator.validate_schema(schema)

    def test_invalid_unsupported_keyword(self, validator):
        schema = {
            "type": "string",
            "minLength": 5
        }
        
        with pytest.raises(OpenAISchemaValidationError):
            validator.validate_schema(schema)

    def test_max_nesting_depth_exceeded(self, validator):
        schema = {"type": "object", "properties": {}}
        current = schema
        
        for i in range(10):
            current["properties"]["nested"] = {
                "type": "object",
                "properties": {}
            }
            current = current["properties"]["nested"]
        
        with pytest.raises(OpenAISchemaValidationError, match="Maximum nesting depth exceeded"):
            validator.validate_schema(schema)

    def test_too_many_properties(self, validator):
        properties = {}
        for i in range(150):
            properties[f"prop_{i}"] = {"type": "string"}
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        with pytest.raises(OpenAISchemaValidationError):
            validator.validate_schema(schema)

    def test_valid_array_schema(self, validator):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        }
        
        validator.validate_schema(schema)

    def test_supported_types(self, validator):
        for schema_type in validator.SUPPORTED_TYPES:
            schema = {"type": schema_type}
            if schema_type == "object":
                schema["properties"] = {}
            elif schema_type == "array":
                schema["items"] = {"type": "string"}
            
            validator.validate_schema(schema)

    def test_multiple_types(self, validator):
        schema = {
            "type": ["string", "null"]
        }
        
        validator.validate_schema(schema)

    def test_enum_values(self, validator):
        schema = {
            "type": "string",
            "enum": ["option1", "option2", "option3"]
        }
        
        validator.validate_schema(schema)

    def test_const_value(self, validator):
        schema = {
            "type": "string",
            "const": "fixed_value"
        }
        
        validator.validate_schema(schema)

    def test_schema_with_description(self, validator):
        schema = {
            "type": "object",
            "description": "A test object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "A test field"
                }
            }
        }
        
        validator.validate_schema(schema)

    def test_additionalProperties_false(self, validator):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": False
        }
        
        validator.validate_schema(schema)

    def test_clear_definitions_cache(self, validator):
        schema = {"type": "string"}
        
        validator.validate_schema(schema, "test1")
        validator.validate_schema(schema, "test2")
        
        assert len(validator._definitions_cache) == 0

    def test_complex_valid_schema(self, validator):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "settings": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string", "enum": ["light", "dark"]},
                                "notifications": {"type": "boolean"}
                            },
                            "additionalProperties": False
                        }
                    },
                    "required": ["id", "name"]
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"},
                        "version": {"type": "number"}
                    }
                }
            },
            "required": ["user"],
            "additionalProperties": False
        }
        
        validator.validate_schema(schema)
