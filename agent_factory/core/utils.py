from __future__ import annotations

from typing import Any, Dict, Set


class OpenAISchemaValidationError(Exception):
    pass


class OpenAISchemaValidator:
    SUPPORTED_TYPES = {"string", "number", "boolean", "integer", "object", "array", "null"}
    MAX_NESTING_DEPTH = 5
    MAX_OBJECT_PROPERTIES = 100

    UNSUPPORTED_KEYWORDS = {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
    }

    def __init__(self):
        self._definitions_cache: Set[str] = set()

    def validate_schema(self, schema: Dict[str, Any], name: str = "schema") -> None:
        if not isinstance(schema, dict):
            raise OpenAISchemaValidationError(f"Schema must be a dictionary, got {type(schema)}")

        self._definitions_cache.clear()
        self._validate_recursive(schema, 0, name)

    def _validate_recursive(self, schema: Dict[str, Any], depth: int, path: str) -> None:
        if depth > self.MAX_NESTING_DEPTH:
            raise OpenAISchemaValidationError(f"Maximum nesting depth exceeded at {path}")

        schema_type = schema.get("type")
        if schema_type:
            # Handle both single type (string) and multiple types (array)
            if isinstance(schema_type, list):
                # Multiple types - validate each one
                for single_type in schema_type:
                    if single_type not in self.SUPPORTED_TYPES:
                        raise OpenAISchemaValidationError(
                            f"Unsupported type '{single_type}' at {path}"
                        )
            else:
                # Single type - validate directly
                if schema_type not in self.SUPPORTED_TYPES:
                    raise OpenAISchemaValidationError(f"Unsupported type '{schema_type}' at {path}")

        for keyword in self.UNSUPPORTED_KEYWORDS:
            if keyword in schema:
                raise OpenAISchemaValidationError(f"Unsupported keyword '{keyword}' at {path}")

        # Handle specific type validations
        # Check if schema_type contains "object" (either as string or in list)
        if (isinstance(schema_type, str) and schema_type == "object") or (
            isinstance(schema_type, list) and "object" in schema_type
        ):
            self._validate_object(schema, depth, path)

        # Check if schema_type contains "array" (either as string or in list)
        if (
            (isinstance(schema_type, str) and schema_type == "array")
            or (isinstance(schema_type, list) and "array" in schema_type)
        ) and "items" in schema:
            self._validate_recursive(schema["items"], depth + 1, f"{path}[]")

        if "$defs" in schema:
            self._validate_definitions(schema["$defs"], depth, path)

    def _validate_object(self, schema: Dict[str, Any], depth: int, path: str) -> None:
        properties = schema.get("properties", {})
        if len(properties) > self.MAX_OBJECT_PROPERTIES:
            raise OpenAISchemaValidationError(f"Too many properties at {path}")

        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                raise OpenAISchemaValidationError(
                    f"Property '{prop_name}' schema must be a dictionary at {path}"
                )
            self._validate_recursive(prop_schema, depth + 1, f"{path}.{prop_name}")

    def _validate_definitions(self, definitions: Dict[str, Any], depth: int, path: str) -> None:
        for def_name, def_schema in definitions.items():
            if def_name in self._definitions_cache:
                continue
            self._definitions_cache.add(def_name)

            if not isinstance(def_schema, dict):
                raise OpenAISchemaValidationError(
                    f"Definition '{def_name}' must be a dictionary at {path}"
                )
            self._validate_recursive(def_schema, depth + 1, f"{path}.$defs.{def_name}")


__all__ = ["OpenAISchemaValidator", "OpenAISchemaValidationError"]
