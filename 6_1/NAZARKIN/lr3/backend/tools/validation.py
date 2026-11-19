"""Small JSON-schema validator for the tool boundary.

The project intentionally avoids a dependency on jsonschema/pydantic: the
schemas are simple and fully controlled by the backend.  Validation failures are
turned into structured errors by the tool executor.
"""
from __future__ import annotations

from typing import Any, Dict, List


class ValidationError(ValueError):
    def __init__(self, message: str, path: str = ""):
        super().__init__(message)
        self.path = path

    def as_dict(self) -> Dict[str, Any]:
        return {"code": "VALIDATION_ERROR", "message": str(self),
                "details": {"path": self.path} if self.path else {}}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate(value: Any, schema: Dict[str, Any], path: str = "") -> None:
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        raise ValidationError(f"expected {expected}", path or "$")

    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(
            f"must be one of {schema['enum']}", path or "$")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"must have at least {schema['minLength']} characters", path or "$")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"must have at most {schema['maxLength']} characters", path or "$")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"must be >= {schema['minimum']}", path or "$")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"must be <= {schema['maximum']}", path or "$")

    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValidationError(f"must have at most {schema['maxItems']} items", path or "$")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate(item, item_schema, f"{path}[{index}]")

    if isinstance(value, dict):
        properties: Dict[str, Any] = schema.get("properties") or {}
        required: List[str] = schema.get("required") or []
        for key in required:
            if key not in value:
                raise ValidationError(f"missing required field {key!r}", path or "$")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValidationError(f"unknown fields: {', '.join(unknown)}", path or "$")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}" if path else key)
