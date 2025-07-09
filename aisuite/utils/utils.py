import json
from pydantic import BaseModel
from unittest.mock import MagicMock

class Utils:
    """
    Utility functions for debugging and inspecting objects.
    """
    @staticmethod
    def spew(obj):
        """
        Recursively inspects a Python object and prints its contents as a
        nicely formatted JSON string. Handles Pydantic models, nested objects,
        lists, and circular references.
        """
        visited = set()

        def default_encoder(o):
            # Handle MagicMock objects to prevent circular reference errors in tests
            if isinstance(o, MagicMock):
                try:
                    # Attempt to get a descriptive name for the mock
                    name = o._extract_mock_name()
                except Exception:
                    name = "unknown"
                return f'<MagicMock name="{name}">' 

            # Handle other circular references
            obj_id = id(o)
            if obj_id in visited:
                return f'<Circular reference to {type(o).__name__} at {obj_id}>'
            visited.add(obj_id)

            # Handle Pydantic models
            if isinstance(o, BaseModel):
                return o.model_dump()
            
            # Handle general objects by converting their __dict__
            if hasattr(o, '__dict__'):
                return o.__dict__
            
            # Handle sets
            if isinstance(o, set):
                return list(o)

            # Fallback for other types
            try:
                return str(o)
            except Exception:
                return f"<Unserializable: {type(o).__name__}>"

        print(json.dumps(obj, default=default_encoder, indent=2))