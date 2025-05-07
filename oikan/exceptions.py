class OIKANError(Exception):
    """Base exception for OIKAN library."""
    pass

class ModelNotFittedError(OIKANError):
    """Raised when a method requires a fitted model."""
    pass