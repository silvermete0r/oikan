class OikanError(Exception):
    """Base exception class for OIKAN library."""
    pass

class InitializationError(OikanError):
    """Raised when model initialization fails."""
    pass

class DimensionalityError(OikanError):
    """Raised when input/output dimensions are incompatible."""
    pass

class BasisError(OikanError):
    """Raised when there are issues with basis functions."""
    pass

class DataTypeError(OikanError):
    """Raised when data types are incompatible."""
    pass

class DeviceError(OikanError):
    """Raised when there are device-related issues."""
    pass

class VisualizationError(OikanError):
    """Raised when formula visualization becomes too complex to be meaningful."""
    pass
