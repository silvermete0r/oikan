class OikanError(Exception):
    """Base exception class for OIKAN"""
    pass

class NotFittedError(OikanError):
    """Raised when prediction is attempted on unfitted model"""
    pass

class DataError(OikanError):
    """Raised when there are issues with input data"""
    pass

class InitializationError(OikanError):
    """Raised when model initialization fails"""
    pass
