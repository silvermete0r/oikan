class OIKANError(Exception):
    """Base exception for OIKAN library."""
    pass

class ModelNotFittedError(OIKANError):
    """Raised when a method requires a fitted model."""
    pass

class InvalidParameterError(OIKANError):
    """Raised when an invalid parameter value is provided."""
    pass

class DataDimensionError(OIKANError):
    """Raised when input data has incorrect dimensions."""
    pass

class NumericalInstabilityError(OIKANError):
    """Raised when numerical computations become unstable."""
    pass

class FeatureExtractionError(OIKANError):
    """Raised when feature extraction or transformation fails."""
    pass

class ModelSerializationError(OIKANError):
    """Raised when model saving/loading operations fail."""
    pass

class ConvergenceError(OIKANError):
    """Raised when the model fails to converge during training."""
    pass    