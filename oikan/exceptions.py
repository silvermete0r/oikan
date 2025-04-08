class OikanError(Exception):
    """Base exception class for all OIKAN-related errors."""
    def __init__(self, message="An error occurred in OIKAN"):
        self.message = message
        super().__init__(self.message)

class InitializationError(OikanError):
    """Raised when model initialization fails."""
    def __init__(self, message="Failed to initialize OIKAN model", details=None):
        self.details = details
        full_message = f"{message}: {details}" if details else message
        super().__init__(full_message)

class DimensionalityError(OikanError):
    """Raised when input/output dimensions are incompatible."""
    def __init__(self, expected_dim, actual_dim, context=""):
        message = f"Dimension mismatch{' in ' + context if context else ''}: "
        message += f"Expected {expected_dim}, got {actual_dim}"
        super().__init__(message)

class BasisError(OikanError):
    """Raised when there are issues with basis functions."""
    def __init__(self, basis_name=None, error_type=None, details=None):
        message = "Basis function error"
        if basis_name:
            message += f" in {basis_name}"
        if error_type:
            message += f": {error_type}"
        if details:
            message += f" - {details}"
        super().__init__(message)

class DataTypeError(OikanError):
    """Raised when data types are incompatible."""
    def __init__(self, expected_type, actual_type, variable_name=None):
        message = f"Invalid data type"
        if variable_name:
            message += f" for {variable_name}"
        message += f": Expected {expected_type}, got {actual_type}"
        super().__init__(message)

class DeviceError(OikanError):
    """Raised when there are device-related issues."""
    def __init__(self, operation=None, device=None, error_details=None):
        message = "Device error"
        if operation:
            message += f" during {operation}"
        if device:
            message += f" on {device}"
        if error_details:
            message += f": {error_details}"
        super().__init__(message)

class VisualizationError(OikanError):
    """Raised when formula visualization becomes too complex."""
    def __init__(self, complexity=None, max_allowed=None, suggestion=None):
        message = "Formula visualization error"
        if complexity and max_allowed:
            message += f": Complexity {complexity} exceeds maximum allowed {max_allowed}"
        if suggestion:
            message += f"\nSuggestion: {suggestion}"
        super().__init__(message)

class FormulaExtractionError(OikanError):
    """Raised when symbolic formula extraction fails."""
    def __init__(self, error_type=None, details=None):
        message = "Failed to extract symbolic formula"
        if error_type:
            message += f": {error_type}"
        if details:
            message += f" - {details}"
        super().__init__(message)

class TrainingError(OikanError):
    """Raised when model training encounters issues."""
    def __init__(self, stage=None, error_type=None, details=None):
        message = "Training error"
        if stage:
            message += f" at {stage}"
        if error_type:
            message += f": {error_type}"
        if details:
            message += f" - {details}"
        super().__init__(message)

class RegularizationError(OikanError):
    """Raised when regularization computation fails."""
    def __init__(self, reg_type=None, details=None):
        message = "Regularization error"
        if reg_type:
            message += f" in {reg_type} regularization"
        if details:
            message += f": {details}"
        super().__init__(message)

class IncompatibleModelError(OikanError):
    """Raised when attempting operations on incompatible model configurations."""
    def __init__(self, operation=None, expected_config=None, actual_config=None):
        message = "Incompatible model configuration"
        if operation:
            message += f" for {operation}"
        if expected_config and actual_config:
            message += f": Expected {expected_config}, got {actual_config}"
        super().__init__(message)

def handle_oikan_error(func):
    """Decorator for consistent error handling across OIKAN operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OikanError as e:
            # Re-raise OIKAN-specific errors
            raise
        except ValueError as e:
            raise DataTypeError(str(e), None, func.__name__)
        except RuntimeError as e:
            raise DeviceError(func.__name__, None, str(e))
        except Exception as e:
            raise OikanError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper
