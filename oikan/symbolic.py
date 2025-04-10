from .utils import ADVANCED_LIB

def symbolic_edge_repr(weights, bias=None, threshold=1e-4):
    """
    Given a list of weights (floats) and an optional bias,
    returns a list of structured terms (coefficient, basis function string).
    """
    terms = []
    # weights should be in the same order as ADVANCED_LIB.items()
    for (_, (notation, _)), w in zip(ADVANCED_LIB.items(), weights):
        if abs(w) > threshold:
            terms.append((w, notation))
    if bias is not None and abs(bias) > threshold:
        # use "1" to represent the constant term
        terms.append((bias, "1"))
    return terms

def format_symbolic_terms(terms):
    """
    Formats a list of structured symbolic terms (coef, basis) to a string.
    """
    formatted_terms = []
    for coef, basis in terms:
        if basis == "1":
            formatted_terms.append(f"{coef:.4f}")
        else:
            formatted_terms.append(f"{coef:.4f}*{basis}")
    return " + ".join(formatted_terms) if formatted_terms else "0"
