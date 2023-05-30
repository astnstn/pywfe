from collections import namedtuple

Eigensolution = namedtuple("Eigensolution", ["lambda_plus", "lambda_minus",
                                             "phi_plus", "phi_minus",
                                             "psi_plus", "psi_minus"])

Boundaries= namedtuple("Boundaries", ["A_right", "B_right", "A_left", "B_left"])