#########################################################################################
# PACKAGES
#########################################################################################
from LapMatPred.myUtilities import extract_block, update_block_matrix

import numpy as np
import cvxpy as cp


#########################################################################################
# CLASSES
#########################################################################################
class ConvexProblem:
    def __init__(self, variable, parameters, constraints, cost, minimize=True):
        """
        __init__ Initializes a ConvexProblem object, containing variable, parameters,
        and problem.

        Parameters
        ----------
        variable: cvxpy.Variable
            The variable of the problem
        parameters: list(cvxpy.Parameter)
            A list of parameters of the problem
        constraints: list(cvxpy.constraints)
            A list containing the constraints of the problem
        cost: cvxpy.Expression
            An expression representing the cost of the problem
        minimize: bool, default to True
            If True the cvxpy.Problem instance that will be created will minimize the
            cost subject to constraints, otherwise it will maximize the cost subject to
            the constraints
        """
        self.variable = variable
        self.parameters = parameters

        if minimize:
            self.problem = cp.Problem(cp.Minimize(cost), constraints)
        else:
            self.problem = cp.Problem(cp.Maximize(cost), constraints)

    def set_parameters(self, parameters):
        """
        setParameters set the value for the problem's parameters, after checking if the
        length of the list of values is the same as the length of self.parameters

        Parameters
        ----------
        parameters: list(numpy.ndarray)
            The list of values to assign to the problem's parameters, shapes must
            correspond with the shapes of self.parameters list elements

        Returns
        -------
        None
        """
        if len(parameters) == len(self.parameters):
            for old, new in zip(self.parameters, parameters):
                old.value = new
        else:
            print(
                "WARNING: Number of parameters does not equal number of parameters of the problem, parameters not set."
            )

    def solve_problem(self, return_solution=True, verbose=False):
        """
        solveProblem calls the solve method on self.problem

        Parameters
        ----------
        return_solution: bool, default to True
            If True the function returns the variable value (solution)
        verbose: bool, default to False
            If True set the solver to verbose mode

        Returns
        -------
        numpy.ndarray
            The problem's solution
        """
        self.problem.solve(solver="CVXOPT", verbose=verbose)

        if return_solution:
            return self.variable.value


################################################################################
# FUNCTIONS
################################################################################
def relative_error(m, est, m_norm=None):
    """
    relativeError computes the relative error between matrix m and its estimation est
    ||m-est|| / ||m|| using Frobenius norm.

    Parameters
    ----------
    m: numpy.ndarray, shape=(n, n)
        The true matrix
    est: numpy.ndarray, shape=(None, n, n)
        A matrix or stack of matrices representing estimates of matrix m. If est.ndim
        == 3 then m is broadcast accordingly to compute the relative error for each of
        the estimates
    m_norm: float, default to None
        The norm of matrix m, if None it is computed using the Frobenius norm

    Returns
    -------
    int
        The relative error between m and estimate est
    """
    order = "fro" if len(m.shape) == 2 else None

    if m_norm is None:
        m_norm = np.linalg.norm(m, ord=order)

    return np.linalg.norm(m - est, ord=order) / m_norm


def coordinate_descent(k, problem, stop_crit=5, max_iter=50, tol=1e-7):
    """
    coordinateDescent estimates the matrix q, a generalized laplacian precision matrix,
    with a coordinate descent algorithm based on convex optimization starting from
    q(0) = diag(k)^-1

    Parameters
    ----------
    k: numpy.ndarray, shape=(n, n)
        An empirical covariance matrix
    problem: ConvexProblem
        A ConvexProblem object containing a cvxpy. Problem and its related variables,
        parameters and constraints
    stop_crit: float, default to 1e-2
        The algorithm iterates until convergence, defined when the maximum variation
        between two iterations is less than stop_crit. The variation is defined as the
        maximum of the absolute value of the difference between the current and previous
        iteration matrix Q. The iteration in this case is defined as an update loop
        over all the rows/columns of Q.
    max_iter: int, default to 20
        The maximum number of iterations for the algorithm
    tol: float, default to 1e-7
        Reference for Lagrange multipliers, used to define when they are considered 0

    Returns
    -------

    """
    q = np.diag(1 / np.diag(k))
    q_new = q

    iterations = 0

    delta = stop_crit + 1
    while delta > stop_crit and iterations < max_iter:
        iterations += 1

        for col in range(k.shape[0]):
            q_block = extract_block(q_new, col)
            k_col = extract_block(k[:, col], col)
            k_scal = k[col, col]

            problem.set_parameters([q_block, k_col[:, np.newaxis]])
            lamb = problem.solve_problem()
            q_col = -np.dot(q_block, k_col + lamb.squeeze()) / k_scal

            # Set values of q_col with Lagrange multiplier greater than 0 to 0
            q_col[lamb.squeeze() > tol] = 0
            q_scal = (1 - np.dot(q_col, k_col)) / k_scal

            q_new = update_block_matrix(q_block, q_col, q_scal, col)

        delta = np.absolute(q - q_new).max()

        q = q_new

    return q, iterations, delta


def initialize_nnqp_convex_problem(nodes_number):
    """
    Initializes a nnqp convex problem to estimate the GLP with Ortega's method, for graphs with nodes_number nodes.

    Args:
        nodes_number (int): The number of nodes in the graphs.

    Returns:
        ConvexProblem: An NNQP Convex problem to be solved with Ortega's method.
    """
    variable = cp.Variable((nodes_number - 1, 1))
    param = [
        cp.Parameter((nodes_number - 1, nodes_number - 1), PSD=True),
        cp.Parameter((nodes_number - 1, 1)),
    ]
    constraints = [variable >= 0]
    cost = cp.quad_form((variable + param[1]), param[0])

    return ConvexProblem(variable, param, constraints, cost)
