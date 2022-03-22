#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <math.h>

namespace py = pybind11;

using namespace Eigen;
using namespace std;

VectorXd lassoShooting(MatrixXd X, VectorXd y, MatrixXd XX, VectorXd Xy,
                       double lambd, VectorXd psi, VectorXd startingValues,
                       bool sqrtLasso = false, double optTol = 1e-10,
                       int maxIter = 1000, double zeroTol = 1e-4) {

  int n = X.rows(), p = X.cols();
  VectorXd beta = startingValues;


  // normal lasso shooting
  if (!sqrtLasso) {

    XX *= 2, Xy *= 2;
    // loop over max iter
    for (int i = 0; i < maxIter; i++) {
      // copy beta to beta_old
      VectorXd betaOld = beta;
      // loop over p
      for (int j = 0; j < p; j++) {
        // calculate s0 and do shooting
        double s0 = (XX.row(j) * beta).sum() - XX(j, j) * beta(j) - Xy(j);

        if (s0 > lambd * psi(j)) {
          beta(j) = (lambd * psi(j) - s0) / XX(j, j);

        } else if (s0 < -lambd * psi(j)) {
          beta(j) = (-lambd * psi(j) - s0) / XX(j, j);

        } else {
          beta(j) = 0;
        }
      }
      // check for convergence
      double diff = (beta - betaOld).norm();
      if (diff < optTol) {
        break;
      }
    }
    // sqrt-lasso shooting algorithm
  } else {
    // rescale XX and Xy
    XX /= n, Xy /= n;
    double MaxErrorNorm = 1.0e-10;

    // get error
    VectorXd error = y - X * beta;
    double qhat = error.squaredNorm() / n;

    for (int i = 0; i < maxIter; i++) {

      VectorXd betaOld = beta;
      for (int j = 0; j < p; j++) {

        if (fabs(beta(j)) > 0) {
          error += X.col(j) * beta(j);
          qhat = error.squaredNorm() / n;
        }

        double S0 = (XX.row(j) * beta).sum() - XX(j, j) * beta(j) - Xy(j);
        double qqhat = max(qhat - (pow(S0, 2) / XX(j, j)), 0.0);

        if (pow(n, 2) < pow(lambd * psi(j), 2) / XX(j, j)) {
          beta(j) = 0;
        }

        else if (S0 > lambd / n * psi(j) * sqrt(qhat)) {
          beta[j] = ((lambd * psi(j) /
                      sqrt(pow(n, 2) - pow(lambd * psi(j), 2) / XX(j, j))) *
                         sqrt(qqhat) -
                     S0) /
                    XX(j, j);
          error -= X.col(j) * beta[j];
        }

        else if (S0 < -lambd / n * psi(j) * sqrt(qhat)) { // Optimal beta(j) > 0
          beta[j] = (-(lambd * psi(j) /
                       sqrt(pow(n, 2) - pow(lambd * psi(j), 2) / XX(j, j))) *
                         sqrt(qqhat) -
                     S0) /
                    XX(j, j);
          error = -X.col(j) * beta(j);
        }

        else {
          beta(j) = 0;
        }
      } // end loop beta^(i)_j

      // Update primal and dual value
      double ErrorNorm = (y - X * beta).squaredNorm();
      double fobj =
          ErrorNorm / sqrt(n) + (lambd / n) * (psi * beta.cwiseAbs()).sum();

      double dual;
      if (ErrorNorm > MaxErrorNorm) {
        VectorXd aaa = (sqrt(n) * error / ErrorNorm);
        VectorXd bbb = (lambd / n) * psi - (X.transpose() * aaa / n).cwiseAbs();
        dual = aaa.transpose() * bbb;
      } else {
        dual = (lambd / n) * (psi * beta.cwiseAbs()).sum();
      }

      if (fobj - dual < optTol) {
        if ((beta - betaOld).norm() < optTol) {
          break;
        }
      }
    }
  }
  // set beta to zero below threshold
  for (int j = 0; j < p; j++) {
    if (abs(beta(j)) < zeroTol) {
      beta(j) = 0;
    }
  }
  return beta;
}

// docstring
string docstring = R"mydelimiter(
    "Lasso Shooting algorithm for and sqrt lasso."
    Parameters
    ----------
    X : numpy array
        design matrix
    y : numpy array
        response vector
    XX : ndarray
      cross product matrix of X
    Xy : ndarray
      cross product vector of X and y
    lambd : float
        Regularization parameter.
    psi : float
        Penalty loadings.
    beta_start : ndarray, optional, default: None
        Initial beta estimate.
    sqrt : bool, optional, default False
        If True, use sqrt lasso.
        beta = min ||(y - X @ beta)||_2^2 + lambd ||psi @ beta||_1
    max_iter : int, optional, default: 1000
        Maximum number of iterations.
    opt_tol : float, optional, default: 1e-10
        Optimality tolerance.
    zero_tol : float, optional, default: 1e-4
        Zero tolerance.
    Returns
    -------
    beta : ndarray
        Estimated beta.
)mydelimiter";


PYBIND11_MODULE(solver_fast, m) {
  // pybind11::module m("code", "auto-compiled c++ extension");
  py::options options;
  options.disable_function_signatures();
  m.doc() = "Coordinate descent solver for lasso and sqrt-lasso";
  m.def("lasso_shooting", &lassoShooting,
        docstring.c_str(),
        py::arg("X").noconvert() = NULL,
        py::arg("y").noconvert() = NULL,
        py::arg("XX").noconvert() = NULL,
        py::arg("Xy").noconvert() = NULL,
        py::arg("lambd").noconvert() = NULL,
        py::arg("psi").noconvert() = NULL,
        py::arg("starting_values").noconvert() = NULL,
        py::arg("sqrt") = false,
        py::arg("opt_tol") = 1e-10, 
        py::arg("max_iter") = 1000,
        py::arg("zero_tol") = 1e-4);
};
