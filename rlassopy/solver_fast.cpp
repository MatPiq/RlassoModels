// adapted from lassopack, see:
// https://github.com/statalasso/lassopack/blob/master/lassoutils.ado
//
#include "../extern/eigen-3.4.0/Eigen/Eigen"
// #include "../extern/eigen-3.4.0/unsupported/Eigen/MatrixFunctions"

#include <iostream>
#include <math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Eigen>
// #include <unsupported/Eigen/MatrixFunctions>

namespace py = pybind11;
using namespace Eigen;
using namespace std;
// using MatrixXdRef = Eigen::Ref<Eigen::MatrixXd>;
// using VectorXdRef = Eigen::Ref<Eigen::VectorXd>;
using MatrixXdRef = Eigen::MatrixXd;
using VectorXdRef = Eigen::VectorXd;


VectorXd lassoShooting(MatrixXdRef X, VectorXdRef y, MatrixXdRef XX, VectorXdRef Xy,
                       double lambd, VectorXdRef psi, VectorXdRef startingValues,
                       bool sqrtLasso = false, bool fitIntercept = true,
                       double optTol = 1e-10, int maxIter = 1000,
                       double zeroTol = 1e-4) {

  int n = X.rows(), p = X.cols();
 
  VectorXd beta = startingValues;
  
  VectorXd sdVec = X.colwise().norm();
  double ySd = y.norm();
  // normal lasso shooting
  if (!sqrtLasso) {

    XX *= 2, Xy *= 2;
    // loop over max iter
    for (int iter = 0; iter < maxIter; iter++) {
      // copy beta to beta_old
      VectorXd betaOld = beta;
      // loop over p
      for (int j = 0; j < p; j++) {
        // calculate s0 and do shooting
        double S0 = (XX.row(j) * beta).sum() - XX(j, j) * beta(j) - Xy(j);

        if (S0 > lambd * psi(j)) {
          beta(j) = (lambd * psi(j) - S0) / XX(j, j);

        } else if (S0 < -lambd * psi(j)) {
          beta(j) = (-lambd * psi(j) - S0) / XX(j, j);

        } else {
          beta(j) = 0;
        }
      }
      // check for convergence
      double diff = ((beta - betaOld).cwiseAbs() * sdVec / ySd).sum();
      // double diff = (beta - betaOld).cwiseAbs().sum();
      if (diff < optTol) {
        break;
      }
    }
    // sqrt-lasso shooting algorithm
  } else {
    // rescale XX and Xy
    XX /= n;
    Xy /= n;
    double MaxErrorNorm = 1.0e-10;

    // get means of X and y for intercept handling
    if (fitIntercept) {
      // deamean X and y
      // repeat for n rows
      MatrixXd meanX = X.colwise().mean().replicate(n, 1);
      VectorXd meanY = VectorXd::Ones(n) * y.mean();
      X = X - meanX;
      y = y - meanY;
    }

    // get error
    VectorXd error = y - X * beta;
    double qhat = error.squaredNorm() / n;

    
    for (int iter = 0; iter < maxIter; iter++) {
      VectorXd betaOld = beta;

      for (int j = 0; j < p; j++) {

        if (fabs(beta(j)) > 0) {
          error += X.col(j) * beta(j);
          qhat = error.squaredNorm() / n;
        }

        double S0 = XX.row(j).dot(beta) - XX(j, j) * beta(j) - Xy(j);
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
          error -= X.col(j) * beta(j);
        }

        else {
          beta(j) = 0;
        }
      } // end loop beta^(i)_j

      // Update primal and dual value
      double errorNorm = (y - X * beta).norm();
      double fobj =
          errorNorm / sqrt(n) + (lambd / n) * (psi * beta.cwiseAbs()).sum();

      double dual;
      if (errorNorm > MaxErrorNorm) {
        VectorXd aaa = sqrt(n) * (error / errorNorm);
        double bbb = ((lambd / n) * psi - (X.transpose() * aaa / n).cwiseAbs()).cwiseAbs().transpose() * beta.cwiseAbs();
        dual = aaa.transpose() * (y / n) - bbb;
      } else {
        dual = (lambd / n) * (psi * beta.cwiseAbs()).sum();
      }
      
      // check for convergence
      // double diff = (beta - betaOld).cwiseAbs().sum();
      double diff = ((beta - betaOld).cwiseAbs() * (sdVec / ySd)).sum();
      if (diff < optTol) {
        // if ((fobj - dual)  < 1e-6) {
        if ((fobj - dual) / ySd < optTol) {
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
    starting_values : ndarray, optional, default: None
        Initial beta estimate.
    sqrt : bool, optional, default False
        If True, use sqrt lasso.
        beta = min ||(y - X @ beta)||_2^2 + lambd ||psi @ beta||_1
    fit_intercept : bool, optional, default true
        If True, fit intercept. Only relevant for sqrt lasso.
    max_iter : int, optional, default: 1000
        Maximum number of iterations.
    opt_tol : float, optional, default: 1e-10
        Optimality tolerance.
    zero_tol : float, optional, default: 1e-4
        Zero tolerance. If beta(j) is smaller than zero_tol, 
        set beta(j) = 0.
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
  m.def("lasso_shooting", &lassoShooting, docstring.c_str(),
        py::arg("X").noconvert() = NULL, 
        py::arg("y").noconvert() = NULL,
        py::arg("XX").noconvert() = NULL, 
        py::arg("Xy").noconvert() = NULL,
        py::arg("lambd").noconvert() = NULL, 
        py::arg("psi").noconvert() = NULL,
        py::arg("starting_values").noconvert() = NULL, 
        py::arg("sqrt") = false,
        py::arg("fit_intercept") = true, 
        py::arg("opt_tol") = 1e-10,
        py::arg("max_iter") = 1000, 
        py::arg("zero_tol") = 1e-4);
};
