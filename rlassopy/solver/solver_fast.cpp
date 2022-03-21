// <%
// cfg['compiler_args'] = ['-std=c++14', '-stdlib=libc++',
// '-mmacosx-version-min=10.7'] cfg['include_dirs'] =
// ['/users/matiaspiqueras/eigen'] setup_pybind11(cfg)
// %>
// Created by matias on 10/11/19.
#include "eigen/Eigen/Eigen"
#include "eigen/unsupported/Eigen/MatrixFunctions"
#include </opt/homebrew/Cellar/pybind11/2.9.1/libexec/lib/python3.9/site-packages/pybind11/include/pybind11/eigen.h>
#include <iostream>
#include <math.h>
// #include <pybind11/eigen.h>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

VectorXd lassoShooting(MatrixXd X, VectorXd y, MatrixXd XX, VectorXd Xy,
                       double lambd, VectorXd psi, VectorXd startingValues,
                       bool sqrtLasso = false, double optTol = 1e-10,
                       int maxIter = 1000) {

  int n = X.rows(), p = X.cols();
  VectorXd beta = startingValues;

  // // check if XX and Xy are provided as input
  // if (XX.rows() == 0) {
  //   Eigen::MatrixXd XX = X.transpose() * X;
  // }
  // if (Xy.rows() == 0) {
  //   Eigen::VectorXd Xy = X.transpose() * y;
  // }

  // normal lasso shooting
  if (!sqrtLasso) {

    // check if starting values are provided
    // computes ridge regression otherwise
    // if (starting_values.rows() == 0) {
    //   MatrixXd L = MatrixXd::Identity(p, p) * lambd;
    //   VectorXd beta = (XX + L * psi.diagonal()).inverse() * Xy;
    //
    // } else {
    //   VectorXd beta = starting_values;
    // }

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

      if (ErrorNorm > MaxErrorNorm) {
        VectorXd aaa = (sqrt(n) * error / ErrorNorm);
        VectorXd bbb = (lambd / n) * psi - (X.transpose() * aaa / n).cwiseAbs();
        double dual = aaa.transpose() * bbb;
        // check convergence
        if (fobj - dual < optTol) {
          if ((beta - betaOld).norm() < optTol) {
            break;
          }
        }
      } else {
        double dual = (lambd / n) * (psi * beta.cwiseAbs()).sum();
        // check convergence
        if (fobj - dual < 1e-6) {
          if ((beta - betaOld).norm() < optTol) {
            break;
          }
        }
      }
    }
  }
  return beta;
}

PYBIND11_MODULE(solver_fast, m) {
  // pybind11::module m("code", "auto-compiled c++ extension");
  m.doc() = "Coordinate descent solver for lasso and sqrt-lasso";
  m.def("lasso_shooting", &lassoShooting, "Lasso shooting solver", py::arg("X"),
        py::arg("y"), py::arg("XX"), py::arg("Xy"), py::arg("lambd"),
        py::arg("psi"), py::arg("starting_values"), py::arg("sqrt") = false,
        py::arg("opt_tol") = 1e-10, py::arg("max_iter") = 1000);
}
