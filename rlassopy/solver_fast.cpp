// <%
// cfg['compiler_args'] = ['-std=c++14', '-stdlib=libc++',
// '-mmacosx-version-min=10.7'] cfg['include_dirs'] =
// ['/users/matiaspiqueras/eigen'] setup_pybind11(cfg)
// %>
//
// Created by matias on 10/11/19.
#include <iostream>
#include <math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stdio.h>

namespace py = pybind11;

using namespace Eigen;

Eigen::VectorXd lassoShooting(Eigen::MatrixXd X, Eigen::VectorXd y,
                              Eigen::MatrixXd XX, Eigen::VectorXd Xy,
                              Eigen::VectorXd starting_values, double lambd,
                              Eigen::VectorXd psi, double opt_tol,
                              int max_iter) {

  // int n = X.rows();
  int p = X.cols();

  // double the XX and Xy
  XX = XX * 2;
  Xy = Xy * 2;
  // set beta to starting values
  Eigen::VectorXd beta = starting_values;

  // loop over max iter
  for (int i = 0; i < max_iter; i++) {
    // copy beta to beta_old
    Eigen::VectorXd beta_old = beta;
    // loop over p
    for (int j = 0; j < p; j++) {
      // calculate s0
      double s0 = 0;
      // for (int k = 0; k < p; k++) {
      //   s0 += beta_old(k) * XX(j, k);
      // }
      // s0 = s0 - XX(j, j) * beta_old(j) - Xy(j);
      double s0 = (XX.row(j) * beta).sum() - XX(j, j) * beta[j] - Xy[j];

      if (s0 > lambd * psi(j)) {
        beta(j) = (lambd * psi(j) - s0) / XX(j, j);

      } else if (s0 < -lambd * psi(j)) {
        beta(j) = (-lambd * psi(j) - s0) / XX(j, j);

      } else {
        beta(j) = 0;
      }
    }
    // check for convergence
    double diff = (beta - beta_old).norm();
    if (diff < opt_tol) {
      break;
    }
  }

  // return p as float
  return beta;
}

PYBIND11_MODULE(solver_fast, m) {
  // pybind11::module m("code", "auto-compiled c++ extension");
  m.doc() = "Coordinate descent solver"; // optional module docstring
  m.def("lasso_shooting", &lassoShooting);
}
