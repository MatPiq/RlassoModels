// <%
// cfg['compiler_args'] = ['-std=c++14', '-stdlib=libc++',
// '-mmacosx-version-min=10.7'] cfg['include_dirs'] =
// ['/users/matiaspiqueras/eigen'] setup_pybind11(cfg)
// %>
// Created by matias on 10/11/19.
#include "eigen/Eigen/Eigen"
#include <iostream>
#include <math.h>
#include </opt/homebrew/Cellar/pybind11/2.9.1/libexec/lib/python3.9/site-packages/pybind11/include/pybind11/eigen.h>
#include "eigen/unsupported/Eigen/MatrixFunctions"
//
// #include<iostream>
//
// #include <pybind11/eigen.h>
// #include <pybind11/pybind11.h>
//
// #include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;

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
    double qhat = error.pow(2).sum() / n;

    for (int i = 0; i < maxIter; i++) {

      VectorXd betaOld = beta;
      for (int j = 0; j < p; j++) {
        double s0 = (XX.row(j) * beta).sum() - XX(j, j) * beta(j) - Xy(j);

        if (fabs(beta(j)) > 0) {
          error += X.col(j) * beta(j);
          qhat = error.pow(2).sum() / n;
        }

        if (pow(n, 2) < pow(lambd, 2) / XX(j, j)) {
          beta(j) = 0;
        }

        double tmp = (lambd / n) * sqrt(qhat);
        double qqhat = qhat - (pow(s0, 2) / XX(j, j));
        if (qqhat < 0) {
          qqhat = 0;
        }

        if (s0 > tmp) { // Optimal beta(j) < 0
          beta[j] = ((lambd / sqrt(pow(n, 2) - pow(lambd, 2) / XX(j, j))) *
                         sqrt(qqhat) -
                     s0) /
                    XX(j, j);
          error = error - X.col(j) * beta[j];
        }
        if (s0 < -tmp) { // Optimal beta(j) > 0
          beta[j] = (-(lambd / sqrt(pow(n, 2) - pow(lambd, 2) / XX(j, j))) *
                         sqrt(qqhat) -
                     s0) /
                    XX(j, j);
          error = error - X.col(j) * beta[j];
        }

        if (fabs(s0) <= tmp) { // Optimal beta(j) = 0
          beta[j] = 0;
        }
      } // end loop beta^(i)_j

      // Update primal and dual value
      double fobj = sqrt((X * beta - y).pow(2).sum() / n) +
                    (lambd * beta.cwiseAbs() / n).sum();

      double ErrorNorm = error.norm();
      VectorXd lambdVec = VectorXd::Ones(p) * lambd;
      if (ErrorNorm > MaxErrorNorm) {
        VectorXd aaa  = (sqrt(n)*error/ErrorNorm);
        VectorXd bbb = (  lambdVec/n - (X.transpose()*aaa/n).cwiseAbs() ).cwiseAbs();
        double dual = (aaa.transpose()) * y/n - bbb.transpose() * (beta).cwiseAbs();
        // check convergence
        if (fobj - dual < optTol) {
          if ((beta - betaOld).norm() < optTol) {
            break;
          }
        }
      } else {
        double dual = (lambd * (beta).cwiseAbs().sum() / n);
        // check convergence
        if (fobj - dual < optTol) {
          if ((beta - betaOld).norm() < optTol) {
            break;
          }
        }
      }
    }
  }
    return beta;
}

  PYBIND11_MODULE(lassoShooting, m) {
    // pybind11::module m("code", "auto-compiled c++ extension");
    m.doc() = "Coordinate descent solver for lasso and sqrt-lasso";

    m.def("lasso_shooting", &lassoShooting);
  }
