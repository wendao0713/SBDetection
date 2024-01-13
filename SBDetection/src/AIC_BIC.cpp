#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
#include <cmath>
#include <algorithm>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to calculate AIC and BIC
List AIC_BIC(const mat& residual, const mat& phi) {
  int k = phi.n_rows;
  int k_lam = phi.n_cols;
  int T_new = residual.n_cols;
  int count = 0;
  
  // Count non-zero elements in phi
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k_lam; ++j) {
      if (phi(i, j) != 0) {
        count++;
      }
    }
  }
  
  // Calculate sigma.hat (covariance matrix of residuals)
  mat sigma_hat = zeros<mat>(k, k);
  for (int i = 0; i < T_new; ++i) {
    sigma_hat += residual.col(i) * trans(residual.col(i));
  }
  sigma_hat /= T_new;
  
  // Check eigenvalues and adjust if necessary
  vec eigenvalues = eig_sym(sigma_hat);
  double ee_temp = eigenvalues.min();
  if (ee_temp <= 0) {
    sigma_hat += 2.0 * (abs(ee_temp) + 1e-4) * eye<mat>(k, k);
  }
  
  // Calculate log determinant of the covariance matrix
  double log_det = log(det(sigma_hat));
  
  // Calculate AIC and BIC
  List results;
  results["AIC"] = log_det + 2 * count / T_new;
  results["BIC"] = log_det + log(T_new) * count / T_new;
  
  return results;
}
