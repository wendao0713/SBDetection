#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
#include <cmath>
#include <algorithm>
#include <iostream>

// [[Rcpp::export]]
mat soft(mat L_in, vec weight, double lambda) {
  mat L(1, L_in.n_elem); // Create a matrix with one row to store the result
  L.row(0) = L_in.row(0); // Copy the input matrix to the first row of L
  for (int i = 0; i < L.n_elem; ++i) {
    lambda *= (1 + weight(i));
    if (L(0, i) > lambda) {
      L(0, i) -= lambda;
    } else if (L(0, i) < -lambda) {
      L(0, i) += lambda;
    } else {
      L(0, i) = 0;
    }
  }
  return L;
}