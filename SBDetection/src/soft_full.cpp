#include "RcppArmadillo.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;


// [[Rcpp::export]]
arma::mat soft_full(arma::mat L, double lambda) {
  int nrows = L.n_rows;
  int ncols = L.n_cols;
  
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      if (L(i, j) > lambda) {
        L(i, j) = L(i, j) - lambda;
      } else if (L(i, j) < -lambda) {
        L(i, j) = L(i, j) + lambda;
      } else {
        L(i, j) = 0;
      }
    }
  }
  
  return L;
}