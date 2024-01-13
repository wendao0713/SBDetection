#include "RcppArmadillo.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
arma::mat pred(const arma::mat& Y, const arma::mat& phi, int p, int T, int k, int h) {
  arma::mat concat_Y = arma::zeros<arma::mat>(k, T + h);
  concat_Y.cols(0, T - 1) = Y.cols(0, T - 1);
  
  for (int j = 0; j < h; ++j) {
    arma::mat temp = arma::zeros<arma::mat>(k, 1);
    for (int i = 1; i <= p; ++i) {
      temp += phi.cols((i - 1) * k, i * k - 1) * concat_Y.col(T + j - i);
    }
    concat_Y.col(T + j) = temp;
  }
  return concat_Y.col(T + h - 1);
}
