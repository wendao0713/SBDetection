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
  arma::mat concat_Y = arma::zerosautolinkarma::matautolink(k, T + h);
  concat_Y.cols(0, T - 1) = Y.cols(0, T - 1);
  for (int j = 0; j < h; ++j) {
    arma::mat temp = arma::zerosautolinkarma::matautolink(k, 1);
    for (int i = 1; i <= p; ++i) {
      temp += phi.cols((i - 1) * k, i * k - 1) * concat_Y.col(T + j - i);
    }
    concat_Y.col(T + j) = temp;
  }
  return concat_Y.col(T + h - 1);
}


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

// [[Rcpp::export]]
List var_break_fit(std::string method,
                   const arma::mat& data,
                   double lambda,
                   int p,
                   const arma::mat& initial_phi,
                   int max_iteration=1000,
                   double tol=1e-4,
                   double step_size=1e-3
) {
  if (method != "LASSO") {
    stop("ERROR: Method not supported.");
  }
  int k = data.n_cols;
  int T = data.n_rows;
  arma::mat iter = zeros<mat>(k, 1);
  int n = T - p;
  // 构建 Y 矩阵
  arma::mat Y = zeros<mat>(k * p, n);
  for (int i = p; i < T; ++i) {
    Y.col(i-1) =data.row(i-1).t();
  }
  // 构建 C 向量
  std::vectorautolinkarma::matautolink C(n);
  for (int jj = 0; jj < n; ++jj) {
    C[jj] = Y.col(jj) * data.row(jj + p);
  }
  // 计算 C.sum  矩阵
  arma::mat C_sum = zeros<mat>(k * p * n, k);
  C_sum.rows(0, kp-1) = C[0];
  for (int i =2; i <= n ;i++){
    *    C_sum.rows((i-1)***k***p, i*k*p-1) = C_sum.rows((i-2)*k*p, (i-1)*k*p-1) + C[i-1];
  }
  // 计算C.sum.new 矩阵
  arma::mat C_sum_new = zeros<mat>(k * p * n, k);
  C_sum_new.rows(0, k * p - 1) = C_sum.rows((n - 1) * k * p, n * k * p - 1);
  for (int i = 2; i <= n; ++i) {
    C_sum_new.rows((i-1) * k * p, i * k * p - 1) = C_sum.rows((n - 1) * k * p, n * k * p - 1) - C_sum.rows((i - 2) * k * p, (i-1)* k * p - 1);
  }
  // 构建 D
  std::vectorautolinkarma::matautolink D(n);
  for (int jj = 0; jj < n; ++jj) {
    D[jj] = Y.col(jj) * Y.col(jj).t();
  }
  // 计算 D.sum
  arma::mat D_sum = zeros<mat>(k * p * n, k * p);
  D_sum.rows(0, kp-1) = D[0];
  for (int i =2; i <= n ;i++){
    *    D_sum.rows((i-1)***k***p, i*k*p-1) = D_sum.rows((i-2)*k*p, (i-1)*k*p-1) + D[i-1];
  }
  // 计算 D.sum.new
  arma::mat D_sum_new = zeros<mat>(k * p * n, k * p);
  D_sum_new.rows(0, k * p - 1) = D_sum.rows((n - 1) * k * p, n * k * p - 1);
  for (int i = 2; i <= n; ++i) {
    D_sum_new.rows((i-1) * k * p, i * k * p - 1) = D_sum.rows((n - 1) * k * p, n * k * p - 1) - D_sum.rows((i - 2) * k * p, (i-1) * k * p - 1);
  }
  // 计算 D.sum.new.inv
  arma::mat D_sum_new_inv = zeros<mat>(k * p * n, k * p);
  for (int jj = 1; jj <= n; ++jj) {
    D_sum_new_inv.rows((jj-1) * k * p, jj * k * p - 1) = inv(D_sum_new.rows((jj-1) * k * p, jj * k * p - 1) + (0.1) * eye<mat>(k * p, k * p));
    
  }
  // 初始化 phi.hat
  arma::mat phi_hat = zeros<mat>(k, k * p * n);
  if (initial_phi.n_elem > 0) {
    phi_hat = initial_phi;
  }
  // 计算 active
  arma::uvec active;
  for (int jjj = 0; jjj < n; ++jjj) {
    double sum_squares = 0;
    for (int i = 0; i < k * p; ++i) {
      sum_squares += std::pow(phi_hat(jjj * k * p + i), 2);
    }
    if (sum_squares != 0) {
      active.insert_rows(active.n_rows, 1);
      active(active.n_rows - 1) = jjj + 1; // C++中索引从0开始，R从1开始
    }
  }
  // 构建 phi_new
  arma::mat phi_new = zeros<mat>(k, k * p * n);
  // 运用lasso方法估计系数
  for (unsigned int ll = 0; ll < 1; ++ll) {
    int l = 2;
    arma::mat phi_compare = phi_hat;
    while (l < max_iteration) {
      ++l;
      for (int ii = 1; ii <= n; ++ii) {
        // 假设 D_sum_new, phi_hat, n, k, p 已经定义
        arma::mat E = arma::zerosautolinkarma::matautolink(k * p, k);
        for (int jj = 1; jj <= n; ++jj) {
          int index = std::max(jj, ii);
          E += D_sum_new.rows((index-1) * k * p, index * k * p - 1) * phi_hat.cols((jj-1) * k * p, jj * k * p - 1).t();
        }
        E -= D_sum_new.rows((ii-1) * k * p, ii * k * p - 1) * phi_hat.cols((ii-1) * k * p, ii * k * p - 1).t();
        // 假设 C_sum_new 和 E 已经定义
        arma::mat S = C_sum_new.rows((ii-1) * k * p, ii * k * p - 1) - E;
        S = soft_full(S, lambda);
        arma::mat phi_temp = D_sum_new_inv.rows((ii - 1) * k * p, ii * k * p - 1) * S;
        phi_temp = phi_temp.t();
        phi_hat.cols((ii-1) * k * p, ii * k * p - 1) = phi_temp;
        phi_new.cols((ii-1) * k * p, ii * k * p - 1) = phi_temp;
      }
      // cout << phi_new.max() << endl;
      // cout << phi_compare.min() << endl;
      // 计算差异的绝对值的最大值
      double max_diff = arma::abs(phi_new - phi_compare).max();
      if (max_diff < tol) {
        cout << max_diff << "<tol" << std::endl;
        break;
      }
      if (max_diff > tol) {
        phi_hat = phi_new;
        cout << max_diff << ">tol" << std::endl;
      }
      // Rcpp::Rcout << l << std::endl;
      // }
    }
  }
  return List::create(Named("phihat") = phi_hat,
                      Named("iter") = iter);}

