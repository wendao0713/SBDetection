#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
#include <cmath>
#include <algorithm>
#include <iostream>


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



Rcpp::List var_lasso_brk(mat data, vec lambda, int p, int max_iteration = 1000, double tol = 1e-4) {
  // weight is not used, so delete it from parameters
  // simulation example use a single value lambda, so change its datatype from vec to double
  int k = data.n_cols;
  int T = data.n_rows;
  int T_1 = T;
  mat iter = zeros<mat>(k, lambda.n_elem);
  mat phi_hat = zeros<mat>(k, k * p);
  mat phi_hat_fista = zeros<mat>(max_iteration, k * p);
  vec pred_error = zeros<mat>(lambda.n_elem);
  mat phi_hat_temp = zeros<mat>(k, k * p * lambda.n_elem);
  mat Y = data.t();
  Y = Y.submat(0, p, k-1, T_1-1);
  mat Z = zeros<mat>(k * p, T_1 - p);
  for (int i = 1; i <= T_1 - p; i++) {
    for (int j = 1; j <= p; j++) {
      Z.submat((j-1)*k, i-1, j*k-1, i-1) = data.row(i + p - j - 1).t();
    }
  }
  // 声明存储svd结果的变量
  mat U;
  vec s;
  mat V;
  svd(U, s, V, Z);
  double step_size = 1 / pow(std::max(s(0), 2.0), 2.0);//获取最大奇异值，步长不小于2
  
  mat phi_new;
  for (int ll = 0; ll < lambda.n_elem; ll++) {
    for (int ii = 0; ii < k; ii++) {
      int l = 2;
      while (l < max_iteration) {
        l++;
        mat phi_temp = phi_hat_fista.row(l-2) + 
          ((l - 2) / (l + 1)) * (phi_hat_fista.row(l-2) - phi_hat_fista.row(l - 3));
        vec mid_outcome = (Y.row(ii)-phi_temp*Z).t();
        phi_new = phi_temp + step_size * (Z*mid_outcome).t();
        phi_new = soft(phi_new, zeros<vec>(k*p), lambda(ll));
        double max_diff = 0.0;
        for (int i = 0; i < k * p; i++) {
          double substra = std::abs(phi_new(0,i) - phi_temp(0,i));
          max_diff = (max_diff>substra)?max_diff:substra;
        }
        if (max_diff < tol) {
          break;
        }
        if (max_diff > tol) {
          phi_hat_fista.row(l-1) = phi_new.row(0);
        }
      }
      iter(ii, ll) = l;
      phi_hat_temp.submat(ii, ll * k * p, ii, (ll+1) * k * p-1) = phi_new;
    }
    mat forecast = zeros<mat>(k, T_1 - p);
    for (int j = p; j < T_1; j++) {
      forecast.col(j - p) = pred(data.t(), 
                   phi_hat_temp.cols(ll * k * p, (ll+1)*k*p-1), p, j, k, 1);
    }
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < T_1 - p; j++) {
        pred_error(ll) += pow(data(j + p, i) - forecast(i, j), 2);
      }
    }
    double mid_sum = accu(abs(phi_hat_temp.cols(ll * k * p, (ll+1)*k*p-1)));
    pred_error(ll) = pred_error(ll) + lambda(ll) * mid_sum ;
  }
  
  int ll_final = 1;
  phi_hat = phi_hat_temp.cols((ll_final-1)*k*p,ll_final*k*p-1);
  return Rcpp::List::create(
    Rcpp::Named("phi_hat") = phi_hat, 
    Rcpp::Named("iter") = iter, 
    Rcpp::Named("pred_error") = pred_error, 
    Rcpp::Named("tune_final") = lambda(ll_final-1)
  );
}

// [[Rcpp::export]]
Rcpp::List break_var(mat data, vec lambda, vec pts, int p, 
                     int max_iteration = 1000, double tol = 1e-4, 
                     double step_size = 1e-3) {
  int k = data.n_cols;
  int T = data.n_rows;
  int m = pts.n_elem;
  vec L_n = zeros<vec>(m + 1);
  vec pts_temp;
  
  if (m == 0) {
    pts_temp = ones<vec>(T + 1);
  }
  
  if (m > 0) {
    pts_temp = zeros<vec>(m+2);
    pts_temp[0] = 1;
    pts_temp[m+1] = T+1;
    pts_temp.subvec(1,m) = pts;
  }
  for (int mm = 0; mm <= m; ++mm) {
    int start = pts_temp[mm] - 1;
    int end = pts_temp[mm + 1] - 2;
    mat data_temp = data.rows(start, end);
    Rcpp::List try_result = var_lasso_brk(data_temp,lambda,p,max_iteration,tol);
    L_n[mm] = try_result["pred_error"];
  }
  return Rcpp::List::create(Rcpp::Named("L_n") = sum(L_n));
}