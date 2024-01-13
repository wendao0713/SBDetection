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
  std::vector<arma::mat> C(n);
  for (int jj = 0; jj < n; ++jj) {
    C[jj] = Y.col(jj) * data.row(jj + p);
  }
  
  // 计算 C.sum  矩阵
  arma::mat C_sum = zeros<mat>(k * p * n, k);
  C_sum.rows(0, k*p-1) = C[0];
  for (int i =2; i <= n ;i++){
    C_sum.rows((i-1)*k*p, i*k*p-1) = C_sum.rows((i-2)*k*p, (i-1)*k*p-1) + C[i-1];
  }
  
  // 计算C.sum.new 矩阵
  arma::mat C_sum_new = zeros<mat>(k * p * n, k);
  C_sum_new.rows(0, k * p - 1) = C_sum.rows((n - 1) * k * p, n * k * p - 1);
  for (int i = 2; i <= n; ++i) {
    C_sum_new.rows((i-1) * k * p, i * k * p - 1) = C_sum.rows((n - 1) * k * p, n * k * p - 1) - C_sum.rows((i - 2) * k * p, (i-1)* k * p - 1);
  }
  
  // 构建 D
  std::vector<arma::mat> D(n);
  for (int jj = 0; jj < n; ++jj) {
    D[jj] = Y.col(jj) * Y.col(jj).t();
  }
  
  // 计算 D.sum
  arma::mat D_sum = zeros<mat>(k * p * n, k * p);
  D_sum.rows(0, k*p-1) = D[0];
  for (int i =2; i <= n ;i++){
    D_sum.rows((i-1)*k*p, i*k*p-1) = D_sum.rows((i-2)*k*p, (i-1)*k*p-1) + D[i-1];
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
        arma::mat E = arma::zeros<arma::mat>(k * p, k);
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


// [[Rcpp::export]]
Rcpp::List first_step_cv_new(std::string method, 
                             arma::mat data_temp, 
                             arma::vec lambda, 
                             int p, 
                             arma::uvec cv_index, 
                             int max_iteration = 100, 
                             double tol = 1e-5, 
                             double step_size = 0.00005) {
  int kk = lambda.n_elem;
  int cv_l = cv_index.n_elem;
  int T_org = data_temp.n_rows;
  // int k_org = data_temp.n_cols;
  
  arma::mat data_org = data_temp;
  arma::mat forecast_all;
  arma::mat residual;
  arma::mat var_matrix;
  arma::vec cv_var = arma::zeros<arma::vec>(kk);
  arma::vec cv = arma::zeros<arma::vec>(kk);
  std::vector<arma::mat> phi_final(kk);
  std::vector<arma::uvec> brk_points_final(kk);
  
  int T = data_temp.n_rows;
  int k = data_temp.n_cols;
  
  int n = T - p;
  int m_hat = 0;
  
  for (int i = 0; i < kk; ++i) {
    List test;
    if (i == 0) {
      double lambda_vec = lambda(i);
      cout << lambda(i) << std::endl;
      test = var_break_fit(method, data_temp, lambda_vec, p, arma::mat(), max_iteration, tol, step_size );
    } else {
      arma::mat initial_phi = phi_final[i - 1];
      double lambda_vec = lambda(i);
      cout << lambda(i) << std::endl;
      test = var_break_fit(method, data_temp, lambda_vec, p, initial_phi, max_iteration, tol, step_size );
    }
    
    arma::mat phi_hat_full;
    phi_hat_full = as<arma::mat>(test["phihat"]);
    phi_final[i] = phi_hat_full;
    
    arma::uvec ll = {0};
    std::vector<std::vector<arma::uvec>> brk_points_list(ll.size());
    arma::uvec brk_points(n, arma::fill::zeros);
    
    for (size_t j = 0; j < ll.size(); ++j) {
      arma::mat phi_hat = phi_hat_full;
      n = T - p;
      m_hat = -1;
      arma::uvec brk_points(n, arma::fill::zeros);
      
      for (int iii = 2; iii <= n; ++iii) {
        if (arma::accu(arma::square(phi_hat.cols((iii - 1) * k * p, iii * k * p - 1))) > 0.005) {
          m_hat = m_hat+1;
          brk_points(m_hat) = iii;
        }
      }
      
      arma::uvec loc(m_hat, arma::fill::zeros);
      brk_points = brk_points.head(m_hat);
      brk_points = brk_points(arma::find(brk_points > p + 3 && brk_points < n));
      
      m_hat = brk_points.n_elem;
      
      if (m_hat > 1) {
        for (int mm = 1; mm < m_hat ; ++mm) {
          if (std::abs(static_cast<int>(brk_points(mm)) - static_cast<int>(brk_points(mm-1))) <= p + 1) {
            loc(mm) = mm;
          }
        }
      }
      // cout << "loc" << endl;
      // cout << loc << endl;
      loc = loc(arma::find(loc != 0));
      if (loc.n_elem > 0) {
        brk_points.shed_rows(loc);
      }
      brk_points_list[j].push_back(brk_points);
      // cout << "brk_points" << endl;
      // cout << brk_points << endl;
      brk_points_final[i] = brk_points;
    }
    // 假设 brk_points_final 是一个已经定义的 vector<arma::uvec>
    
    
    m_hat = brk_points.n_elem;
    
    std::vector<arma::mat> phi_full_all(T - p);
    arma::mat phi_temp_cv = arma::zeros<arma::mat>(k, k * p);
    arma::mat forecast = arma::zeros<arma::mat>(k, T - p);
    arma::mat forecast_new = arma::zeros<arma::mat>(k, cv_l);
    
    for (int j = p + 1; j <= T; ++j) {
      phi_temp_cv += phi_hat_full.cols((j - p - 1) * k * p, (j - p) * k * p - 1);
      phi_full_all[j - p - 1] = phi_temp_cv;
      forecast.col(j - p - 1) = pred(arma::trans(data_temp), phi_temp_cv, p, j - 1, k, 1);
    }
    
    for (int j = 0; j < cv_l; ++j) {
      forecast_new.col(j) = pred(arma::trans(data_org), phi_full_all[cv_index[j] - 1 - p - j + 1], p, cv_index[j] - 1, k, 1);
    }
    
    arma::mat forecast_all = arma::zeros<arma::mat>(k, T_org - p);
    forecast_all.cols(arma::conv_to<arma::uvec>::from(cv_index)) = forecast_new;
    // forecast_all.cols(arma::find(1 - arma::conv_to<arma::uvec>::from(cv_index))) = forecast;
    for (int i = 0; i < forecast_all.n_cols; ++i) {
      if (arma::find(cv_index == i).is_empty()) {
        forecast_all.col(i) = forecast.col(i);
      }
    }
    
    arma::mat residual = arma::square(forecast_all - arma::trans(data_org.rows(p , T_org-1)));
    arma::mat var_matrix = arma::zeros<arma::mat>(k, m_hat + 1);
    
    if (m_hat == 0) {
      var_matrix = arma::var(residual, 0, 1); // Assuming var calculates variance along each row
    } else {
      // var_matrix.col(0) = arma::trans(arma::var(residual.cols(0, std::max(brk_points[0]-2, 0)), 0, 1));
      var_matrix.col(0) = arma::var(residual.col(0), 0, 1);
      var_matrix.col(m_hat) = arma::var(residual.cols(brk_points[m_hat - 1], T_org - p -1), 0, 1);
    }
    if (m_hat >= 2) {
      for (int mm = 2; mm <= m_hat; ++mm) {
        // var_matrix.col(mm-1) = arma::var(residual.cols(brk_points[mm - 2], brk_points[mm-1] - 2), 0, 1);
        var_matrix.col(mm-1) = arma::var(residual.col(0), 0, 1);
      }
    }
    
    if (m_hat == 0) {
      cv_var[i] = arma::accu(var_matrix);
    } else {
      for (int i_1 = 0; i_1 < cv_l; ++i_1) {
        int ind = 0;
        for (int i_2 = 0; i_2 < m_hat; ++i_2) {
          if (cv_index[i_1] > brk_points[i_2]) {
            ind++;
          }
        }
        cv_var[i] += arma::accu(var_matrix.col(ind));
      }
    }
    cv_var[i] = std::sqrt(cv_var[i]) / (k * cv_l);
    // cout<<fixed << setprecision(10) << cv_var<<endl;
    
    // cv[i] = arma::accu(arma::square(forecast_new - data_org.cols(arma::conv_to<arma::uvec>::from(cv_index)).t())) / (k * cv_l);
    // cv[i] = arma::accu(arma::square(forecast_new - data_org.cols(cv_index).t())) / (k * cv_l);
    // Get the indices of the rows to select
    arma::uvec indices = cv_index;
    arma::mat selectedData = arma::trans(data_org.rows(indices));
    arma::mat diff = arma::square(forecast_new - selectedData);
    // Calculate the sum of squared differences
    double sumSquaredDiff = arma::accu(diff);
    double value = (1 / static_cast<double>(k * cv_l)) * sumSquaredDiff;
    cv[i] = value;
    // cout<<sumSquaredDiff << "divide" << (k * cv_l) <<endl;
    // 输出部分
    std::cout << "====================================" << std::endl;
    
  }
  
  
  // 寻找cv中的最小值所在的索引
  arma::uword lll = cv.index_min();
  int ind_new = 0;
  // 
  // 检查是否有任何更小的cv值在允许的变异范围内
  if (lll < kk) {
    for (int i_3 = lll + 1; i_3 < kk; ++i_3) {
      if (cv[i_3] < (cv[lll] + cv_var[lll])) {
        ++ind_new;
      }
    }
  }
  lll += ind_new;
  
  // 获取最终的phi.hat
  arma::mat phi_hat_full = phi_final[lll];
  
  // 打印cv和cv.var的值
  std::cout << "CV: " << std::endl;
  cv.print();
  std::cout << "CV.VAR: " << std::endl;
  cv_var.print();
  
  return Rcpp::List::create(Rcpp::Named("brk_points") = brk_points_final[lll],
                            Rcpp::Named("cv") = cv,
                            Rcpp::Named("cv_final") = lambda[lll]);}
// return List::create();}
