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


arma::mat pred0(const arma::mat& Y, const arma::mat& phi, int p, int T, int k, int h) {
  int rows = Y.n_rows;
  int cols = Y.n_cols;
  arma::mat concat_Y = arma::zeros<arma::mat>(k, T + h);
  
  concat_Y.cols(0, T-1) = Y.cols(0,T-1);
  
  for (int j = 0; j < h; ++j) {
      arma::vec temp = arma::zeros<arma::vec>(k);
      
      for (int i = 0; i < p; ++i) {
        temp += phi.cols(i * k, (i + 1) * k - 1) * concat_Y.col(T + j - i - 1);
      }
      
      concat_Y.col(T + j) = temp;
  }
  
  return concat_Y.col(T + h - 1);
}


arma::vec soft_threshold(arma::vec& L, const arma::vec& weight, double lambda) {
  arma::vec result = L;
  double curr_lambda;
  for (int i = 0; i < L.n_elem; ++i) {
    curr_lambda = lambda * (1 + weight(i));
    if (L(i) > curr_lambda) {
      L(i) = L(i)-curr_lambda;
    } 
    if (L(i) < -curr_lambda) {
      L(i) = L(i) + curr_lambda;
    } 
    if(std::abs(L(i))<=curr_lambda){
      L(i) = 0;
    }
  }
  return L;
}

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

List phi_hat_iter(int& k,int& max_iteration,
                          mat& phi_hat_fista,int step_size,mat& Z,mat Y,
                          vec& lambda,int p,int m_hat,int ll,
                          mat& iter,mat& phi_hat_temp) 
  {
  for(int ii=1;ii<k+1;++ii){
  int l=2;
  while(l<max_iteration){
    l+=1;
    vec phi_temp = phi_hat_fista.row(l-2)+((l-2)/(l+1))*(phi_hat_fista.row(l-2) - phi_hat_fista.row(l-3));
    vec phi_new = phi_temp + step_size*((Z*((Y.row(ii-1)-phi_temp*Z).t())).t());
    phi_new = soft_threshold(phi_new,zeros(k*m_hat*p),lambda[ll-1]);
  }
  iter(ii-1,ll-1)=l;
  phi_hat_temp.row(ii-1).cols((ll-1)*k*m_hat*p,(ll*k*m_hat*p)-1);
  }
  
  List result;
  result['phi_hat_temp']=phi_hat_temp;
  return(result);
  }
  
  // [[Rcpp::export]]
  Rcpp::List ar_est(std::string method, const arma::mat& data, Rcpp::Nullable<Rcpp::NumericVector> weight, 
                        const arma::vec& lambda, int p, const arma::vec& break_pts, int r_n, 
                        int max_iteration = 1000, double tol = pow(10, -4), double step_size = pow(10, -3)) {
    
    // Method check
    if (method != "LASSO") {
      Rcpp::Rcout << "ERROR" << std::endl;
      return Rcpp::List::create();  
    }
    
    
    int k = data.n_cols;
    int T = data.n_rows;
    int T_1 = T;
    int m_hat = break_pts.n_elem + 1;
    
    arma::vec ind_remain = arma::zeros<arma::vec>(2 + 2 * break_pts.n_elem);
    ind_remain(0) = p;
    ind_remain(ind_remain.n_elem - 1) = T_1;
    
    for (int i = 0; i < break_pts.n_elem; ++i) {
      ind_remain(2 * (i+1)-1) = break_pts(i) - r_n - 1;
      ind_remain(2 * (i +1)) = break_pts(i) + r_n + 1;
    }
    
    arma::mat iter = arma::zeros<arma::mat>(k, lambda.n_elem);
    arma::mat phi_hat = arma::zeros<arma::mat>(k, k * m_hat * p);
    arma::mat phi_hat_fista = arma::zeros<arma::mat>(max_iteration, k * m_hat * p);
    arma::vec pred_error = arma::zeros<arma::vec>(lambda.n_elem);
    arma::mat phi_hat_temp = arma::zeros<arma::mat>(k, k * m_hat * p * lambda.n_elem);
    arma::vec std_res = arma::zeros<arma::vec>(lambda.n_elem);
    
    
    arma::mat Y = data.t();
    Y = Y.submat(0, p, k-1, T_1-1);
    arma::mat Z = arma::zeros<arma::mat>(k * p, T_1 - p);
    //std::cout<<Z.n_cols<<Z.n_cols<<endl;
        
    for (int i = 1; i <= T_1 - p; i++) {
      for (int j = 1; j <= p; j++) {
        Z.submat((j-1)*k, i-1, j*k-1, i-1) = data.row(i + p - j - 1).t();
      }
    }
    //std::cout<<Z.n_rows<<Z.n_cols<<endl;
    for (int i = 0; i < break_pts.n_elem; ++i) {
      Y.shed_cols(break_pts(i) - r_n, break_pts(i) + r_n);
    }
    int n = Y.n_cols;
    arma::mat Z_new = arma::zeros<arma::mat>(T_1 - p, k * m_hat * p);
    Z_new(span(0, break_pts(0) - r_n - 2), span(0, k * p - 1)) = (Z.submat(0,0,k*p-1,break_pts(0) - r_n - 2)).t();
    //std::cout<<Z_new.n_cols<<Z_new.n_cols<<endl;
    int i=0;
    
    if (m_hat > 2) {
      
      for (; i < m_hat - 2; ) {
        int start_row = break_pts(i) + r_n ;
        int end_row = break_pts(i + 1) - r_n - 2;
        int start_col = (i+1) * k * p - 1;
        int end_col = (i + 2) * k * p - 2;
        Z_new(span(start_row, end_row), span(start_col, end_col)) = (Z.submat(0,break_pts(i)+r_n,k*p-1,break_pts(i+1) - r_n - 2)).t();
        i++;
      }
    }

    int last_break = break_pts(m_hat - 2);
    //cout<<i;
   Z_new(span(last_break + r_n , T_1 - p - 1), span((m_hat - 1) * k * p, m_hat * k * p - 1)) 
     = (Z.submat(0,last_break+r_n,k*p-1,T_1-p-1)).t();

    arma::uvec del_ind;
    
    for (int i = 0; i < Z_new.n_rows; ++i) {
      if (arma::accu(arma::square(Z_new.row(i))) == 0) {
        del_ind.resize(del_ind.n_elem + 1); // 调整大小
        del_ind(del_ind.n_elem - 1) = i;    // 将索引添加到末尾
        //std::cout<<'i='<<i<<'  ';//<<'del_ind='<<del_ind;
      }
    }

   //std::cout<<del_ind;
    Z_new.shed_rows(del_ind);
      
    Z = Z_new.t();
    //std::cout<<Z.n_rows<<Z.n_cols<<endl;
    //cout<<Z;

    // Updating step size
    step_size = std::pow(1 / arma::max(arma::svd(Z)), 2);
    
    //arma::vec phi_new;
    mat phi_new;
   
    for (int ll = 0; ll < lambda.n_elem; ++ll) {
      for (int ii = 0; ii < k; ++ii) {
        
        int l = 2;
        while (l < max_iteration) {
          l++;
          mat phi_temp = phi_hat_fista.row(l-2) + 
            ((l - 2) / (l + 1)) * (phi_hat_fista.row(l-2) - phi_hat_fista.row(l - 3));
          //std::cout<<phi_temp.nrows();
          //std::cout<<endl<<'======';
          //std::cout<<phi_temp.n_cols;
          vec mid_outcome = (Y.row(ii)-phi_temp*Z).t();
          //std::cout << "mid_outcome sucess " << std::endl;
          phi_new = phi_temp + step_size * (Z*(mid_outcome)).t();
          //std::cout<<phi_temp.n_elem<< std::endl;
          //std::cout << "phi_new sucess "<<'n_elem'<<"   " <<k*m_hat*p<< std::endl;
          phi_new =  soft(phi_new,zeros<vec>(k*m_hat*p) , lambda(ll));
          
          double max_diff = 0.0;
          for (int i = 0; i < k * p; i++) {
            double substra = std::abs(phi_new(0,i) - phi_temp(0,i));
            //std::cout << "substra sucess "<< substra << std::endl;
            max_diff = (max_diff>substra)?max_diff:substra;
          }
          //std::cout << "max loop sucess " << std::endl;
          if (max_diff < tol) {
            //std::cout << "break " << std::endl;
            break;
          }
          if (max_diff > tol) {
            phi_hat_fista.row(l-1) = phi_new.row(0);
          }
          
        }
        iter(ii, ll) = l;
        phi_hat_temp.submat(ii, ll * m_hat* k * p, ii, (ll+1) * k* m_hat * p-1) = phi_new;
        //std::cout << "submat sucess " << std::endl;
        //iter(ii, ll) = l;
        //phi_hat_temp.submat(ii, ll * k * m_hat*p, ii, (ll+1) * k *m_hat* p-1) = phi_new;
        //phi_hat_temp.row(ii).subvec(ll * k * m_hat * p, (ll + 1) * k * m_hat * p - 1) = phi_new.t();
      }
      
      arma::mat forecast = arma::zeros<arma::mat>(k, T_1);
      
      for (int i = 0; i < m_hat; ++i) {
        int len = ind_remain(2 * (i+1)-1) - ind_remain(2 * (i+1) - 2);
        int lb = ind_remain(2 * (i+1) - 2)+1;
        int ub = ind_remain(2 * (i+1) -1);
        
        //cout<<"start forecast"<<endl;
        //cout<<lb<<" "<<ub<<endl;
        //cout<<ind_remain<<endl;
        for (int jjj = lb-1; jjj < ub; ++jjj) {
          forecast.col(jjj ) = 
            pred0((data).t(), phi_hat_temp.cols(ll  * k * m_hat * p + i  * k * p, ll * k * m_hat * p + (i +1)* k * p - 1), p, jjj, k, 1);
       }
      }
      

      
      if (ll == 0) { 
        arma::uvec del_ind0;
        for (int i = 0; i < T_1; ++i) {
          if (arma::accu(arma::square(forecast.col(i))) == 0) {
            del_ind0.resize(del_ind0.n_elem + 1); // 调整大小
            del_ind0(del_ind0.n_elem - 1) = i;  
            
           }
        }
        //forecast.shed_cols(del_ind0);
      }
      arma::mat residual = (data).t() - forecast;
      double BIC_temp = 0;
      
      for (int i = 0; i < m_hat; ++i) {
       int lb = ind_remain(2 * (i+1) - 2)+1;
       int ub = ind_remain(2 * (i+1)-1);
        
        residual.cols(lb-1, ub-1);
        phi_hat_temp.cols(ll  * k * m_hat * p + i  * k * p, ll  * k * m_hat * p + (i+1) * k * p - 1);
        
         double ttt=AIC_BIC(residual.cols(lb-1, ub-1), phi_hat_temp.cols(ll  * k * m_hat * p + i  * k * p, ll  * k * m_hat * p + (i+1) * k * p - 1))["BIC"];
        //std::cout<<ttt;
        BIC_temp =BIC_temp+ttt;
      }
      pred_error(ll) = BIC_temp;
    }
    
      int ll_final = pred_error.index_min();
      
      arma::mat phi_hat_final = phi_hat_temp.cols((ll_final - 1) * k * m_hat * p, ll_final * k * m_hat * p - 1);
  
    return Rcpp::List::create(Rcpp::Named("phi_hat") = phi_hat_final, 
                              Rcpp::Named("iter") = iter,
                              Rcpp::Named("pred_error") = pred_error,
                              Rcpp::Named("tune.final") = lambda(ll_final)
    );
    
  }
  