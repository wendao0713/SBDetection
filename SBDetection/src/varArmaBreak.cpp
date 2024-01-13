#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
Rcpp::List varArmaBreak(const int& nobs, const  mat& sigma, const Rcpp::Nullable<Rcpp::NumericVector>& arlags = R_NilValue,
                          const Rcpp::Nullable<Rcpp::NumericVector>& malags = R_NilValue,
                          const Rcpp::Nullable<Rcpp::NumericVector>& cnst = R_NilValue,
                          const Rcpp::Nullable<Rcpp::NumericMatrix>& phi = R_NilValue,
                          const Rcpp::Nullable<Rcpp::NumericMatrix>& theta = R_NilValue,
                          const int& skip=200 ,  const Rcpp::Nullable<Rcpp::NumericVector>& brk = R_NilValue)
{
  int k = sigma.n_rows;
  int m = brk.isNotNull() ? Rcpp::as<Rcpp::NumericVector>(brk).size() : 1;
  
  int nT = nobs + skip;
  mat mattheta=theta.isNotNull()?  Rcpp::as< mat>(theta): zeros< mat>(k,nT);
   vec cnstVec = cnst.isNotNull() ? Rcpp::as< vec>(cnst) :  zeros< vec>(k);
  mat matphi=phi.isNotNull()?  Rcpp::as< mat>(phi): zeros< mat>(k,nT);
  //noise  
  vec M(k, fill::zeros);
  mat at = mvnrnd(M, sigma,nT).t();
  
  
  int nar = arlags.isNotNull() ? Rcpp::as<Rcpp::NumericVector>(arlags).size() : 0;
  int p = 0;
  
  if (nar > 0) {
    vec arlagsVec = Rcpp::as< vec>(arlags);
    arlagsVec=sort(arlagsVec);
    p = arlagsVec(nar - 1);
  }
  
  int nma = malags.isNotNull() ? Rcpp::as<Rcpp::NumericVector>(malags).size() : 0;
  int q = 0;
  if (nma > 0) {
    vec malagsVec = Rcpp::as< vec>(malags);
    malagsVec=sort(malagsVec);
    q = malagsVec(nma - 1);
  }
  
  int ist = std::max(p, q) + 1;
  mat zt(nT, k,  fill::zeros);
  
  
  if (m == 1){
    for (int it = ist; it <= nT; ++it) {
       rowvec tmp = at.row(it - 1); // 获取at的第it行，Armadillo的索引从0开始
      
      if (nma > 0) {
        for (int j = 0; j < nma; ++j) {
          int jdx = j * k;
           mat thej = mattheta.cols(jdx, jdx + k - 1);
           rowvec atm = at.row(it - Rcpp::as< vec>(malags)[j] - 1);
          tmp -= atm * thej.t();
        }
      }
      
      if (nar > 0) {
        for (int i = 0; i < nar; ++i) {
          int idx = i * k;
           mat phj = matphi.cols(idx, idx + k - 1);
           rowvec ztm = zt.row(it - Rcpp::as< vec>(arlags)[i] - 1);
          tmp += ztm * phj.t();
        }
      }
      
      zt.row(it - 1) = tmp;
    }
    
  }
  
  if (m > 1) {
    vec brkVec=Rcpp::as< vec>(brk);
    for (int i = 0; i < m; ++i) {
      int start = skip + (i == 0 ? 0 : brkVec[i-1]);
      int end = skip + brkVec[i] - 1;
      
      for (int it = start; it <= end; ++it) {
        // Perform calculations for each time point within the current breakpoint
         rowvec tmp = at.row(it - 1); // 获取at的第it行，Armadillo的索引从0开始
        
        if (nma > 0) {
          for (int j = 0; j < nma; ++j) {
            int jdx = j * k;
             mat thej = mattheta.cols(i*q*k+jdx, i*q*k+jdx + k - 1);
             rowvec atm = at.row(it - Rcpp::as< vec>(malags)[j] - 1);
            tmp -= atm * thej.t();
          }
        }
        
        if (nar > 0) {
          for (int i = 0; i < nar; ++i) {
            int idx = i * k;
             mat phj = matphi.cols((i*p*k+idx),(i*p*k+idx + k-1));
             rowvec ztm = zt.row(it - Rcpp::as< vec>(arlags)[i] - 1);
            tmp += ztm * phj.t();
          }
        }
        
        zt.row(it - 1) = tmp;
      }
    }
  }
  
  zt = zt.rows(skip , nT - 1);
  at = at.rows(skip , nT - 1);
  
  return Rcpp::List::create(Rcpp::Named("series") = zt, Rcpp::Named("noises") = at);
}