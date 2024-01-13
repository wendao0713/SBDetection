\name{var_lasso_brk}
\alias{var_lasso_brk}
\title{VAR Model Fitting with LASSO Regularization}
\description{
  This function fits a Vector Autoregression (VAR) model using LASSO regularization.
}
\usage{
var_lasso_brk(data, lambda, p, max_iteration = 1000, tol = 1e-4)
}
\arguments{
  \item{data}{A matrix of time series data, where each column represents a variable.}
  \item{lambda}{A numeric vector of regularization parameters for the LASSO penalty.}
  \item{p}{An integer specifying the order of the VAR model.}
  \item{max_iteration}{An integer specifying the maximum number of iterations for the optimization algorithm. Default is 1000.}
  \item{tol}{A numeric value specifying the tolerance level for convergence. Default is 1e-4.}
}
\value{
  A list containing:
  \itemize{
    \item{phi_hat}{A matrix of the estimated VAR model coefficients.}
    \item{iter}{A matrix representing the number of iterations taken for each variable.}
    \item{pred_error}{A vector of prediction errors for each value of lambda.}
    \item{tune_final}{The lambda value that resulted in the final model.}
  }
}
\details{
  The function implements an optimization algorithm to fit a VAR model using LASSO regularization. It uses Singular Value Decomposition (SVD) for efficient computation and allows for tuning of hyperparameters like lambda, the order of the model, and the convergence criteria.
}
\examples{
# Sample usage
data <- matrix(rnorm(200), ncol = 2)
lambda <- c(0.1, 0.5, 1)
result <- var_lasso_brk(data, lambda, p = 2)
print(result$phi_hat)
}

