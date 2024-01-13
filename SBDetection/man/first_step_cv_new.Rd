\name{first_step_cv_new}
\alias{first_step_cv_new}
\title{Cross-validation for Time Series Models with Potential Break Points}
\description{
  This function performs cross-validation for time series models, particularly focusing on identifying potential break points in the data. It allows the user to specify various parameters for the model fitting process and calculates the cross-validation error for different break point scenarios.
}
\usage{
first_step_cv_new(method, data_temp, lambda, p, cv_index, max_iteration = 100, tol = 1e-5, step_size = 0.00005)
}
\arguments{
  \item{method}{A string specifying the method to be used. Currently, only "LASSO" is supported.}
  \item{data_temp}{A matrix of time series data where each column represents a different time series.}
  \item{lambda}{A vector of lambda values for LASSO regularization.}
  \item{p}{An integer specifying the order of the autoregressive model.}
  \item{cv_index}{An integer vector indicating the indices of the data points to be left out during each cross-validation iteration.}
  \item{max_iteration}{An optional integer specifying the maximum number of iterations for the LASSO algorithm. Default is 100.}
  \item{tol}{An optional tolerance level for the convergence of the LASSO algorithm. Default is 1e-5.}
  \item{step_size}{An optional step size for the LASSO algorithm. Default is 0.00005.}
}
\value{
  A list containing the following components:
  \item{brk_points}{A vector of indices indicating the identified break points in the time series data.}
  \item{cv}{A vector of cross-validation errors for each lambda value.}
  \item{cv_final}{The final selected lambda value after cross-validation.}
}
\examples{
# Example usage
# Assume data_temp is a matrix of time series data
# Assume lambda is a vector of lambda values
# Assume cv_index is a vector of indices for cross-validation
result <- first_step_cv_new("LASSO", data_temp, lambda, 1, cv_index)
}

\note{
  This function requires the Rcpp and RcppArmadillo packages.
}
\seealso{
  \code{\link[stats]{arima}}, \code{\link[glmnet]{cv.glmnet}}
}
\keyword{models}
\keyword{regression}
