\name{ar_est}
\alias{ar_est}
\title{Estimation of Autoregressive Models with Breakpoints and LASSO Regularization}
\description{
  Estimates parameters of autoregressive models with potential breakpoints using LASSO regularization. The function incorporates various user-defined parameters and tuning options.
}
\usage{
ar_est(method, data, weight, lambda, p, break_pts, r_n, 
       max_iteration = 1000, tol = 1e-4, step_size = 1e-3)
}
\arguments{
  \item{method}{A string specifying the estimation method. Currently, only 'LASSO' is supported.}
  \item{data}{A numeric matrix of time series data. Each column represents a variable.}
  \item{weight}{An optional numeric vector of weights for regularization.}
  \item{lambda}{A numeric vector of regularization parameters for LASSO.}
  \item{p}{An integer indicating the order of the autoregressive model.}
  \item{break_pts}{A numeric vector indicating the potential breakpoints in the time series.}
  \item{r_n}{An integer indicating the range around breakpoints to be excluded from the model.}
  \item{max_iteration}{Maximum number of iterations for the LASSO algorithm. Default is 1000.}
  \item{tol}{Tolerance level for the convergence of the LASSO algorithm. Default is 1e-4.}
  \item{step_size}{Step size for the LASSO algorithm. Default is 1e-3.}
}
\value{
  A list with the following components:
  \itemize{
    \item \code{phi_hat}: A matrix of estimated coefficients for the autoregressive model.
    \item \code{iter}: A matrix indicating the number of iterations taken for each lambda value.
    \item \code{pred_error}: A vector of prediction errors for each lambda value.
    \item \code{tune.final}: The lambda value selected as optimal.
  }
}
\examples{
# Example usage
data <- matrix(rnorm(1000), ncol = 10)
lambda <- c(0.1, 0.2, 0.3)
break_pts <- c(50, 100)
ar_est("LASSO", data, NULL, lambda, 2, break_pts, 5)
}
\details{
  The function \code{ar_est} fits an autoregressive model to the provided time series data. It allows for the incorporation of breakpoints in the time series, where the model can change. The LASSO method is used for regularization, which is controlled by the \code{lambda} parameter. The function returns the estimated coefficients, the number of iterations taken, the prediction errors for each lambda, and the optimal lambda value.
}
