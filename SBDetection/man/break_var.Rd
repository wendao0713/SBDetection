\name{break_var}
\alias{break_var}
\title{Identify Breakpoints in Time Series Data using LASSO Regression}
\description{
  Applies LASSO regression to time series data to identify potential breakpoints. The function evaluates segments of the data defined by proposed breakpoints and applies LASSO regression to each segment.
}
\usage{
break_var(data, lambda, pts, p, max_iteration = 1000, tol = 1e-4, step_size = 1e-3)
}
\arguments{
  \item{data}{A matrix of time series data, where each column is a separate series.}
  \item{lambda}{A vector of regularization parameters for LASSO regression.}
  \item{pts}{A vector of proposed breakpoint positions in the time series.}
  \item{p}{An integer representing the order of the autoregression in the LASSO model.}
  \item{max_iteration}{Maximum number of iterations for the LASSO optimization algorithm. Default is 1000.}
  \item{tol}{Tolerance for convergence in the LASSO algorithm. Default is 1e-4.}
  \item{step_size}{Step size used in the optimization. Default is 1e-3.}
}
\value{
  A list with the following components:
  \itemize{
    \item \code{L_n}: A numeric value representing the sum of prediction errors across all segments of the time series after applying LASSO regression.
  }
}
\details{
  The function `break_var` divides the time series data into segments based on the proposed breakpoints. It then applies LASSO regression to each segment to evaluate the model's prediction error. The function returns the sum of these errors, providing a measure of the overall fit of the model with the given breakpoints. This can be useful for assessing the validity of proposed breakpoints in time series data.
}
\examples{
data <- matrix(rnorm(1000), ncol = 10)
lambda <- c(0.1, 0.2, 0.3)
pts <- c(50, 100, 150)
break_var(data, lambda, pts, 2)
}
