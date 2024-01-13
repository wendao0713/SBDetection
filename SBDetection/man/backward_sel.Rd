\name{backward_sel}
\alias{backward_sel}
\title{Backward Selection for Time Series Data with Potential Breakpoints and LASSO Regularization}
\description{
  Performs backward selection on time series data to identify potential breakpoints using LASSO regularization. The function iteratively evaluates the impact of removing each potential breakpoint on the model's performance.
}
\usage{
backward_sel(data, lambda, pts, p, max_iteration = 1000, tol = 1e-4, step_size = 1e-3)
}
\arguments{
  \item{data}{A numeric matrix representing the time series data. Each column is a time series.}
  \item{lambda}{A numeric vector of regularization parameters for the LASSO model.}
  \item{pts}{A numeric vector indicating the initial set of potential breakpoints in the time series.}
  \item{p}{An integer specifying the order of the autoregressive model.}
  \item{max_iteration}{An integer specifying the maximum number of iterations for the LASSO algorithm. Default is 1000.}
  \item{tol}{A numeric value specifying the tolerance for convergence in the LASSO algorithm. Default is 1e-4.}
  \item{step_size}{A numeric value specifying the step size in the LASSO algorithm. Default is 1e-3.}
}
\value{
  A list containing the following elements:
  \itemize{
    \item \code{L_n}: A numeric vector of the loss function values after removing each potential breakpoint.
    \item \code{L_n_curr}: A numeric value representing the loss function value of the current model with all potential breakpoints included.
  }
}
\examples{
# Example usage
data <- matrix(rnorm(1000), ncol = 10)
lambda <- c(0.1, 0.2, 0.3)
pts <- c(50, 100, 150)
backward_sel(data, lambda, pts, 2)
}
\details{
  The \code{backward_sel} function evaluates a time series dataset for potential breakpoints using an autoregressive model with LASSO regularization. It applies a backward selection approach, iteratively removing each breakpoint and recalculating the model's loss function. This helps in understanding the impact of each potential breakpoint on the overall model performance. The function returns the loss function values after removing each breakpoint and the loss value with all breakpoints included.
}
