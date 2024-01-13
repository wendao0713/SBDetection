\name{varArmaBreak}
\alias{varArmaBreak}
\title{Simulate VARMA Model with Break Points}
\description{
  Simulates a vector autoregressive moving-average (VARMA) model with optional break points.
}
\usage{
varArmaBreak(nobs, sigma, arlags = NULL, malags = NULL, cnst = NULL, phi = NULL, theta = NULL, skip = 200, brk = NULL)
}
\arguments{
  \item{nobs}{An integer specifying the number of observations to simulate.}
  \item{sigma}{A matrix representing the covariance matrix of the innovations.}
  \item{arlags}{Optional. A numeric vector of autoregressive lags.}
  \item{malags}{Optional. A numeric vector of moving average lags.}
  \item{cnst}{Optional. A numeric vector representing the model constants.}
  \item{phi}{Optional. A numeric matrix of autoregressive coefficients.}
  \item{theta}{Optional. A numeric matrix of moving average coefficients.}
  \item{skip}{An integer specifying the number of initial values to discard. Default is 200.}
  \item{brk}{Optional. A numeric vector representing the break points in the time series.}
}
\value{
  A list containing:
  \itemize{
    \item{series}{A matrix representing the simulated time series data.}
    \item{noises}{A matrix representing the innovation terms used in the simulation.}
  }
}
\details{
  The function simulates a VARMA model with the ability to include break points. If break points are provided, the model coefficients change at these points. The function allows for customization of various parameters including the autoregressive and moving average lags, coefficients, and the covariance matrix of the innovations.
}
\examples{
# Example usage
nobs <- 10
