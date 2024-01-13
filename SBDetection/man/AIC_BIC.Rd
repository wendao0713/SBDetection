\name{AIC_BIC}
\alias{AIC_BIC}
\title{Calculation of Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)}
\description{
  Calculates the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) for a given model.
}
\usage{
AIC_BIC(residual, phi)
}
\arguments{
  \item{residual}{A matrix of residuals from the model. Each column represents the residuals for a particular observation.}
  \item{phi}{A matrix representing the model parameters. Non-zero elements in this matrix represent the number of parameters in the model.}
}
\value{
  A list containing two elements:
  \itemize{
    \item \code{AIC}: The Akaike Information Criterion value for the model.
    \item \code{BIC}: The Bayesian Information Criterion value for the model.
  }
}
\examples{
residuals <- matrix(rnorm(100), ncol = 10)
phi <- matrix(rnorm(20), nrow = 2)
AIC_BIC(residuals, phi)
}
\details{
  The function counts the non-zero elements in the \code{phi} matrix to determine the number of parameters in the model. It then calculates the covariance matrix of the residuals (\code{sigma_hat}) and its log determinant. The AIC and BIC are calculated based on these values, taking into account the number of observations and parameters.
}
