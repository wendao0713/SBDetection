\name{pred}
\alias{pred}
\title{Time Series Prediction Using a Model Matrix}
\description{
  Generates predictions for a time series using a specified model matrix. This function is particularly useful in the context of time series analysis, where the model matrix (phi) is used to predict future values based on past observations.
}
\usage{
pred(Y, phi, p, T, k, h)
}
\arguments{
  \item{Y}{A matrix of time series data. Each column represents a different time series.}
  \item{phi}{A model matrix used for making predictions. This matrix should have been estimated beforehand.}
  \item{p}{An integer indicating the order of the model.}
  \item{T}{An integer representing the current time point up to which data is available.}
  \item{k}{The number of time series or variables in the matrix Y.}
  \item{h}{The number of steps ahead for which predictions are required.}
}
\value{
  A matrix containing the predicted values for the next \code{h} steps in the time series.
}
\details{
  The \code{pred} function extends the time series data \code{Y} by \code{h} steps using the model matrix \code{phi}. The predictions are based on the specified order \code{p} of the model and are computed for each of the \code{k} time series. The function assumes that the model matrix \code{phi} has been estimated appropriately for the time series data.
}
\examples{
Y <- matrix(rnorm(100), ncol = 10)
phi <- matrix(rnorm(20), ncol = 2)
predicted_values <- pred(Y, phi, 2, 50, 10, 5)
print(predicted_values)
}
