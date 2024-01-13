\name{soft}
\alias{soft}
\title{Weighted Soft Thresholding Function}
\description{
  Applies a weighted soft thresholding operation to each element of a matrix. 
  This function is similar to \code{\link{soft_full}}, but it adjusts the threshold for each element using a vector of weights.
}
\usage{
soft(L_in, weight, lambda)
}
\arguments{
  \item{L_in}{A matrix (usually one row) on which the soft thresholding operation is to be performed.}
  \item{weight}{A vector of weights to adjust the threshold \code{lambda} for each element.}
  \item{lambda}{The base threshold value.}
}
\value{
  A matrix with the weighted soft thresholding operation applied to each element.
}
\examples{
soft(matrix(1:10, nrow = 1), rep(1, 10), 5)
}
