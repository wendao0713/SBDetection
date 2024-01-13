\name{soft_full}
\alias{soft_full}
\title{Soft Thresholding Function}
\description{
  Applies a soft thresholding operation to each element of a matrix. 
  This function is typically used in statistical methods like LASSO.
}
\usage{
soft_full(L, lambda)
}
\arguments{
  \item{L}{A matrix on which the soft thresholding operation is to be performed.}
  \item{lambda}{The threshold value.}
}
\value{
  A matrix with the soft thresholding operation applied to each element.
}
\examples{
soft_full(matrix(1:10, nrow = 2), 5)
}
