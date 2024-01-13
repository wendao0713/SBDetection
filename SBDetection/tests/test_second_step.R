rm(list=ls(all=TRUE))

library("glmnet")
library("matrixcalc")
library("fields")
library("vars")
library("MTS")
library("mvtnorm")
library("xtable")
library("lattice")
library(ggplot2)

source("functions_SBDetection.R")
library(Rcpp)
sourceCpp("second_step.cpp")

# for real data application, this part will be replaced by loading the data!
#############################################
######## DATA GENERATION ####################
#############################################
T <- 300; # number of time points
k <- 20; # number of time series
# TRUE BREAK POINTS WITH T+1 AS THE LAST ELEMENT
brk <- c(floor(T/3),floor(2*T/3),T+1)
m <- length(brk)
p.t <- 1; ## the true AR order

########## PHI&DATA GENERATION ###################
phi.full <- matrix(0,k,k*p.t*m)
aa <- 0.75; bb <- 0.75
for (j in 1:(k-1)){
  phi.full[j,((1-1)*p.t*k+j+1)] <- -0.6;
  phi.full[j,((2-1)*p.t*k+j+1)] <- 0.75;
  phi.full[j,((3-1)*p.t*k+j+1)] <- -0.8;
}

e.sigma <- as.matrix(0.01*diag(k));
try=var.sim.break(T,arlags=seq(1,p.t,1),malags=NULL,phi=phi.full,sigma=e.sigma,brk = brk)
data <- try$series
data <- as.matrix(data)


######################################################################
######## SECOND STEP : SCREENING                  ####################
######################################################################
########### 待用参数 #############
T = 300;
tol = 0.04; # tolerance 
p = 1; # the selected AR order
n = T - p;
fisrt.brk.points <- c(13, 17, 21, 31, 38, 55, 67, 72, 93, 102, 109, 113, 
  120, 143, 153, 167, 182, 187, 193, 196);
omega <- 9.758287; # the penalty term in the information criterion -- the higher omega, the smaller number of break points selected.
lambda.2 <- c(0.05711372); # the second tuning parameter. This default number seems to be working for many simulation and real data examples!

######## 分别运行 ##########################
## R版本的函数
# temp_r <- second.step(data, lambda = lambda.2, p, max.iteration = 1000, tol = tol, step.size = 10^(-3), fisrt.brk.points, omega);
# final.brk.points_r <- temp_r$pts;
# final.brk.points_r;
## C++版本的函数
temp_c <- second_step(data, lambda = lambda.2, pts=fisrt.brk.points, omega, p, max_iteration = 1000,
            tol = 1e-4, step_size = 1e-3)
final.brk.points_c <- temp_c$pts;
final.brk.points_c;

# plotting the data with selected break points
MTSplot(data)
abline(v=final.brk.points_c)

######## 测速 ##########################
library(microbenchmark)
testResult = microbenchmark(second_step(data, lambda = lambda.2, pts=fisrt.brk.points, omega, p, max_iteration = 1000,tol = 1e-4, step_size = 1e-3),
                            second.step(data, lambda = lambda.2, p, max.iteration = 1000, tol = tol, step.size = 10^(-3), fisrt.brk.points, omega),
                            times = 50) 
testResult;


