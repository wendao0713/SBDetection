
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

######## FUNCTIONS ##########################

# source("functions_SBDetection.R")

#############################################

#############################################

#############################################
######## DATA GENERATION ####################
#############################################

# for real data application, this part will be replaced by loading the data!
T <- 300; # number of time points
k <- 20; # number of time series

# TRUE BREAK POINTS WITH T+1 AS THE LAST ELEMENT
brk <- c(floor(T/3),floor(2*T/3),T+1)

m <- length(brk)
p.t <- 1; ## the true AR order
########## PHI GENERATION ###################
phi.full <- matrix(0,k,k*p.t*m)
aa <- 0.75; bb <- 0.75
for (j in 1:(k-1)){
  phi.full[j,((1-1)*p.t*k+j+1)] <- -0.6;
  phi.full[j,((2-1)*p.t*k+j+1)] <- 0.75;
  phi.full[j,((3-1)*p.t*k+j+1)] <- -0.8;
}

# print(plot.matrix(abs(phi.full),p=m,name="TRUTH"))

for(i in 2:m){
  phi.1 <- phi.full[, ((i-2)*k*p.t+1):((i-1)*k*p.t)];
  phi.2 <- phi.full[, ((i-1)*k*p.t+1):((i-0)*k*p.t)];
  print(sqrt( sum((phi.1-phi.2)^2)   ))
}


#############################################
######## DATA ANALYSIS   ####################
#############################################


###### METHODS ##############################
method.all <- c("VAR","LASSO","HVARC","HVAROO","HVARELEM","SSLASSO","SSHVARC","SSHVAROO","SSHVARELEM",
                "DHVAR","DHVARC","SSDHVAR","SSDHVARC");
method.all <- c("LASSO");
final.check <- matrix(0,length(method.all)+1,6)
final.check[2:(length(method.all)+1),1] <- method.all; 
phi.hat.all <- vector("list",length(method.all));
final.check[1,2] <- c("MSPE"); 
final.check[1,3] <- c("SD of MSPE"); 
final.check[1,4] <- c("MEDIAN of MSPE"); 
final.check[1,5] <- c("MEAN of RPE"); 
final.check[1,6] <- c("SD of RPE");
count.non <- 0; 
count.zero <- 0; 
p <- p.t;  
# for (i in 1:k){
#   for (j in 1:(k*p)){
#     if ( phi[i,j] != 0  ){count.non <- count.non + 1;}
#     if ( phi[i,j] == 0  ){count.zero <- count.zero + 1;}
#   }
# }
est.error <-  matrix(0,length(method.all)+2,9)
est.error[3:(length(method.all)+2),1] <- method.all; 
est.error[1,2]<-c("l2 est error"); 
est.error[1,3]<-c("SD of l2"); 
est.error[1,4] <- c("TZ"); 
est.error[2,4] <- c(count.zero); 
est.error[2,1] <- c("TRUTH");
est.error[1,5] <- c("TNZ"); 
est.error[2,5] <- c(count.non);
est.error[1,6] <- c("FZ"); 
est.error[1,7] <- c("FNZ"); 
est.error[1,8] <- c("lag.L.mean"); 
est.error[1,9] <- c("lag.L.sd"); 
#############################################
#############################################
N <- 1; # number of replicates
iter.final <- rep(0,length(method.all));
pts <- vector("list",N); 
data.full <- vector("list",N); 
pts.final <- vector("list",N);
phi.final <- vector("list",N); 
pts.final.sbs <- vector("list",N);


########### GENERAL PARAMETERS ##############
tol <- 4*10^(-2); # tolerance 
step.size <- 2*10^(-4); # step size 
max.iteration <- 100; # max number of iteration for the LASSo solution
p <- p.t; ## the selected AR order
sig <- matrix(0,k,k);
rho <- 0.5;
for(i in 1:k){for(j in 1:k){sig[i,j] <- rho^(abs(i-j))}}


# for ( j.1 in 1:N){

#############################################
j.1 = 1
set.seed(123456*j.1)

e.sigma <- as.matrix(0.01*diag(k));
source("var.break.fit.R")
try=var.sim.break(T,arlags=seq(1,p.t,1),malags=NULL,phi=phi.full,sigma=e.sigma,brk = brk)
data <- try$series
data <- as.matrix(data)
data.full[[j.1]] <- data;
#############################################
######################################################################
######## FIRST STEP : INITIAL BRK POINT SELECTION ####################
######################################################################
method<- c("LASSO"); # other types of penalization has not been coded yet, so the only option as of now is just LASSO!
lambda.1 <- 0.10; #The first tuning parameter 
lambda.1.cv <- seq(0.10,0.30,0.05); #The first tuning parameter
cv.length <- 10;
cv.index <- seq(p+cv.length, T-1, cv.length); # cv index


# debug
# 导入c++函数：first_step_cv_new
setwd("C://Users//36322//Desktop//23-24文件与资料//计算机基础//期末大作业//rpack//SBDetection//src")
Rcpp::sourceCpp("first_step_cv_new.cpp")
test = first_step_cv_new(method, data, lambda.1.cv, p, cv.index-1, max_iteration = 10, 0.01, 0.01) #转换为cpp函数后index需要-1
test$brk_points #表示经过第一步lasso方法筛选出的可能的断点
