library(R.matlab)
library(functional)
library(pracma)
library(grid)
source("common.R")

imagechannels <- 3
patchdim      <- 8
numpatches    <- 100000
visiblesize   <- patchdim * patchdim * imagechannels
outputsize    <- visiblesize
hiddensize    <- 400
sparsityparam <- 0.035

lambda        <- 3e-3
beta          <- 5

epsilon       <- 0.1

# step 1 create and modify sparseautoencoderlinearcost to use a linear decoder
#debughiddensize  <- 5
#debugvisiblesize <- 8
#patches          <- matrix(runif(8 * 10), nrow = 10)
#theta            <- initialize_parameters(debughiddensize, debugvisiblesize)

#grad <- sparse_autoencoder_linearcost(theta, vis = debugvisiblesize, hid = debughiddensize,
#    lambda = lambda, sparsity = sparsityparam, beta = beta, patches = patches, ret = 2)
#
#numgrad <- compute_numerical_gradient(
#    Curry(sparse_autoencoder_linearcost, vis = debugvisiblesize, hid = debughiddensize,
#          lambda = lambda, sparsity = sparsityparam, beta = beta, patches = patches, ret = 1),
#    theta, grad)

# step 2 learn features on small patches
patches <- readMat("stl10_patches/stlSampledPatches.mat")[[1]]
patches <- apply(patches, 2, '-', rowMeans(patches))
sigma <- patches %*% t(patches) / numpatches
t <- svd(sigma)
zcawhite <- t$u %*% diag(diag(1/sqrt(diag(t$d) + epsilon))) %*% t(t$u)
patches <- zcawhite %*% patches
theta   <- initialize_parameters(hiddensize, visiblesize)
f <- Curry(sparse_autoencoder_linearcost, vis = visiblesize, hid = hiddensize,
           lambda = lambda, sparsity = sparsityparam, beta = beta, patches = t(patches))
opt <- optim(theta, Curry(f, ret = 1), Curry(f, ret = 2), method = "L-BFGS-B", control= c(trace = 1, maxit=400))
W <- matrix(opt$par[1:(visiblesize * hiddensize)], nrow = hiddensize)
b <- opt$par[(visiblesize * hiddensize + 1) : (visiblesize * hiddensize + hiddensize)]
display_color_network(t(W %*% zcawhite))
