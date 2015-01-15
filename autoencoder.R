library(R.matlab)
library(functional)
library(pracma)
source("common.R")

sampleIMAGES <- function() {
	IMAGES <- readMat("IMAGES.mat")[[1]]
	IMAGE  <- IMAGES[, , runif(1,1, dim(IMAGES)[3])]

	patchsize <- 8
	numpatches <- 10000

	patches <- matrix(nrow = numpatches, ncol = patchsize * patchsize)
	for (i in 1:numpatches) {
		r <- runif(1, 1, nrow(IMAGE) - patchsize)
		c <- runif(1, 1, ncol(IMAGE) - patchsize)
		patches[i, ] <- as.vector(IMAGE[r:(r + patchsize - 1), c:(c + patchsize - 1)])
	}
	patches <- normalize_data(patches)
}

train <- function () {
	visible_size <- 8 * 8 # number of input units
	hidden_size <- 25 # number of hidden units
	sparsity_param <- 0.01 # desired average activation of the hidden units.
	lambda <- 0.0001 # weight decay parameter
	beta <- 3 # weight of sparsity penalty term

	#patches <- sampleIMAGES()[1:10000,]
	patches <- sampleIMAGES()

	theta <- initialize_parameters(hidden_size, visible_size)
	#grad <- sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda,
	#                                sparsity_param, beta, patches, ret = 2)
	#compute_numerical_gradient(Curry(sparse_autoencoder_cost,
	#    vis = visible_size, hid = hidden_size, lambda = lambda,
	#    sparsity = sparsity_param, beta = beta, patches = patches, ret = 1), theta, grad)
	opt <- optim(theta, Curry(sparse_autoencoder_cost,
 		     vis = visible_size, hid = hidden_size, lambda = lambda,
 	             sparsity = sparsity_param, beta = beta, patches = patches, ret = 1),
 	             Curry(sparse_autoencoder_cost,
 		     vis = visible_size, hid = hidden_size, lambda = lambda,
 	             sparsity = sparsity_param, beta = beta, patches = patches, ret = 2),
 		     method = "L-BFGS-B", control= c(trace = 1, maxit=400))
	display_network(matrix(opt$par[1:(hidden_size * visible_size)], nrow = hidden_size), sqrt(hidden_size))
	opt
}
