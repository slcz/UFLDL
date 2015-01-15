library(R.matlab)
library(functional)
library(pracma)
library(lbfgs)

source("common.R")

sampleIMAGES <- function() {
	h <- file("train-images-idx3-ubyte", "rb")
	readBin(h, integer(), n=1, endian="big")
	n_images <- readBin(h, integer(), n=1, endian="big")
	row <- readBin(h, integer(), n=1, endian="big")
	col <- readBin(h, integer(), n=1, endian="big")
	cat("nimages = ", n_images, " row = ", row, " col = ", col, "\n")
	IMAGES <- matrix(ncol = row * col, nrow = n_images)
	for (i in 1:n_images) {
		m <- matrix(readBin(h, integer(), n=row*col, size=1, endian="big", signed = FALSE), nrow = row, byrow = TRUE) / 255
		IMAGES[i,] <- (as.vector(m))
	}
	close(h)
	IMAGES
}

train <- function () {
	visible_size <- 28 * 28 # number of input units
	hidden_size <- 196 # number of hidden units
	sparsity_param <- 0.1 # desired average activation of the hidden units.
	lambda <- 3e-3 # weight decay parameter
	beta <- 3 # weight of sparsity penalty term

	patches <- sampleIMAGES()[1:10000,]

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
