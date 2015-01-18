library(R.matlab)
library(functional)
library(pracma)
library(lbfgs)
library(Matrix)
library(R.utils)

source("common.R")

# ret = 1 => J(theta), ret = 2 => grad(J(theta))
softmax_cost <- function(theta, k, n, lambda, x, y, ret) {
	m <- nrow(x)
	t <- matrix(theta, nrow = k)
	a <- t %*% t(x)
	a <- exp(t(t(a) - apply(a, 2, max)))
	a <- t(t(a) / colSums(a))
	Y <- sparseMatrix(y, 1:m)
	if (ret == 1)
		# J(theta)
		-1/m * sum(log(colSums(Y * a))) + lambda / 2 * sum(t ^ 2)
	else
		# grad(J(theta))
		as.vector(-1/m * ((Y - a) %*% x) + lambda * t)
}

predict <- function (theta, k, x) {
	t <- matrix(theta, nrow = k)
	a <- t %*% t(x)
	a <- exp(t(t(a) - apply(a, 2, max)))
	htheta <- t(t(a) / colSums(a))
	apply(htheta, 2, which.max)
}

train <- function () {
	numclasses <- 10
	inputsize <- 28 * 28
	lambda <- 1e-4
	images <- loadimages("train-images-idx3-ubyte")
	labels <- loadlabels("train-labels-idx1-ubyte")
	labels <- sapply(labels, function(x) { if (x == 0) 10 else x })
	range <- 1:nrow(images)
	images <- images[range,]
	labels <- labels[range]

	theta <- runif(numclasses * inputsize, 0, 0.005)
	#grad <- softmax_cost(theta, numclasses, inputsize, lambda, images, labels, 2)
 	#compute_numerical_gradient(Curry(softmax_cost,
 	#    k = numclasses, n = inputsize, lambda = lambda, x = images, y = labels, ret = 1),
	#    theta, grad)
 	opt <- optim(theta,
	             Curry(softmax_cost,
	                   k = numclasses, n = inputsize, lambda = lambda, x = images,
	                   y = labels, ret = 1),
	             Curry(softmax_cost,
	                   k = numclasses, n = inputsize, lambda = lambda, x = images,
	                   y = labels, ret = 2),
  		     method = "L-BFGS-B", control= c(trace = 1, maxit=400))
	images <- loadimages("t10k-images-idx3-ubyte")
	labels <- loadlabels("t10k-labels-idx1-ubyte")
	labels <- sapply(labels, function(x) { if (x == 0) 10 else x })
	pred <- predict(opt$par, numclasses, images)
	printf("%0.3f%%\n", sum(pred == labels) / length(labels) * 100)
	opt
}
