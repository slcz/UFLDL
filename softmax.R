library(R.matlab)
library(functional)
library(pracma)
library(lbfgs)
library(Matrix)
library(R.utils)

source("common.R")

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
	pred <- softmax_predict(opt$par, numclasses, images)
	printf("%0.3f%%\n", sum(pred == labels) / length(labels) * 100)
	opt
}
