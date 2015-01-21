# Stacked Autoencoder Exercise

library(R.matlab)
library(functional)
library(pracma)
library(lbfgs)
library(Matrix)
library(R.utils)

source("common.R")

# step 0 parameters
inputsize     <- 28 * 28
numclasses    <- 10
hiddensizel1  <- 200
hiddensizel2  <- 200
sparsityparam <- 0.1
lambda        <- 3e-3
beta          <- 3

# step 1 load data from the mnist database
traindata   <- loadimages("train-images-idx3-ubyte")
trainlabels <- loadlabels("train-labels-idx1-ubyte")
trainlabels <- sapply(trainlabels, function(x) { if (x == 0) 10 else x })

# step 2 train the first sparse autoencoder
train <- function(hiddensize, inputsize, data) {
	maxiteration <- 100
	theta <- initialize_parameters(hiddensize, inputsize)

	opt <- optim(theta,
	     Curry(sparse_autoencoder_cost, vis = inputsize,
	     hid = hiddensize, lambda = lambda, sparsity = sparsityparam,
	     beta = beta, patches = data, ret = 1),
	     Curry(sparse_autoencoder_cost, vis = inputsize,
	     hid = hiddensize, lambda = lambda, sparsity = sparsityparam,
	     beta = beta, patches = data, ret = 2),
	     method = "L-BFGS-B", control= c(trace = 1, maxit=maxiteration))
}

sae1theta <- train(hiddensizel1, inputsize, traindata)
# step 3 train the second sparse autoencoder
sae1features <- t(feedforwardautoencoder(traindata, hiddensizel1, inputsize, sae1theta$par))
sae2theta <- train(hiddensizel2, hiddensizel1, sae1features)

# step 4 train the softmax classifier
sae2features <- t(feedforwardautoencoder(sae1features, hiddensizel2, hiddensizel1, sae2theta$par))

numclasses <- 10
inputsize  <- ncol(sae2features)
saesoftmaxtheta <- runif(hiddensizel2 * numclasses, 0, 0.005)
lambda <- 1e-4
optsoftmax <- optim(saesoftmaxtheta,
                    Curry(softmax_cost,
                        k = numclasses, n = inputsize, lambda = lambda, x = sae2features,
                        y = trainlabels, ret = 1),
                    Curry(softmax_cost,
                        k = numclasses, n = inputsize, lambda = lambda, x = sae2features,
                        y = trainlabels, ret = 2),
                    method = "L-BFGS-B", control= c(trace = 1, maxit=400))
saesoftmaxtheta <- optsoftmax$par

finetune_cost <- function (theta, input, hid, numclasses, netconfig, lambda, data, label, ret) {
	softmaxtheta <- matrix(theta[1:(hid * numclasses)], nrow = numclasses)
	rest <- theta[(hid * numclasses + 1) : length(theta)]
	W <- NULL
	b <- NULL
	for (i in 2:length(netconfig)) {
		W[[i - 1]] <- matrix(rest[1:(netconfig[i-1] * netconfig[i])], nrow = netconfig[i])
		rest <- rest[(netconfig[i-1] * netconfig[i] + 1) : length(rest)]
		b[[i - 1]] <- rest[1:netconfig[i]]
		rest <- rest[(netconfig[i] + 1) : length(rest)]
	}

	d <- t(data)
	z <- NULL
	a <- NULL
	for (i in 2:length(netconfig)) {
		z[[i-1]] <- W[[i-1]] %*% d + b[[i-1]]
		a[[i-1]] <- sigmoid(z[[i-1]])
		d <- a[[i-1]]
	}

	t <- softmaxtheta %*% a[[length(netconfig)-1]]
	t <- exp(t(t(t) - apply(t, 2, max)))
	t <- t(t(t) / colSums(t))
	m <- nrow(data)
	y <- sparseMatrix(label, 1:m)
	cost <- -1/m * sum(log(colSums(y * t))) + lambda / 2 * sum(softmaxtheta ^ 2)
	if (ret == 1)
		return (cost)
	softmaxgrad <- as.vector(-1/m * ((y - t) %*% t(a[[length(netconfig)-1]])) + lambda * softmaxtheta)
	delta <- lapply(1:(length(netconfig)-1), function(i) 0)
	delta[[length(netconfig) - 1]] <- -(t(softmaxtheta) %*% (y-t)) *
	        (a[[length(netconfig)-1]] * (1 - a[[length(netconfig)-1]]))
	for (i in (length(netconfig)-1) : 2)
		delta[[i-1]] <- t(W[[i]]) %*% delta[[i]] * a[[i-1]] * (1-a[[i-1]])
	param <- softmaxgrad
	for (i in 2:length(netconfig)) {
		if (i == 2)
			dt <- t(data)
		else
			dt <- a[[i-2]]
		gradnew <- delta[[i-1]] %*% t(dt) / m
		bnew <- rowMeans(delta[[i-1]])
		param <- c(param, as.vector(gradnew))
		param <- c(param, bnew)
	}
	return (param)
}

checkstackedaecost <- function() {
	inputsize <- 4
	hiddensize <- 5
	lambda <- 0.01
	data <- matrix(runif(5 * inputsize, 0, 1), nrow = 5)
	labels <- c(1, 2, 1, 2, 1)
	numclasses <- 2
	params <- c(runif(inputsize * 3, 0, 0.1), rep(0, 3),
	    runif(hiddensize * 3, 0, 0.1), rep(0, hiddensize))
	softmaxtheta <- runif(hiddensize * numclasses, 0, 0.005)
	params <- c(softmaxtheta, params)
	netconfig <- c(inputsize, 3, hiddensize)
	grad <- finetune_cost(params, input = inputsize, hid = hiddensize,
	        numclasses = numclasses, netconfig = netconfig, lambda = lambda,
	        data = data, label = labels, ret = 2)
	print(grad)
	numgrad <- compute_numerical_gradient(
	    Curry(finetune_cost, input = inputsize, hid = hiddensize,
	          numclasses = numclasses, netconfig = netconfig, lambda = lambda,
	          data = data, label = labels, ret = 1), params, grad)
}

# step 5 finetune softmax model
netconfig <- c(28 * 28, hiddensizel1, hiddensizel2)
params <- c(sae1theta$par[1:(netconfig[1] * netconfig[2])],
            sae1theta$par[(2 * netconfig[1] * netconfig[2] + 1):
            (2 * netconfig[1] * netconfig[2] + netconfig[2])],
            sae2theta$par[1:(netconfig[2] * netconfig[3])],
            sae2theta$par[(2 * netconfig[2] * netconfig[3] + 1):
            (2 * netconfig[2] * netconfig[3] + netconfig[3])])
params <- c(saesoftmaxtheta, params)
optfinetune <- optim(params,
                    Curry(finetune_cost,
                          input = netconfig[1], hid = netconfig[length(netconfig)],
                          numclasses = numclasses, netconfig = netconfig, lambda = lambda,
                          data = traindata, label = trainlabels, ret = 1),
                    Curry(finetune_cost,
                          input = netconfig[1], hid = netconfig[length(netconfig)],
                          numclasses = numclasses, netconfig = netconfig, lambda = lambda,
                          data = traindata, label = trainlabels, ret = 2),
                    method = "L-BFGS-B", control= c(trace = 1, maxit=100))

# step 6 test

testdata <- loadimages("t10k-images-idx3-ubyte")
testlabels <- loadlabels("t10k-labels-idx1-ubyte")
testlabels <- sapply(testlabels, function(x) { if (x == 0) 10 else x })

softmaxtheta <- optfinetune$par[1:(numclasses * netconfig[length(netconfig)])]
rest <- optfinetune$par[(numclasses * netconfig[length(netconfig)] + 1) : length(optfinetune$par)]
finetunetheta1 <- rest[1:(netconfig[1] * netconfig[2] + netconfig[2])]
rest <- rest[(netconfig[1] * netconfig[2] + netconfig[2] + 1): length(rest)]
finetunetheta2 <- rest

test1features <- t(feedforwardautoencoder2(testdata, netconfig[2],
                                           netconfig[1], finetunetheta1))
test2features <- t(feedforwardautoencoder2(test1features, netconfig[3],
                                           netconfig[2], finetunetheta2))
pred <- softmax_predict(softmaxtheta, numclasses, test2features)
printf("%0.3f%%\n", sum(pred == testlabels) / length(testlabels) * 100)
