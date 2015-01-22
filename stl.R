# self taught learning.

library(R.matlab)
library(functional)
library(pracma)
library(lbfgs)
library(Matrix)
library(R.utils)

source("common.R")

# step 0 parameters
inputsize     <- 28 * 28
numlabels     <- 5
hiddensize    <- 200
sparsityparam <- 0.1
lambda        <- 3e-3
beta          <- 3
maxiteration  <- 400

# step 1 load data from the mnist database
images <- loadimages("train-images-idx3-ubyte")
labels <- loadlabels("train-labels-idx1-ubyte")

# simulate a labeled and unlabeled set
labeledset   <- which(labels <= 4)
unlabeledset <- which(labels >= 5)

numtrain <- round(length(labeledset) / 2)
trainset <- labeledset[1:numtrain]
testset  <- labeledset[(numtrain + 1): length(labeledset)]

unlabeleddata <- images[unlabeledset,]
traindata     <- images[trainset,]
trainlabels   <- labels[trainset] + 1
testdata      <- images[testset,]
testlabels    <- labels[testset] + 1

printf("# examples in unlabeled set: %d\n", nrow(unlabeleddata))
printf("# examples in supervised training set: %d\n\n", nrow(traindata))
printf("# examples in supervised testing  set: %d\n\n", nrow(testdata))

# step 2: train the sparse autoencoder

train <- function() {
	theta <- initialize_parameters(hiddensize, inputsize)

	opt <- optim(theta,
	     Curry(sparse_autoencoder_cost, vis = inputsize,
	     hid = hiddensize, lambda = lambda, sparsity = sparsityparam,
	     beta = beta, patches = unlabeleddata, ret = 1),
	     Curry(sparse_autoencoder_cost, vis = inputsize,
	     hid = hiddensize, lambda = lambda, sparsity = sparsityparam,
	     beta = beta, patches = unlabeleddata, ret = 2),
	     method = "L-BFGS-B", control= c(trace = 1, maxit=maxiteration))
}

opt <- train()

trainfeatures <- t(feedforwardautoencoder(traindata, hiddensize, inputsize, opt$par))
testfeatures  <- t(feedforwardautoencoder(testdata,  hiddensize, inputsize, opt$par))

numclasses <- 5
inputsize  <- ncol(trainfeatures)
theta <- runif(numclasses * inputsize, 0, 0.005)
lambda <- 1e-4
optsoftmax <- optim(theta,
             Curry(softmax_cost,
                   k = numclasses, n = inputsize, lambda = lambda, x = trainfeatures,
                   y = trainlabels, ret = 1),
             Curry(softmax_cost,
                   k = numclasses, n = inputsize, lambda = lambda, x = trainfeatures,
                   y = trainlabels, ret = 2),
             method = "L-BFGS-B", control= c(trace = 1, maxit=400))
pred <- softmax_predict(optsoftmax$par, numclasses, testfeatures)
printf("%0.3f%%\n", sum(pred == testlabels) / length(testlabels) * 100)
