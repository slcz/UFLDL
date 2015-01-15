library(R.matlab)
library(functional)
library(pracma)

normalize_data <- function(patches) {
	# Squash data to [0.1, 0.9] since we use sigmoid as the activation
	# function in the output layer

	# Remove DC (mean of images). 
	patches <- patches - rowMeans(patches)

	# Truncate to +/-3 standard deviations and scale to -1 to 1
	pstd <- 3 * std(patches);
	patches <- matrix(pmax(pmin(pstd, patches), -pstd) / pstd, nrow=nrow(patches))

	# Rescale from [-1,1] to [0.1,0.9]
	patches <- (patches + 1) * 0.4 + 0.1;
}

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

display_network <- function(A, cols) {
	A  <- A - mean(A)
	sz <- sqrt(ncol(A))
	M <- nrow(A)
	n <- cols
	m <- ceil(M / n)
	buf <- 1
	array <- matrix(data = rep(-1, (buf + m * (sz + buf)) * (buf + n * (sz + buf))),
	    nrow = buf + m * (sz + buf))
	k <- 1
	for (i in 1:m) {
		for (j in 1:n) {
			if (k > M)
				next
			clim <- max(abs(A[k,]))
			array[buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz)] <-
			    matrix(A[k,],nrow = sz, ncol = sz) / clim;
			k <- k + 1
		}
	}
	image(t(array[nrow(array):1,]), col = gray.colors(256, start=0, end=1.0))
}

initialize_parameters <- function(hidden_size, visible_size) {
	r <- sqrt(6) / sqrt(hidden_size + visible_size + 1)
	W1 <- matrix(runif(hidden_size * visible_size, -r, r), nrow = hidden_size)
	W2 <- matrix(runif(hidden_size * visible_size, -r, r), nrow = visible_size)
	b1 <- rep(0, hidden_size)
	b2 <- rep(0, visible_size)
	c(W1, W2, b1, b2)
}

sparse_autoencoder_cost <- function(theta, vis, hid, lambda, sparsity, beta, patches, ret) {
	t      <- 1
	W1     <- matrix(theta[t:(t - 1 + vis * hid)], nrow = hid)
	t      <- t + vis * hid
	W2     <- matrix(theta[t:(t - 1 + vis * hid)], nrow = vis)
	t      <- t + vis * hid
	b1     <- theta[t:(t - 1 + hid)]
	t      <- t + hid
	b2     <- theta[t:(t - 1 + vis)]
	stopifnot(t + vis == length(theta) + 1)
	# feed forward pass
	z2     <- W1 %*% t(patches) + b1
	a2     <- sigmoid(z2)
	div2   <- a2 * (1 - a2)
	z3     <- W2 %*% a2 + b2
	a3     <- sigmoid(z3)
	rho_hat <- rowMeans(a2)
	KL     <- sum(sparsity * log(sparsity / rho_hat) +
	              (1 - sparsity) * log((1 - sparsity) / (1 - rho_hat)))
	J      <- mean(apply(a3 - t(patches), 2, function(x) { t(x) %*% x }) / 2)
	J      <- J + lambda / 2 * (sum(W1 ^ 2) + sum(W2 ^2)) + beta * KL
	if (ret == 1)
		return(J)
	div3   <- a3 * (1 - a3)
	# for output layer
	delta3 <- -(t(patches) - a3) * div3
	# for hidden layer(s)
	delta2 <- (t(W2) %*% delta3 + beta * (-sparsity/rho_hat + (1-sparsity)/(1-(rho_hat)))) * div2
	b1grad <- rowMeans(delta2)
	b2grad <- rowMeans(delta3)
	W1grad <- (delta2 %*% patches) / ncol(delta2) + lambda * W1
	W2grad <- (delta3 %*% t(a2)) / ncol(delta3) + lambda * W2
	c(W1grad, W2grad, b1grad, b2grad)
}

cost <- function(theta, vis, hid, lambda, sparsity, beta, patches) {
	sparse_autoencoder_cost(theta, vis, hid, lambda, sparsity, beta, patches)[[1]]
}

grad <- function(theta, vis, hid, lambda, sparsity, beta, patches) {
	sparse_autoencoder_cost(theta, vis, hid, lambda, sparsity, beta, patches)[[2]]
}

compute_numerical_gradient <- function(FUNC, theta, grad) {
	epsilon <- 1e-4
	g <- vector(length = length(theta))
	for (i in 1:length(theta)) {
print(i)
		theta[i] <- theta[i] + epsilon
		J1 <- FUNC(theta)
		theta[i] <- theta[i] - 2 * epsilon
		J2 <- FUNC(theta)
		theta[i] <- theta[i] + epsilon
		g[i] <- (J1 - J2) / (2 * epsilon)
	}
	print(grad - g)
	print(sqrt(sum((g - grad) ^ 2)) / sqrt(sum((g + grad) ^ 2)))
}

check_numerical_gradient <- function() {
	x <- c(4, 10)
	grad <- c(2 * x[1] + 3 * x[2], 3 * x[1])
	compute_numerical_gradient(function(x) { x[1] ^ 2 + 3 * x[1] * x[2] }, x, grad)
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
	#opt <- lbfgs(Curry(sparse_autoencoder_cost,
	#	     vis = visible_size, hid = hidden_size, lambda = lambda,
	#             sparsity = sparsity_param, beta = beta, patches = patches, ret = 1),
	#             Curry(sparse_autoencoder_cost,
	# 	     vis = visible_size, hid = hidden_size, lambda = lambda,
	#             sparsity = sparsity_param, beta = beta, patches = patches, ret = 2),
	#             theta)
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
