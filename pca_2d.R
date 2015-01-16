library(ggplot2)
library(grid)

raw <- as.matrix(read.table("pcaData.txt"))
sigma <- 1 / ncol(raw) * (raw %*% t(raw))
t <- svd(sigma)
data <- data.frame(x = raw[1,], y = raw[2,])
p <- ggplot(data, aes(x=x, y=y)) + geom_point(shape = 0, size = 4)
v1 <- cbind(c(0,0), t$u[1,] / sqrt(sum(v1^2)) / 2)
v2 <- cbind(c(0,0), t$u[2,] / sqrt(sum(v2^2)) / 2)
p <- p + geom_line(data=data.frame(x=v1[1,], y=v1[2,]), aes(x=x, y=y), arrow=arrow(ends="first"))
p <- p + geom_line(data=data.frame(x=v2[1,], y=v2[2,]), aes(x=x, y=y), arrow=arrow(ends="first"))

xrot <- t(t$u) %*% raw
p <- ggplot(data = data.frame(x = xrot[1,], y=xrot[2,]), aes(x=x, y=y)) + geom_point(shape = 0, size = 4)

k <- 1
xtilde <- t(t$u[,1:k]) %*% raw
xhat <- t$u %*% rbind(xtilde, 0)
p <- ggplot(data = data.frame(x=xhat[1,], y=xhat[1,]), aes(x=x, y=y)) + geom_point(shape = 0, size = 4)

epsilon <- 1e-5
xpcawhite <- diag(diag(1/sqrt(diag(t$d) + epsilon))) %*% t(t$u) %*% raw
p <- ggplot(data = data.frame(x=xpcawhite[1,], y=xpcawhite[2,]), aes(x=x, y=y)) + geom_point(shape = 0, size = 4)

zpcawhite <- t$u %*% xpcawhite
p <- ggplot(data = data.frame(x=zpcawhite[1,], y=zpcawhite[2,]), aes(x=x, y=y)) + geom_point(shape = 0, size = 4)
