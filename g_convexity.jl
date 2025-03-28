# This code demonstrates the convexity of the function t -> log( k( e^(tX) A e^(tY) ) ) 
# For a random matrix A and random symmetric matrices X,Y, plots the function t -> log( k( e^(tX) A e^(tY) ) )
# for the condition number k that arises from the operator and the Frobenius norm.

using Pkg
using Plots
using LinearAlgebra



A = rand([-3,3],100,100)
X = rand([-0.2,0.05],100,100)
Y = rand([0.05,0.15],100,100)
X = X + X'
Y = Y + Y'


function nuc_norm(B)
	return sum(diag(sqrt(B' * B)))
end

function log_op(t)
	return log(opnorm(exp(t*X)*A*exp(t*Y)))
end

function log_F(t)
	return log(norm(exp(t*X)*A*exp(t*Y)))
end

function log_nuc(t)
	return log(nuc_norm(exp(t*X)*A*exp(t*Y)))
end


t = range(-0.5,0.5,length=1000)
y1 = log_op.(t)
y2= log_F.(t)
#y3 = log_nuc.(t)
# println(test_convex())


p = plot(t,[y1 y2],label=["log-opnorm" "log-F-norm"], linewidth=1.5)
savefig(p,"myplot.pdf")
