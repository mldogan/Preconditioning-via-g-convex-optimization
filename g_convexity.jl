# This code demonstrates the convexity of the function t -> log( k( e^(tX) A e^(tY) ) ) 
# For a random matrix A and random symmetric matrices X,Y, plots the function t -> log( k( e^(tX) A e^(tY) ) )
# for the condition number k that arises from the operator and the Frobenius norm.


#uncomment below if it does not compile
#using Pkg
#Pkg.add("Plots")
#Pkg.add("LinearAlgebra")
#Pkg.add("LaTeXStrings")
using Plots
using LinearAlgebra
using LaTeXStrings


A = rand([-3,3],100,100)  # random 100x100 matrix with entries drawn uniformly and independently from the interval
X = rand([-0.2,0.05],100,100)  # random 100x100 matrix with entries drawn uniformly and independently from the interval
Y = rand([0.05,0.15],100,100)  # random 100x100 matrix with entries drawn uniformly and independently from the interval
X = X + X' # symmetrize X
Y = Y + Y' # symmetrize Y

function nuc_norm(B)
	return sum(diag(sqrt(B' * B)))
end

#logarithm of the operator norm of e^(tX) A e^(tY)
function log_op(t)
	return log(opnorm(exp(t*X)*A*exp(t*Y)))
end

#logarithm of the Frobenius norm of e^(tX) A e^(tY)
function log_F(t)
	return log(norm(exp(t*X)*A*exp(t*Y)))
end

##logarithm of the nuclear norm (Schatten 1) of e^(tX) A e^(tY)
function log_nuc(t)
	return log(nuc_norm(exp(t*X)*A*exp(t*Y)))
end


t = range(-0.5,0.5,length=1000) # a range of length many numbers from the interval
y1 = log_op.(t) 
y2= log_F.(t)
#y3 = log_nuc.(t)

# below we plot log_F(t) and log_op(t) over the given interval for t
p = plot(t,[y1 y2],label=[L"\log \kappa (e^{tX}A e^{tY})" L"\log {\kappa}_F (e^{tX}A e^{tY})"], linewidth=1.5)
savefig(p,"myplot.pdf")
