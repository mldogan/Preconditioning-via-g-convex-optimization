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

function test_convex()
	for i in 1:1000
		t_0 = rand([-1,1])
		t_1 = rand([-1,1])
		t_m = (t_0 + t_1)*0.5
		if log_op(t_m) > (log_op(t_1) + log_op(t_0)) * 0.5
			println("Operators failed me")
			println(t_0,t_1,t_m)
			return false
		elseif log_F(t_m) > (log_F(t_1) + log_F(t_0)) * 0.5
			println("No way!")
			println(t_0,t_1,t_m)
			return false
		elseif log_nuc(t_m) > (log_nuc(t_1) + log_nuc(t_0)) * 0.5
			println("Nuked!")
			println(t_0,t_1,t_m)
			return false
		end
	end
	return true
end


t = range(-0.5,0.5,length=1000)
y1 = log_op.(t)
y2= log_F.(t)
#y3 = log_nuc.(t)
# println(test_convex())


p = plot(t,[y1 y2],label=["log-opnorm" "log-F-norm"], linewidth=1.5)
savefig(p,"myplot.pdf")