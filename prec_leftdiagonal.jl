function k_F(A)
    return norm(A) * norm(pinv(A))
end

using LinearAlgebra
println("Reading data...")
left = true
include("data/housing_scale_f.jl")
println("Reading data complete")
if !left
    A = A'
end
println(size(A))

global block_size = 5
n = size(A)[1]
rem = n % block_size

if size(A)[1] == size(A)[2]
    A = A[1:end-rem, 1:end-rem]
else
    A = A[1:end-rem, 1:end]
end

C = pinv(A)
n = size(A)[1]
B = A * A'
D = C' * C

F_norm = norm(A)
F_norm_inv = norm(C)
k = F_norm * F_norm_inv
op_norm = opnorm(A)
op_norm_inv = opnorm(C)
k_op = op_norm * op_norm_inv
svd_min = minimum(svdvals(A))

println("The Frobenius norm is $F_norm")
println("The operator norm is $op_norm")
println("The Frobenius condition number is $k")
println("The condition number is $k_op")
println("The minimum singular value is $svd_min")


function f(M,p) 
    Dg = Diagonal(p)
    return log(norm(Dg*A)) + log(norm(C*inv(Dg)))
end

function grad_f(M,p)
    Dg = Diagonal(p)
    X = (Dg * B * Dg) / (norm(Dg*A)^2)
    Y = (inv(Dg) * D * inv(Dg)) / ( norm(C*inv(Dg))^2 )
    return vec(diag(X)) - vec(diag(Y))
end


using Manifolds, Manopt


M = PositiveVectors(n)
U = vec(ones(n,1))
println(is_point(M,U))
println(is_vector(M,U,grad_f(M,U)))


opt1 = trust_regions(M, f, grad_f, U;
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.20f | "), "\n", :Stop, 20],
    stopping_criterion = StopWhenGradientNormLess(1e-6) | StopAfterIteration(100)
)

A_opt = Diagonal(opt1) * A
C_opt = pinv(A_opt)
new_F_norm = norm(A_opt)
new_F_norm_inv = norm(C_opt)
new_op_norm = opnorm(A_opt)
new_op_norm_inv = opnorm(C_opt)
k_opt = new_F_norm * new_F_norm_inv
k_op_opt = new_op_norm * new_op_norm_inv

println("The new Frobenius norm is $new_F_norm")
println("The new operator norm is $new_op_norm")
println("The new Frobenius condition number is $k_opt")
println("The new condition number is $k_op_opt")
println("The improvement of the Frobenius condition number is $(k/k_opt)")
println("The improvement of the condition number is $(k_op/k_op_opt)")

