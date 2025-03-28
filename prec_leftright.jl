using LinearAlgebra, Manifolds, Manopt, BlockDiagonals, RecursiveArrayTools;


function k_F(A)
    return norm(A) * norm(pinv(A))
end



include("data/housing_scale_f.jl")
println(size(A))

n = size(A)[1]
m = size(A)[2]
C = pinv(A)

kF = k_F(A)

println("The Frobenius condition number is $kF")


function f(M,p)
    p1 = p.x[1]
    p2 = p.x[2]
    Dg1 = Diagonal(p1)
    return log(norm(Dg1*A*inv(p2))) + log(norm(p2*C*inv(Dg1)))
end;

function grad_f(M,p)
    p1 = p.x[1]
    p2 = p.x[2]
    p2inv = inv(p2)
    Dg1 = Diagonal(p1)
    p1inv = inv(Dg1)
    X = Dg1 * A * p2inv * p2inv' * A' * Dg1' / ( norm(Dg1 * A * p2inv)^2 )
    Y = p1inv' * C' * p2' * p2 * C * p1inv / ( norm(p2 * C * p1inv)^2 )
    Z = p2 * C * p1inv * p1inv' * C' * p2' / ( norm(p2 * C * p1inv)^2 )
    W = p2inv' * A' * Dg1' * Dg1 * A * p2inv / ( norm(Dg1 * A * p2inv)^2 )
    return ArrayPartition( vec(diag(X)) - vec(diag(Y)) , Z - W )
end;

M = ProductManifold( PositiveVectors(n) , SymmetricPositiveDefinite(m) )

U1 = vec(ones(n,1))
U2 = Matrix{Float64}(I,m,m)
U = ArrayPartition(U1, U2)


opt1 = gradient_descent(M, f, grad_f, U;
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.20f | "), "\n", :Stop, 20],
    stopping_criterion = StopWhenGradientNormLess(1e-6) | StopAfterIteration(200)
)


A_opt = Diagonal(opt1.x[1]) * A * inv(opt1.x[2])

kF_opt = k_F(A_opt)

println("The new Frobenius condition number is $kF_opt")
println("The improvement is $(kF/kF_opt)")

