#uncomment below if the code does not compile. Comment them again after it compiles.
#using Pkg
#Pkg.add("LinearAlgebra")
#Pkg.add("Manifolds")
#Pkg.add("Manopt")
#Pkg.add("BlockDiagonals")
using LinearAlgebra, Manifolds, Manopt, BlockDiagonals

function k_F(A)
    return norm(A) * norm(pinv(A))
end


include("data/mpg_scale_f.jl") # defines A to be the matrix given in the directory.

println(size(A)) # dimensions of A


global block_size = 5 # we precondition from the left with block diagonal matrices whose blocks are of size block_size
n = size(A)[1]
rem = n % block_size # we remove rows of A until its size is divisible by block_size

if size(A)[1] == size(A)[2]
    A = A[1:end-rem, 1:end-rem]
else
    A = A[1:end-rem, 1:end]
end

println(size(A)) # new dimensions of A (some rows are removed)


B = A * A'
C = pinv(A)
D = C' * C

F_norm = norm(A)
F_norm_inv = norm(C)
k = F_norm * F_norm_inv
op_norm = opnorm(A)
op_norm_inv = opnorm(C)
k_op = op_norm * op_norm_inv
svd_min = minimum(svdvals(A))

println("The Frobenius norm of the original matrix is $F_norm")
println("The operator norm of the original matrix is $op_norm")
println("The Frobenius condition number of the original matrix is $k")
println("The condition number of the original matrix is $k_op")
println("The minimum singular value of the original matrix is $svd_min")


# the below is the function we want to optimize: log( kappa_F ( D * A ) ) over block diagonal D
function f(M,p) 
    Dg = BlockDiagonal(p)
    return log(norm(Dg*A)) + log(norm(C*inv(Dg)))
end;

function k_block(X,s)
    bl = zeros((block_size,block_size))
    for i in 1:block_size
        for j in 1:block_size
            bl[i,j] = X[block_size*(s-1)+i,block_size*(s-1)+j]
        end
    end
    return bl
end;

function block_diag(X)
    sz = size(X)[1]
    m = div(n,block_size)
    return [ k_block(X,i) for i in 1:m]
end;

function grad_f(M,p)
    Dg = BlockDiagonal(p)
    X = Dg * B * Dg' / (norm(Dg*A)^2)
    Y = inv(Dg)' * D * inv(Dg) / ( norm(C*inv(Dg))^2 )
    return block_diag(X - Y)
end;


m = div(n,block_size)
println("Block size is $block_size")
M = PowerManifold(SymmetricPositiveDefinite(block_size) , NestedPowerRepresentation() , m);


Ik = Matrix{Float64}(I,block_size,block_size)
U = [Ik for i in 1:m]

# the below is the main part: optimizes f over the manifold M using trust regions method. 
opt1 = trust_regions(M, f, grad_f, U;
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
    (:Cost, " F(x): %1.20f | "), "\n", :Stop, 20],
    stopping_criterion = StopWhenGradientNormLess(1e-6) | StopAfterIteration(200)
)


A_opt = BlockDiagonal(opt1) * A 
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


