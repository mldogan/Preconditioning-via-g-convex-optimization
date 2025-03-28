using LinearAlgebra;
using Manifolds;
using ManifoldsBase;
using Manopt;
using RecursiveArrayTools;
## Set up product manifold
S2 = Sphere(2);
Sym3 = SymmetricMatrices(3);
M = S2 × Sym3;

## Pick an initial point on this product manifold
p0 = ArrayPartition(normalize([.25, .25, 0]), Matrix(1.0 * I(3)));
println(is_point(M,p0))

## Set up objective and gradient

g = [1.0; 0.0; 0.0];
I3 = Matrix(1.0 * I(3));


# Set up objective:  F(x, A) = g⋅x + tr(A)
L(p::ProductRepr) = dot(p.parts[1], g) + tr(p.parts[2]);

# Set up gradient:  gradL = proj_p dL / dp
function gradL(M::ProductManifold, p::ProductRepr)

    return project(M, p, ProductRepr(g, I3));
end
## Set up and solve optimization problem
prob = GradientProblem(M, L, gradL);
opts = GradientDescentOptions(p0);

result = solve(prob, opts);
