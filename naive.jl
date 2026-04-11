using Printf
using LinearAlgebra
"""
matmul_naive!(c, a, b)
Compute the product of matrices ‘a‘ and ‘b‘ and store in ‘c‘.
"""
function matmul_naive!(C, A, B)
    (N, M) = size(A)
    (P, R) = size(B)
    (n, r) = size(C)
    @assert M == P # check matrix size
    @assert n == N
    @assert r == R
    for i = 1:N
        for j = 1:R
            for k = 1:M
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
end

N = 100
M = 200
L = 50
C = zeros(N, L)
A = rand(N, M)
B = rand(M, L)
#= Do some timing tests
between Julia native
and naive matrix multiply.
=#
println()
println("Julia’s Native A * B")
@time C = A * B
@time C = A * B
println("---------------")
println()
println("Using Naive! Function!")
D = zeros(N, L)
matmul_naive!(D, A, B) #call once to compile
@time matmul_naive!(D, A, B)
@time matmul_naive!(D, A, B)
C += (A * B)
C += (A * B)
@printf "norm(C - D) = \x1b[31m %e \x1b[0m\n" norm(C - D)
println("---------------")
println()