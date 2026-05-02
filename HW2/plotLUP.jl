#=
Author: Trevor Robbins
Class: CS410
Instructor: McLaughlin
Project: Assignment 1 — Dense vs Sparse LU vs CG
=#

using LinearAlgebra
using SparseArrays
using Plots
using Printf

include("my_solvers.jl")  # provides conjugateGrad

# Setup
const Ns = [10, 100, 1000]

# Storage for each method's timings across all N
t_dense_lu    = Float64[]
t_dense_solve = Float64[]
t_dense_cg    = Float64[]

t_sparse_lu    = Float64[]
t_sparse_solve = Float64[]
t_sparse_cg    = Float64[]

mem_dense  = Int[]
mem_sparse = Int[]

# Helper: build the SPD tridiagonal Poisson matrix in dense form
function build_dense(N)
    A = zeros(N, N)
    for i in 1:N
        A[i, i] = 2.0
        if i > 1
            A[i, i-1] = -1.0
            A[i-1, i] = -1.0
        end
    end
    return A
end

# Helper: build the same matrix directly in sparse form (no dense intermediate)
function build_sparse(N)
    I_idx = Int[]; J_idx = Int[]; V = Float64[]
    for i in 1:N
        push!(I_idx, i); push!(J_idx, i);   push!(V,  2.0)
        if i > 1
            push!(I_idx, i);   push!(J_idx, i-1); push!(V, -1.0)
            push!(I_idx, i-1); push!(J_idx, i);   push!(V, -1.0)
        end
    end
    return sparse(I_idx, J_idx, V, N, N)
end

# Helper: total bytes for a sparse matrix (struct + underlying arrays)
sparse_bytes(A) = sizeof(A) + sizeof(A.colptr) + sizeof(A.rowval) + sizeof(A.nzval)

# Timing
for N in Ns
    @printf "\n=== N = %d ===\n" N
    b = ones(N)

    # --- Dense ---
    Ad = build_dense(N)

    # warm-up
    lu(Ad); (lu(Ad) \ b); conjugateGrad(Ad, b, N)

    println("Dense:")
    @time Fd = lu(Ad)
    tDL = @elapsed lu(Ad)

    @time Fd \ b
    tDS = @elapsed Fd \ b

    @time conjugateGrad(Ad, b, N)
    tDC = @elapsed conjugateGrad(Ad, b, N)

    push!(t_dense_lu,    tDL)
    push!(t_dense_solve, tDS)
    push!(t_dense_cg,    tDC)
    push!(mem_dense,     sizeof(Ad))

    # --- Sparse ---
    As = build_sparse(N)

    # warm-up
    lu(As); (lu(As) \ b); conjugateGrad(As, b, N)

    println("Sparse:")
    @time Fs = lu(As)
    tSL = @elapsed lu(As)

    @time Fs \ b
    tSS = @elapsed Fs \ b

    @time conjugateGrad(As, b, N)
    tSC = @elapsed conjugateGrad(As, b, N)

    push!(t_sparse_lu,    tSL)
    push!(t_sparse_solve, tSS)
    push!(t_sparse_cg,    tSC)
    push!(mem_sparse,     sparse_bytes(As))

    @printf "  Memory: dense = %d bytes, sparse = %d bytes (%.1fx reduction)\n" sizeof(Ad) sparse_bytes(As) (sizeof(Ad) / sparse_bytes(As))
end

# Empirical slopes
println("\nEmpirical log-log slopes:")
for i in 1:length(Ns)-1
    Δlog = log10(Ns[i+1]) - log10(Ns[i])
    s_dl = (log10(t_dense_lu[i+1])    - log10(t_dense_lu[i]))    / Δlog
    s_ds = (log10(t_dense_solve[i+1]) - log10(t_dense_solve[i])) / Δlog
    s_dc = (log10(t_dense_cg[i+1])    - log10(t_dense_cg[i]))    / Δlog
    s_sl = (log10(t_sparse_lu[i+1])    - log10(t_sparse_lu[i]))    / Δlog
    s_ss = (log10(t_sparse_solve[i+1]) - log10(t_sparse_solve[i])) / Δlog
    s_sc = (log10(t_sparse_cg[i+1])    - log10(t_sparse_cg[i]))    / Δlog
    @printf "  N=%d → N=%d\n" Ns[i] Ns[i+1]
    @printf "    Dense  │ LU: %.2f │ solve: %.2f │ CG: %.2f\n" s_dl s_ds s_dc
    @printf "    Sparse │ LU: %.2f │ solve: %.2f │ CG: %.2f\n" s_sl s_ss s_sc
end

# O(N^3) reference, anchored to dense LU at smallest N
t_cubic = [t_dense_lu[1] * (N / Ns[1])^3 for N in Ns]

# Plotting
p = plot(
    xscale    = :log10,
    yscale    = :log10,
    xlabel    = "Matrix size N",
    ylabel    = "Time (seconds)",
    title     = "Dense vs Sparse: LU and CG Runtime vs N  (log-log)",
    legend    = :topleft,
    frame     = :box,
    minorgrid = true,
    size      = (800, 540),
    dpi       = 150,
    margin    = 5Plots.mm
)

plot!(p, Ns, t_cubic,
    label = "O(N^3) reference", lw = 2, ls = :dash, color = :gray)

# Dense — warm tones
plot!(p, Ns, t_dense_lu,
    label = "Dense LU",         lw = 2, color = :coral,
    markershape = :circle,  markersize = 6)
plot!(p, Ns, t_dense_solve,
    label = "Dense solve",      lw = 2, color = :darkorange,
    markershape = :square,  markersize = 6)
plot!(p, Ns, t_dense_cg,
    label = "Dense CG",         lw = 2, color = :firebrick,
    markershape = :diamond, markersize = 6)

# Sparse — cool tones
plot!(p, Ns, t_sparse_lu,
    label = "Sparse LU",        lw = 2, color = :steelblue,
    markershape = :circle,  markersize = 6, ls = :dot)
plot!(p, Ns, t_sparse_solve,
    label = "Sparse solve",     lw = 2, color = :teal,
    markershape = :square,  markersize = 6, ls = :dot)
plot!(p, Ns, t_sparse_cg,
    label = "Sparse CG",        lw = 2, color = :darkorchid,
    markershape = :diamond, markersize = 6, ls = :dot)

xticks!(p, Ns, string.(Ns))

savefig(p, "dense_vs_sparse_timing.png")
println("\nFigure saved → dense_vs_sparse_timing.png")
display(p)