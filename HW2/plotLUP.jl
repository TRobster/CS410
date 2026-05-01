
#=
Author: Trevor Robbins
Class: CS410
Instructor: McLaughlin
Project: Assignment 1
=# 

using Plots
using Printf

include("my_solvers.jl")

#Setup
const Ns = [10, 100, 1000]

t_lup   = Float64[]
t_solve = Float64[]

# Timing
# Quickly call functions to get rid of Julia's compilation time
# then @time as required by the assignment.
for N in Ns
    B = rand(N, N)
    A = B'B + Matrix(1.0*I, N, N)   # B^T B + I 
    b = rand(N)

    # Warm-up (compile)
    computeLUP(A, N)
    LUP_solve(A, b, N)

    @printf "\n   N = %d  \n" N

    println("computeLUP:")
    @time computeLUP(A, N)
    tL = @elapsed computeLUP(A, N)

    println("LUP_solve:")
    @time LUP_solve(A, b, N)
    tS = @elapsed LUP_solve(A, b, N)

    push!(t_lup,   tL)
    push!(t_solve, tS)
end

# O(N^3) reference line anchored to first LUP measurement 
t_cubic = [t_lup[1] * (N / Ns[1])^3 for N in Ns]

# Empirical slopes
println("\nEmpirical log-log slopes (expect ≈ 3.0 for O(N³)):")
for i in 1:length(Ns)-1
    s_lup   = (log10(t_lup[i+1])   - log10(t_lup[i]))   / (log10(Ns[i+1]) - log10(Ns[i]))
    s_solve = (log10(t_solve[i+1]) - log10(t_solve[i])) / (log10(Ns[i+1]) - log10(Ns[i]))
    @printf "  N=%d → N=%d │ computeLUP: %.2f │ LUP_solve: %.2f\n" Ns[i] Ns[i+1] s_lup s_solve
end

# Plotting
p = plot(
    xscale    = :log10,
    yscale    = :log10,
    xlabel    = "Matrix size N",
    ylabel    = "Time (seconds)",
    title     = "LUP Runtime vs. N  (log-log scale)",
    legend    = :topleft,
    frame     = :box,
    minorgrid = true,
    size      = (700, 480),
    dpi       = 150,
    margin    = 5Plots.mm
)

plot!(p, Ns, t_cubic,
    label       = "O(N^3) reference",
    lw          = 2,
    ls          = :dash,
    color       = :steelblue,
    markershape = :none
)

plot!(p, Ns, t_lup,
    label       = "computeLUP (factorization only)",
    lw          = 2,
    color       = :coral,
    markershape = :circle,
    markersize  = 7,
    markercolor = :coral
)

plot!(p, Ns, t_solve,
    label       = "LUP_solve (factorize + substitution)",
    lw          = 2,
    color       = :darkorchid,
    markershape = :diamond,
    markersize  = 7,
    markercolor = :darkorchid
)

# Annotate empirical slope on each segment of the LUP_solve curve
for i in 1:length(Ns)-1
    s     = (log10(t_solve[i+1]) - log10(t_solve[i])) / (log10(Ns[i+1]) - log10(Ns[i]))
    x_mid = sqrt(Ns[i] * Ns[i+1])
    y_mid = sqrt(t_solve[i] * t_solve[i+1])
    annotate!(p, x_mid, y_mid * 1.8,
        text("slope ≈ $(round(s, digits=2))", 8, :darkorchid, :center))
end

xticks!(p, Ns, string.(Ns))

savefig(p, "lup_timing.png")
println("\nFigure saved → lup_timing.png")
display(p)
