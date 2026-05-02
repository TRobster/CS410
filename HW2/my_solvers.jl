#=
Author: Trevor Robbins
Class: CS410
Instructor: McLaughlin
Project: Assignment 1
=# 

using LinearAlgebra
function eye(n)
    #=
    Basic identity matrix generation function, 
        simply allocates, zeros L and U, then iterates
        over the (i, i) diagonal for 1 placement
    =#
    I = Matrix{Float64}(undef, n, n)
    I .= 0
    for i in 1:n
        I[i, i] = 1.0
    end
    return I
end

function computeLUP(mat, n)
    #=
    Params: mat <-- an n x n symmetric positive definite (SPD) matrix
            n <-- The base linear-dimension, when squared gives us our mat-dimension
    Use: Given an n x n SPD-matrix, this function splits a matrix using gaussian elimination into
        three different factors, that being its lower quadrant matrix (L), its upper quadrant matrix (U),
        and the documentation of row swaps in the partial pivoting steps in a vector (P).
    
    =#
    
    L = eye(n)
    # In theory P is an identity matrix, however for memory conservation its a vector with entries representing "1" on the (i, i)-diag
    P = collect(1:n)

    # Main iteration for elimination, however first we must find the maximum absolute column value then swap for that row (partial pivoting)
    for k in 1:n-1 
        i_max = k 
        v_max = abs(mat[i_max, k])
        for i in k+1:n
            # Partial pivoting
            if (abs(mat[i, k]) > v_max)
                v_max = abs(mat[i, k])
                i_max = i
            end
        end

        # Records partial pivotting swap. 
        if i_max != k
            swap(mat, k, i_max, n)
            P[k], P[i_max] = P[i_max], P[k] 
        end

        # Main elimination step. Computes numerically stable multiplier given the row below our current row. 
        for i in k+1:n
            m = mat[i, k] / mat[k, k]
            L[i, k] = m
            for j in k+1:n
                mat[i, j] = mat[i, j] - mat[k, j] * m
                end
            mat[i, k] = 0
        end
    end
    return L, mat, P
end 


function swap(mat, i, j, n)
    # Simple swapper help function 
    for k in 1:n
        temp = mat[i, k]
        mat[i, k] = mat[j, k]
        mat[j, k] = temp
    end
end 

function forward(l, b, n)
    #=  
    Params: In basic format, our parameters correspond to the form Ly = Pb where
        L = lower triangular portion of orignal matrix
        b = Our original solution vector that integrates solving with use of immediate vector y
        n = Same as before, dimension
    =#
    y = zeros(n)
    for i in 1:n
        s = 0
        # running sum 
        for j in 1:i-1
            s = s + l[i, j] * y[j]
        end
        y[i] = b[i] - s
    end
    return y
end 

function backward(u, y, n)
    #=
    Params: In basic format, our parameters correspond to the form Ux = y
        U = Upper triangular portion of original matrix
        y = Solved porition-vector of L, now containing all solutions that can be substituted backwards down from n
        n = Same as before, dimension
    =#
    x = zeros(n)
    # If y's solution began as the literal first entry, x's solution lies at the bottom-most entry, meaning a countdown loop is needed
    for i in n:-1:1
        s = 0 
        # same logic as before, running sum
        for j in i+1:n
            s = s + u[i, j] * x[j]
        end 
        x[i] = (y[i] - s) / u[i, i]
    end
    return x 
end
    

function LUP_solve(A, b, n)
    #=
    Params: A = Solvable SPD matrix with N x N entries 
            b = Solution vector that must satisfy Ax = b where we solve for x 
            n = Linear dimension of A
    =#
    l, u, p = computeLUP(copy(A), n)
    y = forward(l, b[p], n)
    x = backward(u, y, n)
    return x
end

function conjugateGrad(A, b, n)
    x = zeros(n)
    r = copy(b)
    p = copy(r)
    rdotR = dot(r, r)
    α = (dot(r, r))/ dot(p, A*p)
    ε = 1e-8

    for k in 1:n-1
        Ap = A*p
        α = rdotR / dot(p, Ap)
        x = x + α * p
        r = r - α*Ap ## Put this AFTER x1 gets computed (i.e this will give you the same result as base-case!)

        # Option 2: dot product, comparing squared values (faster)
        rdotr_n = dot(r, r)        
        if rdotr_n < ε^2
            return x
        end
        α = rdotr_n/ dot(p, Ap)
        β = rdotr_n / rdotR
        p = r + β * p 
        rdotR = rdotR_n
        
    end
    @warn "CG did not converge in $n iterations"
    return x
end 

### testing below 
#=
for N in [10, 100, 1000]
    local B = rand(N, N)
    local A = B' * B + eye(N)
    local b = rand(N)
    println("N = $N")
    @time LUPsolve(A, b, N)
end
=#
