

function eye(n)
    I = zeros(n, n)
    for i in 1:n
        I[i, i] = 1.0
    end
    return I
end

function computeLUP(mat, n)
    
    L = eye(n)
    P = collect(1:n)

    for k in 1:n-1 
        i_max = k 
        v_max = abs(mat[i_max, k])
        for i in k+1:n
            if (abs(mat[i, k]) > v_max)
                v_max = abs(mat[i, k])
                i_max = i
            end
        end


        if i_max != k
            swap(mat, k, i_max, n)
            P[k], P[i_max] = P[i_max], P[k] 
        end

        for i in k+1:n
            m = mat[i, k] / mat[k, k]
            L[i, k] = m
            for j in k+1:n
                mat[i, j] = mat[i, j] - mat[k, j] * m
                end
            mat[i, k] = 0
        end
    end
    return mat, L, P
end 


function swap(mat, i, j, n)
    for k in 1:n
        temp = mat[i, k]
        mat[i, k] = mat[j, k]
        mat[j, k] = temp
    end
end 

function forward(l, b, n)
    y = zeros(n)
    for i in 1:n
        s = 0
        for j in 1:i-1
            s = s + l[i, j] * y[j]
        end
        y[i] = b[i] - s
    end
    return y
end 

function backward(u, y, n)
    x = zeros(n)
    for i in n:-1:1
        s = 0 
        for j in i+1:n
            s = s + u[i, j] * x[j]
        end 
        x[i] = (y[i] - s) / u[i, i]
    end
    return x 
end
    

function LUPsolve(A, b, n)
    l, u, p = computeLUP(A, n)
    y = forward(l, b[p], n)
    x = backward(u, y, n)
    return x
end


### testing below 

N = 10
I = eye(N)
B = rand(N, N)
A = B' * B + I
b = rand(N)
print("normal instance of A before solving ")
display(A)
print("normal instance of b-vector before solving ")
display(b)