


function forward(mat, n)
    
    L = zeros(n, n)
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
end 


function swap(mat, i, j, n)
    for k in 1:n
        temp = mat[i, k]
        mat[i, k] = mat[j, k]
        mat[j, k] = temp
    end
end 