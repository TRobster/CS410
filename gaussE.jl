
function I(n)
    id = []
    for i in 1:n
        id[i] = i
    end 
    return i 
end

function forward(mat, n)
    
    L = zeros(n, n)
    P = I(n)

    for k in 1:n-1 
        i_max = k 
        v_max = abs(mat[i_max, k])
        for i in k+1:n
            if (abs(mat[i, k]) > v_max)
                v_max = mat[i, k]
                i_max = i
            end
        end


        if i_max != k
            mat[k], mat[i_max] = mat[i_max], mat[k]
        end

        for i in k+1:n
            m = mat[i, k] / mat[k, k]
            for j in k+1:n
                mat[i, j] = mat[i, j] - mat[k, j] * m
                end
            mat[i, k] = 0
        end
    end     
end 



