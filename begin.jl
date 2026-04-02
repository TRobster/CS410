println("hello world")


function fibonacciSeq(x)
    ### Returns a list of all numbers in the fibonacci Sequence up to number x
    arr = Int[]
    push!(arr, 0, 1)
    for i in 2:x
        sum = arr[i-1] + arr[i]
        push!(arr, sum)
    end
    return arr
end 
n = 15

println(fibonacciSeq(n))