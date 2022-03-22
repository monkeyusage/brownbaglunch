
f(a, b) = a + b

@code_lowered f(1, 1)
@code_typed f(1, 1)
@code_llvm f(1, 1)
@code_native f(1, 1)

@code_native f(1.0, 3.0)
@code_native f(1.2, 4)

using BenchmarkTools, LinearAlgebra

arr = rand(Float32, 1_000, 10)
arr = rand(Float32, 100_000, 10)

function f(a)
    b = a * a'
    b[diagind(b)] .= 0
    sum(b, dims=2)
end

f(arr)
@benchmark f(arr)

function g(a::Array{Float32, 2})::Array{Float32}
    """
    # vectorized version of the following operations with M (n, m) => NM (n, n) => n
    out = a * a' => creates a a we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = size(a)

    out = Array{Float32, 1}(undef, N)

    Threads.@threads for i in 1:N
        total = zero(Float32)
        @inbounds for ii in 1:N
            if (i == ii) continue end
            @inbounds @simd for j in 1:M
                total += a[i, j] * a[ii, j]
            end
        end
        @inbounds out[i] = total
    end
    return out
end

g(arr)
@benchmark g(arr)


using CUDA
function g_kernel(matrix::CuDeviceMatrix{Float32}, out::CuDeviceVector{Float32}, len::Int, N::Int, M::Int)::Nothing
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (index) < len
        for i in 1:N
            if index == i continue end
            for j in 1:M
                @inbounds out[index] += matrix[index, j] * matrix[i, j]
            end
        end
    end
    return nothing
end

function g(matrix::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer})::Array{Float32}
    len = length(matrix)
    N, M = size(matrix)
    threads_per_block = 256
    blocks = Int(ceil(N / threads_per_block))

    out = CUDA.zeros(Float32, N)

    @cuda threads=threads_per_block blocks=blocks g_kernel(matrix, out, len, N, M)
    return Array(out)
end

arr = CUDA.rand(Float32, 100_000, 10)
g(arr)

@CUDA.time g(arr)

