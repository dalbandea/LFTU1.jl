module LFTU1CUDA

using LFTU1
import CUDA, CUDAKernels

to_device(::CUDAKernels.CUDADevice, x) = CUDA.CuArray(x)

allowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(true)
disallowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(false)

end

