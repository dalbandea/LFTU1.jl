module LFTU1

using LFTSampling

using KernelAbstractions
import Random
import BDIO
import LinearAlgebra
import Elliptic, Elliptic.Jacobi
# import CUDA, CUDAKernels

abstract type U1 <: AbstractLFT end
abstract type U1Quenched <: U1 end
abstract type U1Nf2 <: U1 end
abstract type U1Nf <: U1 end
export U1Quenched, U1Nf2, U1Nf

abstract type U1Parm <: LFTParm end

abstract type AbstractBoundaryCondition end
abstract type PeriodicBC <: AbstractBoundaryCondition end
abstract type OpenBC <: AbstractBoundaryCondition end
export PeriodicBC, OpenBC

Base.@kwdef struct U1QuenchedParm{B <: AbstractBoundaryCondition} <: U1Parm
    iL::Tuple{Int64,Int64}
    beta::Float64
    BC::Type{B}
end
export U1QuenchedParm

Base.@kwdef struct U1Nf2Parm{B <: AbstractBoundaryCondition} <: U1Parm
    iL::Tuple{Int64,Int64}
    beta::Float64
    am0::Float64
    BC::Type{B}
end
export U1Nf2Parm

Base.@kwdef struct U1NfParm{B <: AbstractBoundaryCondition} <: U1Parm
    iL::Tuple{Int64,Int64}
    beta::Float64
    am0::Array{Float64}
    BC::Type{B}
end
export U1NfParm

struct KernelParm
    threads::Tuple{Int64,Int64}
    blocks::Tuple{Int64,Int64}
end

KernelParm(lp::U1Parm) = KernelParm((lp.iL[1], 1), (1, lp.iL[1]))
export KernelParm

struct RHMCParm
	r_b::Float64
	n::Int64
	eps::Float64
	A::Float64
	rho::Vector{Float64}
	mu::Vector{Float64}
    nu::Vector{Float64}
    delta::Float64
    reweighting_N::Int64
    reweighting_Taylor::Int64
end
export RHMCParm


include("U1fields.jl")
export U1quenchedworkspace, U1Nf2workspace, U1Nfworkspace, coldstart!, randomize!

include("U1action.jl")
export action, U1plaquette!, U1action, gauge_action, top_charge

include("U1hmc.jl")
export Hamiltonian, generate_momenta!, update_fields!, U1_update_field!, update_momenta!, generate_pseudofermions!

include("U1rhmc.jl")

include("U1io.jl")

include("U1dirac.jl")
export gamm5Dw!, gamm5Dw_sqr_msq!

# to_device(::CUDAKernels.CUDADevice, x) = CUDA.CuArray(x)
# to_device(::ROCKernels.ROCDevice, x) = AMDGPU.ROCArray(x)
to_device(::KernelAbstractions.CPU, x) = x

allowscalar(::KernelAbstractions.CPU) = nothing
disallowscalar(::KernelAbstractions.CPU) = nothing
# allowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(true)
# disallowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(false)
# allowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(true)
# disallowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(false)

if !isdefined(Base, :get_extension)
  include("../ext/LFTU1CUDAExt.jl")
end

# Glossary of variable name meanings

# ws = workspace
# lp = lattice parameter
# frc = force

end # module LFTU1
