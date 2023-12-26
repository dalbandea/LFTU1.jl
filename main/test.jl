# U1 Nf2
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using BDIO

ENV["JULIA_DEBUG"] = "all"

beta = 5.555
lsize = 14
mass = 0.2

model = U1Nf2(Float64, beta = beta, iL = (lsize, lsize), am0 = mass, BC = OpenBC)

model = U1Quenched(Float64, beta = beta, iL = (lsize, lsize), BC = OpenBC)

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 15)))

LFTU1.randomize!(model)

@time sample!(model, samplerws)

Ss = Vector{Float64}(undef, 100000)

for i in 1:100000
    @time sample!(model, samplerws)
    Ss[i] = gauge_action(model)
end

using ADerrors

id = "test2"
uws = uwreal(Ss, id)
uwp = 1 - uws/(beta*(lsize-1)^2)
uwerr(uwp)
uwp




using LinearAlgebra
function compute_Dwsr!(D, x1, x2, xtmp, xmodel)
    D .= 0.0
    x1 .= 0.0
    x2 .= 0.0
    xtmp .= 0.0
    for i in eachindex(x1)
        x1[i] = 1.0
        gamm5Dw_sqr_msq!(x2, xtmp, x1, xmodel)
        xtmp .= 0.0
        for j in eachindex(xtmp)
            xtmp[j] = 1.0
            D[i,j] = dot(x2, xtmp)
            xtmp[j] = 0.0
        end
        x1[i] = 0.0
    end
end

x1 = similar(copy(model.U))
x2 = similar(x1)
xtmp = similar(x1)
D = similar(x1, prod(size(x1)), prod(size(x1)))

compute_Dwsr!(D, x1, x2, xtmp, model)

det(D)
