# The generic HMC tests of LFTSampling, run on the U(1) models. LFTSampling knows
# nothing about LFTU1: the only LFTU1-specific code is field_distance below (which
# would live in src/), everything else uses the methods LFTU1 defines for sampling.
using Test
using LFTSampling
using LFTU1

LFTSampling.field_distance(a::LFTU1.U1, b::LFTU1.U1) = sum(abs2, a.U .- b.U)

@testset verbose = true "LFTSampling test suite" begin
    for BC in (PeriodicBC, OpenBC)
        models = [
            U1Quenched(Float64, beta = 5.0, iL = (8, 8), BC = BC),
            U1Nf2(Float64, beta = 5.0, iL = (8, 8), am0 = 0.2, BC = BC),
            U1Nf(Float64, beta = 5.0, iL = (8, 8), am0 = [0.6, 0.3], BC = BC,
                 ns_rat = [3, 5], r_as = [sqrt(0.45), sqrt(0.45)],
                 r_bs = [sqrt(22), sqrt(22)]),
        ]
        for model in models
            LFTU1.randomize!(model)
            LFTSampling.test_hmc(model, name = "$(nameof(typeof(model))) $BC")
        end
    end
end
