
using LinearAlgebra

let
    beta = 5.555
    lsize = 8

    ns_rat = [5]
    r_as = [sqrt(0.30)]
    r_bs = [sqrt(22)]
    masses = [0.2]
    model = U1Nf(Float64,
                 beta = beta,
                 iL = (lsize, lsize),
                 am0 = masses,
                 BC = PeriodicBC,
                 ns_rat = ns_rat,
                 r_as = r_as,
                 r_bs = r_bs,
                )
    LFTU1.randomize!(model)
    lambda_min, lambda_max = power_method(model, masses[1], iter=10000)
    model.rprm .= LFTU1.get_rhmc_params(ns_rat, [lambda_min*0.99 |> real |> sqrt], [lambda_max*1.01 |> real |> sqrt])
    samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))
    generate_pseudofermions!(model, samplerws)
    LFTU1.R(samplerws.F[1], samplerws.X, model.params.am0[1], model.rprm[1], model)
    tmp = copy(samplerws.F[1])
    LFTU1.R(samplerws.F[1], tmp, model.params.am0[1], model.rprm[1], model)
    X_f = similar(samplerws.X)
    gamm5Dw!(X_f, samplerws.F[1], model.params.am0[1], model::U1Nf)
    tmp2 = copy(X_f)
    gamm5Dw!(X_f, tmp2, model.params.am0[1], model::U1Nf)
    deviation = samplerws.X - X_f
    delta_rhmc = LFTU1.delta(model.rprm[1].n, model.rprm[1].eps) |> (x -> x*(2+x)) # maximum error
    delta_X = sqrt(dot(deviation, deviation))/sqrt(dot(samplerws.X,samplerws.X))
    # Se debe cumplir que delta_X ≤ delta_rhmc.

    @testset "Zolotarev bound" begin
        @test real(delta_X) < delta_rhmc
    end
end
