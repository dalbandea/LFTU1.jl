
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
                 maxiter = 100000,
                 tol = 1e-26
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


    #####################################
    #  Statistical fluctuations of Z^p  #
    #####################################

    # Lattice and Zolotarev parameters
    ns_rat = [10]          # number of Zolotarev monomial pairs
    # Generate Zolotarev parameters
    rprm = LFTU1.get_rhmc_params(ns_rat, r_as, r_bs)
    model.rprm .= rprm

    # Check (X, ZᵖX) ≤ 2N(2δ)ᵖ for p = 1,...,10. Luscher eq. (4.5), (4.6)
    tmp .= samplerws.X
    ZpX  = similar(samplerws.X)

    @testset "Zolotarev Z^p statistical fluctuations" begin
        for i in 1:10
            LFTU1.LuscherZ(ZpX, tmp, model.params.am0[1], model.rprm[1], model)
            Ztrace = dot(tmp, ZpX) |> real
            @test abs(Ztrace) < 2 * model.params.iL[1]^2 * (2 * model.rprm[1].delta)^i
            tmp .= ZpX
        end
    end

    #####################################
    #  Statistical fluctuations of W_N  #
    #####################################

    V = model.params.iL[1] * model.params.iL[2]
    W_1 = 1.0 # reweighting factor W_N with N=1

    @testset "Zolotarev W_N statistical fluctuations" begin
        for i in 1:10
            # Lattice and Zolotarev parameters
            ns_rat = [i]          # number of Zolotarev monomial pairs
            # Generate Zolotarev parameters
            rprm = LFTU1.get_rhmc_params(ns_rat, r_as, r_bs)
            W_1 = LFTU1.reweighting_factor(model.params.am0[1], rprm[1], model)
            @test exp(-2 * V * rprm[1].delta) < W_1 < exp(2 * V * rprm[1].delta)
        end
    end
end
