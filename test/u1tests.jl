using LinearAlgebra

let
    # U1 Theory parameters
    lsize1 = 8
    lsize2 = 10
    beta = 5.0
    mass = 0.6

    model = LFTU1.U1Nf2(Float64, iL = (lsize1, lsize2), beta = beta, am0 = mass, BC = PeriodicBC)

    LFTU1.randomize!(model)

    @testset "Random gauge transformation" begin
        S1 = gauge_action(model)
        random_gauge_trafo(model)
        S2 = gauge_action(model)
        @test isapprox(zero(model.PRC), mapreduce(x -> abs2(x), +, S2-S1), atol = 1e-15)
    end



    model.U .= 1

    LW = 4

    winding(LW, model, sign=-1)

    Q1 = top_charge(model)
    Sw = gauge_action(model)
    DSw_theo = -4*LW*beta*(cos(pi/2/LW)-1)

    @testset "Winding transformation" begin
        @test isapprox(1.0, Q1, atol = 1e-15)
        @test isapprox(Sw, DSw_theo, atol = 1e-8)
    end

end
