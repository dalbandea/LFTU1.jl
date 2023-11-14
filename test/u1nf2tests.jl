using LFTSampling
using LFTU1
using LinearAlgebra


import LFTSampling: analytic_force, infinitesimal_transformation, get_field
function analytic_force(u1ws::LFTU1.U1Nf2, hmcws::LFTU1.U1Nf2HMC)
    LFTU1.force!(u1ws, hmcws)
    frc = hmcws.frc1 .+ hmcws.frc2 .+ hmcws.pfrc
    return frc
end

function infinitesimal_transformation(field_elem, epsilon, lftws2::LFTU1.U1)
    return field_elem * exp(im*epsilon)
end

get_field(u1ws::LFTU1.U1Nf2) = u1ws.U

# U1 Theory parameters
lsize = 8
beta = 5.0
mass = 0.6

model = LFTU1.U1Nf2(Float64, iL = (lsize, lsize), beta = beta, am0 = mass, BC = PeriodicBC)

alg = HMC(
          integrator = Leapfrog(1.0, 10),
         )

hmcws = LFTU1.sampler(model, alg)

LFTU1.randomize!(model)
LFTU1.generate_pseudofermions!(model, hmcws)

@testset "PBC Pseudofermion generation" begin
    xi = similar(hmcws.X)
    invert!(xi, gamm5Dw_sqr_msq!, hmcws.F, model.sws, model)
    @test isapprox(dot(xi, hmcws.F), dot(hmcws.X, hmcws.X))
end

@testset "PBC HMC reversibility" begin
    model_bckp = deepcopy(model)
    LFTSampling.reversibility!(model, hmcws)
    ΔU = model.U .- model_bckp.U
    @test isapprox(zero(model.PRC), mapreduce(x -> abs2(x), +, ΔU), atol = 1e-15)
end

@testset verbose = true "PBC HMC force" begin
    ΔF = LFTSampling.force_test(model, hmcws, 1e-5)
    @test isapprox(zero(model.PRC), ΔF, atol = 1e-5)
end


model = LFTU1.U1Nf2(Float64, iL = (lsize, lsize), beta = beta, am0 = mass, BC = OpenBC)

alg = HMC(
          integrator = Leapfrog(1.0, 10),
         )

hmcws = LFTU1.sampler(model, alg)

LFTU1.randomize!(model)
LFTU1.generate_pseudofermions!(model, hmcws)

@testset "OBC Pseudofermion generation" begin
    xi = similar(hmcws.X)
    invert!(xi, gamm5Dw_sqr_msq!, hmcws.F, model.sws, model)
    @test isapprox(dot(xi, hmcws.F), dot(hmcws.X, hmcws.X))
end

@testset "OBC HMC reversibility" begin
    model_bckp = deepcopy(model)
    LFTSampling.reversibility!(model, hmcws)
    ΔU = model.U .- model_bckp.U
    @test isapprox(zero(model.PRC), mapreduce(x -> abs2(x), +, ΔU), atol = 1e-15)
end

@testset verbose = true "OBC HMC force" begin
    ΔF = LFTSampling.force_test(model, hmcws, 1e-5)
    @test isapprox(zero(model.PRC), ΔF, atol = 1e-5)
end
