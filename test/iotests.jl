using LFTSampling
using LFTU1

beta = 5.555
lsize = 8
mass = 0.6

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
isfile(fname) && error("File already exists!")
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Quenched)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Quenched OBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end

ens = []
samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))

for i in 1:2
    sample!(model, samplerws)
    push!(ens, model)
end

isfile(fname) && error("File already exists!")
for conf in ens
    LFTU1.save_cnfg(fname, conf)
end

ens_file = LFTSampling.read_ensemble(fname, U1Quenched)
rm(fname, force=true)

@testset "Quenched OBC read ensemble" begin
    for i in eachindex(ens)
        @test ens[i].U == ens_file[i].U
    end
end


model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = PeriodicBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Quenched)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Quenched PBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end

model = U1Nf2(Float64,
                   beta = beta,
                   am0 = mass,
                   iL = (lsize, lsize),
                   BC = PeriodicBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Nf2)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Nf=2 PBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end


model = U1Nf2(Float64,
                   beta = beta,
                   am0 = mass,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Nf2)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Nf=2 OBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end
