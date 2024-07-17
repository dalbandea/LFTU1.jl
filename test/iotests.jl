using LFTU1

let

    beta = 5.555
    lsize = 8

    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = OpenBC,
                      )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Quenched)
    @testset "Quenched OBC" begin
        for i in 1:length(ens)
            @test ens[i].params == rens[i].params
            @test ens[i].U == rens[i].U
        end
    end
    rm(fname, force=true)



    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = PeriodicBC,
                      )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Quenched)
    @testset "Quenched PBC" begin
        for i in 1:length(ens)
            @test ens[i].params == rens[i].params
            @test ens[i].U == rens[i].U
        end
    end
    rm(fname, force=true)


    beta = 5.555
    l1 = 8
    l2 = 10
    mass = 0.6
    model = U1Nf2(Float64,
                  beta = beta,
                  am0 = mass,
                  iL = (l1, l2),
                  BC = PeriodicBC,
                 )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Nf2)
    roboens = deepcopy(rens)
    LFTU1.coldstart!.(roboens)
    fb, model = read_cnfg_info(fname, U1Nf2)
    for i in 1:length(roboens)
        read_next_cnfg(fb, model)
        roboens[i] = deepcopy(model)
    end
    close(fb)
    @testset "Nf=2 PBC" begin
        for i in 1:length(ens)
            @test ens[i].params == rens[i].params
            @test ens[i].U == rens[i].U
            @test ens[i].params == roboens[i].params
            @test ens[i].U == roboens[i].U
        end
    end
    rm(fname, force=true)



    beta = 5.0
    lsize = 8
    masses = [0.2, 0.2]
    model = U1Nf(Float64,
                  beta = beta,
                  am0 = masses,
                  iL = (lsize, lsize),
                  BC = PeriodicBC,
                 )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Nf)
    roboens = deepcopy(rens)
    LFTU1.coldstart!.(roboens)
    fb, model = read_cnfg_info(fname, U1Nf)
    for i in 1:length(roboens)
        read_next_cnfg(fb, model)
        roboens[i] = deepcopy(model)
    end
    close(fb)
    @testset "Nf PBC" begin
        for i in 1:length(ens)
            @test ens[i].params.am0 == rens[i].params.am0
            @test ens[i].U == rens[i].U
            @test ens[i].params.beta == roboens[i].params.beta
            @test ens[i].U == roboens[i].U
        end
    end
    # Test to read the last configuration
    ncfgs = LFTSampling.count_configs(fname)
    fb, model2 = read_cnfg_info(fname, U1Nf)
    LFTSampling.read_cnfg_n(fb, ncfgs, model2)
    close(fb)
    @testset "Nf PBC read last config" begin
        @test ens[end].params.am0 == model2.params.am0
        @test ens[end].U == model2.U
    end

    rm(fname, force=true)

end
