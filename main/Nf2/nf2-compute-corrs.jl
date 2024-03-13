# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using LinearAlgebra
using ProgressBars
using Dates

abstract type AbstractObservable end
abstract type AbstractCorrelator <: AbstractObservable end
abstract type AbstractU1Correlator <: AbstractCorrelator end

import Base: write, read

function write(obs::AbstractCorrelator)
    global io_stat = open(obs.filepath, "a")
    write(io_stat, "$(obs.result[1])")
    for i in 2:length(obs.result)
        write(io_stat, ",$(obs.result[i])")
    end
    write(io_stat, "\n")
    close(io_stat)
    return nothing
end

function save!(obs::AbstractCorrelator)
    push!(obs.history, obs.result)
    return nothing
end

function read(::Type{T}, filepath::String) where T <: AbstractCorrelator
    return DelimitedFiles.readdlm(filepath, ',', Float64)
end


struct U1PionCorrelator <: AbstractU1Correlator
    name::String
    ID::String
    filepath::String
    R1
    R2
    S
    S0
    result::Vector{Float64} # correlator
    history::Vector{Vector{Float64}}
    function U1PionCorrelator(u1ws::U1Nf2; wdir::String = "./trash/", 
                               name::String = "U(1) pion correlator with Nf=2", 
                               ID::String = "corr_pion", 
                               mesdir::String = "measurements/", 
                               extension::String = ".txt")
        dt = Dates.now()
        wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
        lp = u1ws.params
        filepath = joinpath(wdir, mesdir, ID*wdir_sufix*extension)
        R1 = LFTU1.to_device(u1ws.device, zeros(complex(Float64), lp.iL..., 2))
        R2 = copy(R1)
        S = copy(R1)
        S0 = copy(R1)
        C = zeros(Float64, lp.iL[1])
        history = []
        mkpath(dirname(filepath))
        return new(name, ID, filepath, R1, R2, S, S0, C, history)
    end
end
export U1PionCorrelator


function invert_sources!(corrws::AbstractU1Correlator, u1ws::U1Nf2)

    S0 = corrws.S0
    S = corrws.S
    R1 = corrws.R1
    R2 = corrws.R2
    lp = u1ws.params

    # Source 1
    S0 .= zero(eltype(S0))
    LFTU1.allowscalar(u1ws.device)
    S0[1,1,1] = one(eltype(S0))
    LFTU1.disallowscalar(u1ws.device)
    S = similar(S0)

    ## Solve g5D S = S0 for S
	iter = invert!(S, gamm5Dw_sqr_msq!, S0, u1ws.sws, u1ws)
    gamm5Dw!(R1, S, u1ws)

    # Source 2
    S0 .= zero(eltype(S0))
    LFTU1.allowscalar(u1ws.device)
    S0[1,1,2] = one(eltype(S0))
    LFTU1.disallowscalar(u1ws.device)

	iter = invert!(S, gamm5Dw_sqr_msq!, S0, u1ws.sws, u1ws)
    gamm5Dw!(R2, S, u1ws)

    # NOTE: R1[1,1,1]-R2[1,1,2] is the chiral condensate. Checking that it has
    # no imaginary part is a good test
    
    return nothing
end

function pion_correlator_function(corrws::AbstractU1Correlator, t, u1ws::U1Nf2)
    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)
    c = zero(ComplexF64)
    d = zero(ComplexF64)

    # NOTE: this should be ultraslow. It may be better to put R1 and R2 into the
    # CPU prior to calling this function. For GPU, the best one can do is to
    # reduce columns of the GPU array.
    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.R1[x,t,1]
        b = corrws.R1[x,t,2]
        c = corrws.R2[x,t,1]
        d = corrws.R2[x,t,2]

        Ct += abs(dot(a,a) + dot(b,b) + dot(c,c) + dot(d,d))
    end
    LFTU1.disallowscalar(u1ws.device)

    return Ct
end

function pion_correlator_function(corrws::AbstractU1Correlator, u1ws::U1Nf2)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = pion_correlator_function(corrws, t, u1ws) |> real
    end
end

function (corrws::U1PionCorrelator)(u1ws::U1Nf2)
    invert_sources!(corrws, u1ws)
    pion_correlator_function(corrws, u1ws)
    return nothing
end



struct U1PCACCorrelator <: AbstractU1Correlator
    name::String
    ID::String
    filepath::String
    R1
    R2
    S
    S0
    result::Vector{Float64} # correlator
    history::Vector{Vector{Float64}}
    function U1PCACCorrelator(u1ws::U1Nf2; wdir::String = "./trash/", 
                               name::String = "U(1) PCAC correlator with Nf=2", 
                               ID::String = "corr_pcac", 
                               mesdir::String = "measurements/", 
                               extension::String = ".txt")
        dt = Dates.now()
        wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
        lp = u1ws.params
        filepath = joinpath(wdir, mesdir, ID*wdir_sufix*extension)
        R1 = LFTU1.to_device(u1ws.device, zeros(complex(Float64), lp.iL..., 2))
        R2 = copy(R1)
        S = copy(R1)
        S0 = copy(R1)
        C = zeros(Float64, lp.iL[1])
        history = []
        return new(name, ID, filepath, R1, R2, S, S0, C, history)
    end
end
export U1PCACCorrelator

function pcac_correlation_function(corrws::AbstractU1Correlator, t, u1ws::U1Nf2)

    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)
    c = zero(ComplexF64)
    d = zero(ComplexF64)

    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.R1[x,t,1]
        b = corrws.R1[x,t,2]
        c = corrws.R2[x,t,1]
        d = corrws.R2[x,t,2]

        Ct += -imag(a*conj(c)) - imag(b*conj(d))
    end
    LFTU1.allowscalar(u1ws.device)
    Ct *= 2

    # NOTE: another test would be to check if there is imaginary part of Ct

    return Ct
end

function pcac_correlation_function(corrws::AbstractU1Correlator, u1ws::U1Nf2)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = pcac_correlation_function(corrws, t, u1ws)
    end
end


function (corrws::AbstractU1Correlator)(u1ws::U1Nf2)
    invert_sources!(corrws, u1ws)
    pcac_correlation_function(corrws, u1ws)
    return nothing
end

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")
cfile = ARGS[1]

ncfgs = LFTSampling.count_configs(cfile)
fb, model = read_cnfg_info(cfile, U1Nf2)
pws = U1PionCorrelator(model, wdir=dirname(cfile))
pcac = U1PCACCorrelator(model, wdir=dirname(cfile))
for i in ProgressBar(1:ncfgs)
    read_next_cnfg(fb, model)
    pws(model)
    pcac(model)
    write(pws)
    write(pcac)
end
close(fb)
