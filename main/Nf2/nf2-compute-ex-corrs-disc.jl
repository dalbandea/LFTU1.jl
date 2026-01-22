# Quantum Rotor
import Pkg
Pkg.activate(".")
using Revise
using LFTSampling

using LFTU1
# using ProgressBars
using Dates
import LinearAlgebra
using ArgParse
using Distributed
using TOML
using HDF5

parse_commandline() = parse_commandline(ARGS)
function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "-L"
        help = "lattice spatial size"
        required = true
        arg_type = Int

        "-T"
        help = "lattice temporal size"
        required = true
        arg_type = Int

        "--start"
        help = "start from configuration"
        required = false
        arg_type = Int
        default = 1

        "--nconf"
        help = "number of configurations to analyze; 0 means until the end"
        required = false
        arg_type = Int
        default = 0

        "--cfgsstep"
        help = "Separation between configurations in the Markov chain"
        required = true
        arg_type = Int

        "--ens"
        help = "path to ensemble with configurations"
        required = true
        arg_type = String

        "--sens"
        help = "path to save measurements"
        required = true
        arg_type = String
        # default = "configs/"

        "--Pmax"
        help = "Highest momentum frame considered"
        required = true
        arg_type = Int

        "--Onum"
        help = "Number of two-particle operators per frame"
        required = true
        arg_type = Int

        "--tstep"
        help = "Separation of source times"
        required = true
        arg_type = Int

end
return parse_args(args, s)
end

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")


if length(ARGS) == 1
    infile = ARGS[1]
else
    infile = "main/Nf2/infile.in"
end
parsed_args = TOML.parsefile(infile)

# args = [
#     "-L", "12",
#     "-T", "10",
#     "--ens", "Tests/ensemble1.bdio",
#     "--start", "1",
#     "--nconf", "1",
#     "--Pmax", "3",
#     "--Onum", "5",
#     ]
# parsed_args = parse_commandline(args)

const NFL = 2
const NL0 = parsed_args["L"]
const NT0 = parsed_args["T"]
const tstep = parsed_args["tstep"]
const Pmax = parsed_args["Pmax"]
const Onum = parsed_args["Onum"]

cfile = parsed_args["ens"]
spath = parsed_args["sens"]
isfile(cfile) || error("Path provided is not a file")
NT0 % tstep == 0 || error("tstep needs to divide NT0")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
cfgsstep = parsed_args["cfgsstep"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + (ncfgs-1)*cfgsstep

# fb, model = read_cnfg_info(cfile, U1Nf2)

function getAllTwoMomentum()
    mom_comb = zeros(Int, (Pmax+1, Onum, 2))
    pmax = 0
    pmin = 0
    for P in 0:Pmax
        for i in 0:Onum-1
            newpmax = ceil((P+1) ÷ 2) + i
            mom_comb[P+1,i+1,1] = newpmax
            pmax = max(pmax, newpmax)

            newpmin = floor(P ÷ 2) - i
            mom_comb[P+1,i+1,2] = newpmin
            pmin = min(pmin, newpmin)
        end
    end
    abspmax = max(pmax, -pmin)

    return mom_comb, abspmax
end

mom_comb, abspmax = getAllTwoMomentum()
pnum = 2 * abspmax + 1

data = (
    Bini = zeros(complex(Float64), abspmax + 1, 4, NT0),
    Bfin = zeros(complex(Float64), abspmax + 1, 4, NT0),
    BB = zeros(complex(Float64), abspmax + 1, 4, 4),
    Vini = zeros(complex(Float64), Pmax + 1, Onum, NT0),
    Vfin = zeros(complex(Float64), Pmax + 1, Onum, NT0),
    VV = zeros(complex(Float64), Pmax + 1, Onum^2*2),
    Tdisini = zeros(complex(Float64), Pmax + 1, 4, Onum),
    Tdisfin = zeros(complex(Float64), Pmax + 1, 4, Onum),
    )
Data = typeof(data)

function reset_data(data::Data)
    reset_data_disconnected(data)
    reset_data_bubbles(data)
end

function reset_data_disconnected(data::Data)
    data.BB .= 0.0
    data.VV .= 0.0
    data.Tdisini .= 0.0
    data.Tdisfin .= 0.0
    return nothing
end

function reset_data_bubbles(data::Data)
    data.Bini .= 0.0
    data.Bfin .= 0.0
    data.Vini .= 0.0
    data.Vfin .= 0.0
end

"""
- Computes all correlation functions required to study pion-pion scattering, in all isospin channels. Includes correlation functions between a two-pion and a single-particle state.
""" 
function multiplyPhaseLeft(correlator, mom)
    if mom == 0
        return correlator
    end
    L = NL0
    x = collect(range(0.,length=L))
    expfactor = exp.(- 1im * x * mom * 2. * pi / L)
    for y in 1:L
        correlator[1:L, y] = expfactor .* correlator[1:L, y]
        correlator[L+1:2*L, y] = expfactor .* correlator[L+1:2*L, y]
        correlator[1:L, L+y] = expfactor .* correlator[1:L, L+y]
        correlator[L+1:2*L, L+y] = expfactor .* correlator[L+1:2*L, L+y]
    end
    return correlator
end

function multiplyPhaseRight(correlator, mom)
    if mom == 0
        return correlator
    end
    L = NL0
    x = collect(range(0.,length=L))
    expfactor = exp.(- 1im * x * mom * 2. * pi / L)
    for y in 1:L
        correlator[y,1:L] = expfactor .* correlator[y, 1:L]
        correlator[y, L+1:2*L] = expfactor .* correlator[y, L+1:2*L]
        correlator[L+y, 1:L] = expfactor .* correlator[L+y, 1:L]
        correlator[L+y, L+1:2*L] = expfactor .* correlator[L+y, L+1:2*L]
    end
    return correlator
end

function oneEndTrick(correlator)
    aux = zeros(complex(Float64),size(correlator))
    res = zeros(complex(Float64),size(correlator))
    L = NL0
    aux[1:L, 1:L] = correlator[1:L, 1:L]
    aux[1:L, L+1:2*L] = - correlator[1:L, L+1:2*L]
    aux[L+1:2*L, 1:L] = correlator[L+1:2*L, 1:L]
    aux[L+1:2*L, L+1:2*L] = - correlator[L+1:2*L, L+1:2*L]
    aux = aux'
    res[1:L, 1:L] = aux[1:L, 1:L]
    res[1:L, L+1:2*L] = aux[1:L, L+1:2*L]
    res[L+1:2*L, 1:L] = - aux[L+1:2*L, 1:L]
    res[L+1:2*L, L+1:2*L] = - aux[L+1:2*L, L+1:2*L]
    return res #Conjugate transpose to do the one-end trick
end

function gammag5g0(correlator)
    res = zeros(complex(Float64),size(correlator))
    L = NL0
    res[1:L, 1:L] = -1im * (correlator[L+1:2*L, 1:L])
    res[1:L, L+1:2*L] = -1im * (correlator[L+1:2*L, L+1:2*L])
    res[L+1:2*L, 1:L] = 1im * (correlator[1:L, 1:L])
    res[L+1:2*L, L+1:2*L] = 1im * (correlator[1:L, L+1:2*L])
    return res
end

function gammaId(correlator)
    res = zeros(complex(Float64),size(correlator))
    L = NL0
    res[1:L, 1:L] = correlator[1:L, 1:L]
    res[1:L, L+1:2*L] = correlator[1:L, L+1:2*L]
    res[L+1:2*L, 1:L] = -correlator[L+1:2*L, 1:L]
    res[L+1:2*L, L+1:2*L] = -correlator[L+1:2*L, L+1:2*L]
    return res
end

function gammaIg0(correlator)
    res = zeros(complex(Float64),size(correlator))
    L = NL0
    res[1:L, 1:L] =  correlator[L+1:2*L, 1:L]
    res[1:L, L+1:2*L] =  correlator[L+1:2*L, L+1:2*L]
    res[L+1:2*L, 1:L] =  correlator[1:L, 1:L]
    res[L+1:2*L, L+1:2*L] =  correlator[1:L, L+1:2*L]
    return res
end

function solveCtt(gD)
    res = zeros(complex(Float64),NT0,2*NL0,2*NL0)
    source = zeros(complex(Float64),NL0*NT0*2)
    for t in 0:NT0-1
        for x in 1:NL0
            for s in 0:1
                source[x + t * NL0 + s * NL0 * NT0] =  1.
                solve = gD \ source
                res[t+1, 1:NL0, x+s*NL0] = solve[1 + NL0*t : NL0 * (t+1)]
                res[t+1, NL0+1:2*NL0, x+s*NL0] = solve[1 + NL0 * t + NL0*NT0 : NL0 * (t+1) + NL0*NT0]
                source[x + t * NL0 + s * NL0 * NT0] = 0.
            end
        end
    end
    return res
end

function getCorrelator(gD, tini)
    res = zeros(complex(Float64),NT0,2*NL0,2*NL0)
    source = zeros(complex(Float64),NL0*NT0*2)
    for x in 1:NL0
        for s in 0:1
            source[x + (tini-1) * NL0 + s * NL0 * NT0] =  1.
            solve = gD \ source
            for t in 0:NT0-1
                res[t+1, 1:NL0, x + s * NL0] = solve[1 + NL0 * t : NL0 * (t + 1)]
                res[t+1, NL0+1:2*NL0, x + s * NL0] = solve[1 + NL0 * t + NL0*NT0 : NL0 * (t+1) + NL0*NT0]
            end
            source[x + (tini-1) * NL0 + s * NL0 * NT0] = 0.
        end
    end
    return res
end

gsignini = [+1im, +1.0, +1.0, -1.0]
gsignfin = [+1im, +1.0, +1.0, +1.0]

function computeTwoPionCorrelationFunction(sfile, data::Data, gD)

    groups = createdatasets(sfile)
    reset_data_bubbles(data)

    Cgtt = [zeros(complex(Float64),2*NL0,2*NL0) for i=1:3]

    correlatortt = solveCtt(gD)

    L = NL0
    T = NT0

    for tini in 1:NT0

        ctt = correlatortt[tini,:,:]
        aux = copy(ctt)

        Cgtt[1] = gammag5g0(ctt)
        Cgtt[2] = gammaId(ctt)
        Cgtt[3] = gammaIg0(ctt)

        p1old = 0
        p2old = 0
        ptotold2 = 0
        for ptot in 0:Pmax
            for o in 1:Onum
                q1 = mom_comb[ptot+1,(o-1)%Onum + 1,o <= Onum ? 1 : 2]
                q2 = mom_comb[ptot+1,(o-1)%Onum + 1,o <= Onum ? 2 : 1]
                ctt = multiplyPhaseRight(ctt, q2 - p2old)
                ctt = multiplyPhaseLeft(ctt, q1 - p1old)
                data.Vfin[ptot+1,o,tini] = sum(transpose(ctt) .* (aux))
                data.Bfin[ptot+1,1,tini] = gsignfin[1] * LinearAlgebra.tr(ctt)
                ctt = multiplyPhaseRight(ctt, -2*q2)
                ctt = multiplyPhaseLeft(ctt, -2*q1)
                data.Vini[ptot+1,o,tini] = sum(transpose(ctt) .* (aux))
                data.Bini[ptot+1,1,tini] = gsignini[1] * LinearAlgebra.tr(ctt)
                p1old = -q1
                p2old = -q2
            end
        end
        ctt = multiplyPhaseRight(ctt, - p2old)
        ctt = multiplyPhaseLeft(ctt, - p1old)

        ptotold2 = 0
        ptotold3 = 0
        for ptot in 0:abspmax

            if ptot > Pmax
                ctt = multiplyPhaseLeft(ctt, ptot - ptotold2)
                data.Bfin[ptot+1,1,tini] += gsignfin[1] * LinearAlgebra.tr(ctt)
                ctt = multiplyPhaseLeft(ctt, -2*ptot)
                data.Bini[ptot+1,1,tini] += gsignini[1] * LinearAlgebra.tr(ctt)
                ptotold2 = -ptot
            end
            for g in 1:3
                Cgtt[g] = multiplyPhaseLeft(Cgtt[g], ptot - ptotold2)
                data.Bfin[ptot+1,g+1,tini] += gsignfin[g+1] * LinearAlgebra.tr(Cgtt[g])
                Cgtt[g] = multiplyPhaseLeft(Cgtt[g], -2*ptot)
                data.Bini[ptot+1,g+1,tini] += gsignini[g+1] * LinearAlgebra.tr(Cgtt[g])
            end
            ptotold3 = ptot
        end
        ctt = multiplyPhaseLeft(ctt, - ptotold2)
    end

    save_allt(data, sfile)

    for dt in 0:NT0-1
        reset_data_disconnected(data)
        for tini in 1:NT0
            tfin = (tini + dt - 1) % T + 1
            for ptot in 0:Pmax
                id = 1
                for ini in 1:Onum
                    for fin in 1:Onum
                        data.VV[ptot+1,id] += data.Vini[ptot+1,ini,tini] * data.Vfin[ptot+1,fin,tfin] / (T)
                        id += 1
                    end
                    for g in 1:4
                        data.Tdisini[ptot+1,g,ini] += data.Bini[ptot+1,g,tini] * data.Vfin[ptot+1,ini,tfin] / (T)
                        data.Tdisfin[ptot+1,g,ini] += data.Bfin[ptot+1,g,tfin] * data.Vini[ptot+1,ini,tini] / (T)
                    end
                end
            end
            for ptot in 0:abspmax
                for g1 in 1:4
                    for g2 in 1:4
                        data.BB[ptot+1,g1,g2] += data.Bini[ptot+1,g1,tini] * data.Bfin[ptot+1,g2,tfin] / (T)
                    end
                end
            end
        end
        save_singletdisconnected(data, sfile, dt)
    end

    return nothing
end

function createdatasets(sfile)

    gBB = create_group(sfile, "BB")
    gVV = create_group(sfile, "VV")
    gTdisini = create_group(sfile, "Tdisini")
    gTdisfin = create_group(sfile, "Tdisfin")

    for p in 0:abspmax
        gBBmom = create_group(gBB, "P$(p)")
        for g1 in 1:4
            for g2 in 1:4
                create_dataset(gBBmom, "mini$(g1)_mfin$(g2)", complex(Float64), NT0)
            end
        end
    end


    for p in 0:Pmax
        gVVmom = create_group(gVV, "P$(p)")
        gTdisinimom = create_group(gTdisini, "P$(p)")
        gTdisfinmom = create_group(gTdisfin, "P$(p)")
        for ini in 1:2*Onum
            k1 = mom_comb[p+1,(ini-1)%Onum + 1,ini <= Onum ? 1 : 2]
            k2 = mom_comb[p+1,(ini-1)%Onum + 1,ini <= Onum ? 2 : 1]
            if ini > Onum && k1 == k2
                continue
            end
            for fin in 1:2*Onum
                p1 = mom_comb[p+1,(fin-1)%Onum + 1,fin <= Onum ? 1 : 2]
                p2 = mom_comb[p+1,(fin-1)%Onum + 1,fin <= Onum ? 2 : 1]
                if fin > Onum && p1 == p2
                    continue
                end

                if ini <= Onum
                    if fin <= Onum
                        create_dataset(gVVmom, "ini_$(k1)_$(k2)_fin_$(p1)_$(p2)", complex(Float64), NT0)
                    end
                end
            end

            for g in 1:4
                create_dataset(gTdisinimom, "mini$(g)_fin_$(k1)_$(k2)", complex(Float64), NT0)
                create_dataset(gTdisfinmom, "ini_$(k1)_$(k2)_mfin$(g)", complex(Float64), NT0)
            end
        end
    end

    return nothing
end

function save_singletdisconnected(data, sfile, t)

    for p in 0:abspmax
        for g1 in 1:4
            for g2 in 1:4
                sfile["BB/P$(p)/mini$(g1)_mfin$(g2)"][t+1] = data.BB[p+1,g1,g2]
            end
        end
    end


    for p in 0:Pmax
        id = 1
        for ini in 1:Onum
            k1 = mom_comb[p+1,(ini-1)%Onum + 1,ini <= Onum ? 1 : 2]
            k2 = mom_comb[p+1,(ini-1)%Onum + 1,ini <= Onum ? 2 : 1]
            for fin in 1:Onum
                p1 = mom_comb[p+1,(fin-1)%Onum + 1,fin <= Onum ? 1 : 2]
                p2 = mom_comb[p+1,(fin-1)%Onum + 1,fin <= Onum ? 2 : 1]
                sfile["VV/P$(p)/ini_$(k1)_$(k2)_fin_$(p1)_$(p2)"][t+1] = data.VV[p+1,id]
                id += 1
            end
            for g in 1:4
                sfile["Tdisini/P$(p)/mini$(g)_fin_$(k1)_$(k2)"][t+1] = data.Tdisini[p+1,g,ini]
                sfile["Tdisfin/P$(p)/ini_$(k1)_$(k2)_mfin$(g)"][t+1] = data.Tdisfin[p+1,g,ini]
            end
        end
    end

    return nothing
end

function save_allt(data::Data, sfile)

    gBini = create_group(sfile, "Bini")
    gBfin = create_group(sfile, "Bfin")

    for p in 0:abspmax
        gBinimom = create_group(gBini, "P$(p)")
        gBfinmom = create_group(gBfin, "P$(p)")
        for g in 1:4
            write_dataset(gBinimom, "mini$(g)", data.Bini[p+1, g, :])
            write_dataset(gBfinmom, "mfin$(g)", data.Bfin[p+1, g, :])
        end
    end

    gVini = create_group(sfile, "Vini")
    gVfin = create_group(sfile, "Vfin")

    for p in 0:Pmax
        gVinimom = create_group(gVini, "P$(p)")
        gVfinmom = create_group(gVfin, "P$(p)")
        for ini in 1:Onum
            k1 = mom_comb[p+1,ini,1]
            k2 = mom_comb[p+1,ini,2]
            write_dataset(gVinimom, "ini_$(k1)_$(k2)", data.Vini[p+1,ini,:])
            write_dataset(gVfinmom, "fin_$(k1)_$(k2)", data.Vfin[p+1,ini,:])
        end
    end

    return nothing
end

function save_topcharge(model, sfile)
    write_dataset(sfile, "top_charge", top_charge(model))
end

# function write_vector(vec, filepath)
#     global io_stat = open(filepath, "a")
#     write(io_stat, "$(vec[1])")
#     for i in 2:length(vec)
#         write(io_stat, ", $(vec[i])")
#     end
#     write(io_stat, "\n")
#     close(io_stat)
#     return nothing
# end


fb, model = read_cnfg_info(cfile, U1Nf2)
pws =  U1exCorrelator(model, wdir=dirname(cfile))
for i in start:cfgsstep:finish
    @time begin
        if i == start && start != 1
            LFTSampling.read_cnfg_n(fb, start, model)
        else
            read_next_cnfg(fb, model)
        end
        sfile = h5open(joinpath(spath, "measurements$(i).h5"), "w")
        save_topcharge(model, sfile)
        construct_gD!(pws, model, model.params.am0[1])
        computeTwoPionCorrelationFunction(sfile, data, LinearAlgebra.lu(pws.gD))
        close(sfile)
    end
end
close(fb)

