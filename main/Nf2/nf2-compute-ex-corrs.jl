# Quantum Rotor
import Pkg
Pkg.activate(".")
Pkg.add("Revise")
using Revise
using LFTSampling
using LFTU1
# using ProgressBars
using Dates
import LinearAlgebra
using ArgParse
using Distributed
using TOML

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

        "--ens"
        help = "path to ensemble with configurations"
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
#     "-L", "48",
#     "-T", "48",
#     "--ens", "Tests/Nf2sim-b5.0-L48-T48-m-0.03_D2024-11-11-10-09-44.301/Nf2sim-b5.0-L48-T48-m-0.03_D2024-11-11-10-09-44.301.bdio",
#     "--start", "1",
#     "--nconf", "1",
#     "--Pmax", "0",
#     "--Onum", "2",
#     ]
# parsed_args = parse_commandline(args)

const NFL = 2
const NL0 = parsed_args["L"]
const NT0 = parsed_args["T"]
const Pmax = parsed_args["Pmax"]
const Onum = parsed_args["Onum"]

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Nf2)

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
    P = zeros(complex(Float64), abspmax + 1, NT0),
    Pr = zeros(complex(Float64), abspmax + 1, NT0),
    Ps = zeros(complex(Float64), abspmax + 1, NT0),
    disc = zeros(complex(Float64), abspmax + 1, NT0),
    discs = zeros(complex(Float64), abspmax + 1, NT0),
    Delta = zeros(complex(Float64), 2*abspmax + 1, NT0),
    Deltas = zeros(complex(Float64), 2*abspmax + 1, NT0),
    Vini = zeros(complex(Float64), Pmax+1, Onum, NT0),
    Vfin = zeros(complex(Float64), Pmax+1, Onum, NT0),
    R = zeros(complex(Float64), Pmax+1, Onum^2*2, NT0), #This is not the exact number. I am overcounting those operators with p1=p2. However, the computations are always performed once and never overdone
    C = zeros(complex(Float64), Pmax+1, Onum^2*2, NT0),
    D = zeros(complex(Float64), Pmax+1, Onum^2*2, NT0),
    VV = zeros(complex(Float64), Pmax+1, Onum^2*2, NT0),
    Tsini = zeros(complex(Float64), Pmax+1, Onum*2, NT0),
    Tsinidis = zeros(complex(Float64), Pmax+1, Onum, NT0),
    Tsfin = zeros(complex(Float64), Pmax+1, Onum*2, NT0),
    Tsfindis = zeros(complex(Float64), Pmax+1, Onum, NT0),
    Trini = zeros(complex(Float64), Pmax+1, Onum*2, NT0),
    Trfin = zeros(complex(Float64), Pmax+1, Onum*2, NT0),
    )
Data = typeof(data)

function reset_data(data::Data)
    data.P .= 0.0
    data.Pr .= 0.0
    data.Ps .= 0.0
    data.disc .= 0.0
    data.discs .= 0.0
    data.Delta .= 0.0
    data.Deltas .= 0.0
    data.Vini .= 0.0
    data.Vfin .= 0.0
    data.R .= 0.0
    data.C .= 0.0
    data.D .= 0.0
    data.VV .= 0.0
    data.Tsini .= 0.0
    data.Tsinidis .= 0.0
    data.Tsfin .= 0.0
    data.Tsfindis .= 0.0
    data.Trini .= 0.0
    data.Trfin .= 0.0
    return nothing
end

"""
- Computes all correlation functions required to study pion-pion scattering, in all isospin channels. Includes correlation functions between a two-pion and a single-particle state.
""" 
function multiplyAllMomenta(correlator)
    id = 0
    L = NL0
    for p in -abspmax:abspmax
        id += 1
        if p == 0
            continue
        end
        x = collect(range(0.,length=L))
        expfactor = exp.(- 1im * x * p * 2. * pi / L)
        Threads.@threads for tini in 1:NT0
            for tend in 1:NT0
                for y in 1:L
                    correlator[id, tini, tend, 1:L, y] = expfactor .* correlator[1+abspmax, tini, tend, 1:L, y]
                    correlator[id, tini, tend, L+1:2*L, y] = expfactor .* correlator[1+abspmax, tini, tend, L+1:2*L, y]
                    correlator[id, tini, tend, 1:L, L+y] = expfactor .* correlator[1+abspmax, tini, tend, 1:L, L+y]
                    correlator[id, tini, tend, L+1:2*L, L+y] = expfactor .* correlator[1+abspmax, tini, tend, L+1:2*L, L+y]
                end
            end
        end
    end
end

function getPhases(abspmax, L)
    res =  Any[]
    for p in -abspmax:abspmax
        x = collect(range(0.,length=L))
        expfactor = exp.(- 1im * x * p * 2. * pi / L)
        push!(res, expfactor)
    end
    return res
end

function computeTwoPionCorrelationFunction(data::Data, corrws::U1exCorrelator, u1ws)
    reset_data(data)
    correlator = zeros(complex(Float64), 2*abspmax+1, NT0, NT0, NL0*2, NL0*2)
    g5_correlator = zeros(complex(Float64), 2*abspmax+1, NT0, NT0, NL0*2, NL0*2)
    MR = zeros(complex(Float64), 2*abspmax+1, NT0, NT0, NL0*2, NL0*2)

    phases = getPhases(abspmax, NL0)
    L = NL0
    T = NT0

    Threads.@threads for tini in 1:NT0
        for tend in 1:NT0
            correlator[abspmax+1, tini, tend, 1:L, 1:L] = corrws.invgD[1][1 + L * (tini-1) : L * (tini), 1 + L * (tend-1) : L * (tend)]
            correlator[abspmax+1, tini, tend, L+1:2*L, 1:L] = corrws.invgD[1][1 + L * (tini-1) + L*T : L * (tini) + L*T, 1 + L * (tend-1) : L * (tend)]
            correlator[abspmax+1, tini, tend, 1:L, L+1:2*L] = corrws.invgD[1][1 + L * (tini-1) : L * (tini), 1 + L * (tend-1) + L*T : L * (tend) + L*T]
            correlator[abspmax+1, tini, tend, L+1:2*L, L+1:2*L] = corrws.invgD[1][1 + L * (tini-1) + L*T : L * (tini) + L*T, 1 + L * (tend-1) + L*T : L * (tend) + L*T]
        end
    end
    corrws = nothing

    multiplyAllMomenta(correlator)

    for p in -abspmax:abspmax
        Threads.@threads for tini in 1:NT0
            for tend in 1:NT0
                MR[p+1+abspmax,tini,tend,:,:] = correlator[1+abspmax, tini, tini, :, :] * correlator[1+abspmax+p, tini, tend, :, :]
            end
        end
    end

    # Euclidean gamma matrices convention:
    # gamma_0 = [0 -I; I 0]
    # gamma_1 = [0 1; 1 0]
    # gamma_5 = [1 0; 0 -1]
    for P in -abspmax:abspmax
        Threads.@threads for tini in 1:NT0
            for tend in 1:NT0
                g5_correlator[P+1+abspmax, tini, tend, 1:L, 1:L] = correlator[P+abspmax+1, tini, tend, 1:L, 1:L]
                g5_correlator[P+1+abspmax, tini, tend, L+1:2*L, 1:L] = -correlator[P+abspmax+1, tini, tend, L+1:2*L, 1:L]
                g5_correlator[P+1+abspmax, tini, tend, 1:L, L+1:2*L] = correlator[P+abspmax+1, tini, tend, 1:L, L+1:2*L]
                g5_correlator[P+1+abspmax, tini, tend, L+1:2*L, L+1:2*L] = -correlator[P+abspmax+1, tini, tend, L+1:2*L, L+1:2*L]
            end
        end
    end

    all_correlators1(correlator, g5_correlator, MR, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)

    g5_correlator = zeros(complex(Float64), 0, 0, 0, 0, 0)
    mIg0_correlator = zeros(complex(Float64), 2*abspmax+1, NT0, NT0, NL0*2, NL0*2)

    for P in -abspmax:abspmax
        Threads.@threads for tini in 1:NT0
            for tend in 1:NT0
                mIg0_correlator[P+1+abspmax, tini, tend, 1:L, 1:L] = correlator[P+abspmax+1, tini, tend, L+1:2*L, 1:L]
                mIg0_correlator[P+1+abspmax, tini, tend, L+1:2*L, 1:L] = - correlator[P+abspmax+1, tini, tend, 1:L, 1:L]
                mIg0_correlator[P+1+abspmax, tini, tend, 1:L, L+1:2*L] = correlator[P+abspmax+1, tini, tend, L+1:2*L, L+1:2*L]
                mIg0_correlator[P+1+abspmax, tini, tend, L+1:2*L, L+1:2*L] = - correlator[P+abspmax+1, tini, tend, 1:L, L+1:2*L]
            end
        end
    end

    all_correlators3(correlator, mIg0_correlator, MR, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)

    mIg0_correlator = zeros(complex(Float64), 0, 0, 0, 0, 0)
    MR = zeros(complex(Float64), 0, 0, 0, 0, 0)
    MC = zeros(complex(Float64), 2*abspmax+1, NT0, NT0, NL0*2, NL0*2)

    for p in -abspmax:abspmax
        Threads.@threads for tini in 1:NT0
            for tend in 1:NT0
                MC[p+1+abspmax,tini,tend,:,:] = correlator[1+abspmax, tini, tend, :, :] * correlator[1+abspmax+p, tend, tini, :, :]
            end
        end
    end

    all_correlators2(MC, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)

    return nothing
end

function save_data(data::Data, dirpath, start)

    for p in -abspmax:abspmax
        deltafile = joinpath(dirpath, "measurements$(start)/exdelta_$(p)_confs$start-$finish.txt")
        write_vector(data.Delta[abspmax+p+1,:],deltafile)
        deltasfile = joinpath(dirpath, "measurements$(start)/exdeltas_$(p)_confs$start-$finish.txt")
        write_vector(data.Deltas[abspmax+p+1,:],deltasfile)
    end
    for p in 0:abspmax
        connfile = joinpath(dirpath,"measurements$(start)/exconn_$(p)_confs$start-$finish.txt")
        write_vector(data.P[p+1,:],connfile)
        connfiler = joinpath(dirpath,"measurements$(start)/exconn_rho_$(p)_confs$start-$finish.txt")
        write_vector(data.Pr[p+1,:],connfiler)
        connfiles = joinpath(dirpath,"measurements$(start)/exconn_sigma_$(p)_confs$start-$finish.txt")
        write_vector(data.Ps[p+1,:],connfiles)
        discfile = joinpath(dirpath, "measurements$(start)/exdisc_$(p)_confs$start-$finish.txt")
        write_vector(data.disc[p+1,:],discfile)
        discfiles = joinpath(dirpath, "measurements$(start)/exdisc_sigma_$(p)_confs$start-$finish.txt")
        write_vector(data.discs[p+1,:],discfiles)
    end

    for P in 0:Pmax
        id = 1
        for ini in 1:Onum
            q1 = mom_comb[P+1,ini,1]
            q2 = mom_comb[P+1,ini,2]
            for fin in 1:Onum
                p1 = mom_comb[P+1,fin,1]
                p2 = mom_comb[P+1,fin,2]

                Dfile = joinpath(dirpath, "measurements$(start)/exD_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")
                VVfile = joinpath(dirpath, "measurements$(start)/exVV_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")
                Cfile = joinpath(dirpath, "measurements$(start)/exC_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")
                Rfile = joinpath(dirpath, "measurements$(start)/exR_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")

                write_vector(data.R[P+1, id, :],Rfile)
                write_vector(data.C[P+1, id, :],Cfile)
                write_vector(data.D[P+1, id, :],Dfile)
                write_vector(data.VV[P+1, (ini-1)*Onum+fin, :],VVfile)

                id += 1
                if p1 == p2
                    continue
                end

                p2 = mom_comb[P+1,fin,1]
                p1 = mom_comb[P+1,fin,2]

                Dfile = joinpath(dirpath, "measurements$(start)/exD_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")
                Cfile = joinpath(dirpath, "measurements$(start)/exC_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")
                Rfile = joinpath(dirpath, "measurements$(start)/exR_P$(P)_ini_$(q1)_$(q2)_fin_$(p1)_$(p2)_confs$start-$finish.txt")

                write_vector(data.R[P+1, id, :],Rfile)
                write_vector(data.C[P+1, id, :],Cfile)
                write_vector(data.D[P+1, id, :],Dfile)

                id += 1
            end

        end

        id = 1
        for op in 1:Onum
            q1 = mom_comb[P+1,op,1]
            q2 = mom_comb[P+1,op,2]

            Vfile = joinpath(dirpath, "measurements$(start)/exVini_P$(P)_mom_$(q1)_$(q2)_confs$start-$finish.txt")
            write_vector(data.Vini[P+1, op, :],Vfile)

            Vfile = joinpath(dirpath, "measurements$(start)/exVfin_P$(P)_mom_$(q1)_$(q2)_confs$start-$finish.txt")
            write_vector(data.Vfin[P+1, op, :],Vfile)

            Tsinifile = joinpath(dirpath, "measurements$(start)/exTsini_P$(P)_fin_$(q1)_$(q2)_confs$start-$finish.txt")
            Tsinidisfile = joinpath(dirpath, "measurements$(start)/exTsinidis_P$(P)_fin_$(q1)_$(q2)_confs$start-$finish.txt")
            Trinifile = joinpath(dirpath, "measurements$(start)/exTrini_P$(P)_fin_$(q1)_$(q2)_confs$start-$finish.txt")
            Tsfinfile = joinpath(dirpath, "measurements$(start)/exTsfin_P$(P)_ini_$(q1)_$(q2)_confs$start-$finish.txt")
            Tsfindisfile = joinpath(dirpath, "measurements$(start)/exTsfindis_P$(P)_ini_$(q1)_$(q2)_confs$start-$finish.txt")
            Trfinfile = joinpath(dirpath, "measurements$(start)/exTrfin_P$(P)_ini_$(q1)_$(q2)_confs$start-$finish.txt")

            write_vector(data.Tsini[P+1, id, :],Tsinifile)
            write_vector(data.Tsinidis[P+1, op, :],Tsinidisfile)
            write_vector(data.Trini[P+1, id, :],Trinifile)
            write_vector(data.Tsfin[P+1, id, :],Tsfinfile)
            write_vector(data.Tsfindis[P+1, op, :],Tsfindisfile)
            write_vector(data.Trfin[P+1, id, :],Trfinfile)

            id += 1
            if q1 == q2
                continue
            end

            q2 = mom_comb[P+1,op,1]
            q1 = mom_comb[P+1,op,2]

            Tsinifile = joinpath(dirpath, "measurements$(start)/exTsini_P$(P)_fin_$(q1)_$(q2)_confs$start-$finish.txt")
            Trinifile = joinpath(dirpath, "measurements$(start)/exTrini_P$(P)_fin_$(q1)_$(q2)_confs$start-$finish.txt")
            Tsfinfile = joinpath(dirpath, "measurements$(start)/exTsfin_P$(P)_ini_$(q1)_$(q2)_confs$start-$finish.txt")
            Trfinfile = joinpath(dirpath, "measurements$(start)/exTrfin_P$(P)_ini_$(q1)_$(q2)_confs$start-$finish.txt")

            write_vector(data.Tsini[P+1, id, :],Tsinifile)
            write_vector(data.Trini[P+1, id, :],Trinifile)
            write_vector(data.Tsfin[P+1, id, :],Tsfinfile)
            write_vector(data.Trfin[P+1, id, :],Trfinfile)

            id += 1
        end
    end
end

# function correlators(data::Data, corrws::U1exCorrelator, u1ws)
#     reset_data(data)
#     for ifl in 1:2
#         ex_disconnected_correlator(corrws, u1ws, ifl)
#         data.Delta[ifl,:] .+= corrws.result
#         for jfl in 1:2
#             for it in 1:N0
#                 ex_connected_correlator(corrws, u1ws, it, ifl, jfl)
#                 for t in 1:N0
#                     tt=((t-it+N0)%N0+1);
#                     data.P[ifl,jfl,tt] += corrws.result[t] / N0
#                 end
#             end
#         end
#     end
#     compute_disconnected!(data)
#     return nothing
# end
#
# """
# Compute all combinations of disconnected traces and save them to data.disc[ifl, jfl, t]
# """
# function compute_disconnected!(data::Data)
#     data.disc .= 0.0
#     for ifl in 1:2, jfl in ifl:2
#         for t in 1:N0, tt in 1:N0
#             data.disc[ifl, jfl, t] += data.Delta[ifl, tt] * data.Delta[jfl, (tt+t-1-1)%N0+1] / N0
#         end
#     end
#     return nothing
# end
#
# function save_data(data::Data, dirpath)
#     for ifl in 1:2
#         deltafile = joinpath(dirpath, "measurements$(start)/exdelta-$(ifl)_confs$start-$finish.txt")
#         write_vector(data.Delta[ifl, :],deltafile)
#         for jfl in ifl:2
#             connfile = joinpath(dirpath,"measurements$(start)/exconn-$ifl$(jfl)_confs$start-$finish.txt")
#             discfile = joinpath(dirpath, "measurements$(start)/exdisc-$ifl$(jfl)_confs$start-$finish.txt")
#             write_vector(data.P[ifl, jfl, :],connfile)
#             write_vector(data.disc[ifl, jfl, :],discfile)
#         end
#     end
# end

function save_topcharge(model, dirpath, start)
    qfile = joinpath(dirpath,"measurements$(start)/topcharge_confs$start-$finish.txt")
    global io_stat = open(qfile, "a")
    write(io_stat, "$(top_charge(model))\n")
    close(io_stat)
end

function write_vector(vec, filepath)
    global io_stat = open(filepath, "a")
    write(io_stat, "$(vec[1])")
    for i in 2:length(vec)
        write(io_stat, ", $(vec[i])")
    end
    write(io_stat, "\n")
    close(io_stat)
    return nothing
end


pws =  U1exCorrelator(model, wdir=dirname(cfile))
for i in start:finish
    @time begin
        if i == start && start != 1
            LFTSampling.read_cnfg_n(fb, start, model)
        else
            read_next_cnfg(fb, model)
        end
        model.U .= 1
        construct_invgD!(pws, model)
        computeTwoPionCorrelationFunction(data, pws, model)
        save_data(data, dirname(cfile), start)
        save_topcharge(model, dirname(cfile), start)
    end
end
close(fb)



