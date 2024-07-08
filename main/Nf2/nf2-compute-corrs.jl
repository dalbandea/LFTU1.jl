# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ProgressBars
using ArgParse

parse_commandline() = parse_commandline(ARGS)
function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
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
    end
    return parse_args(args, s)
end

args = [
"--ens", "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nf2sim-b4.0-L24-m0.02_D2024-04-12-17-03-40.579/Nf2sim-b4.0-L24-m0.02_D2024-04-12-17-03-40.579.bdio",
"--start", "2",
"--nconf", "5",
]
parsed_args = parse_commandline(args)

parsed_args = parse_commandline(ARGS)

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Nf2)

wdir=joinpath(dirname(cfile),"pointcorrs")
pws = U1PionCorrelator(model, ID="corr_pion_confs$start-$finish", wdir=wdir)
pcac = U1PCACCorrelator(model, ID="corr_pcac_confs$start-$finish", wdir=wdir)
for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    pws(model)
    pcac(model)
    write(pws)
    write(pcac)
end
close(fb)
