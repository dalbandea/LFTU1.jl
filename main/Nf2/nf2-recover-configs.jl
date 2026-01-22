using Revise
import Pkg
Pkg.activate(".")
using TOML
using LFTSampling
using LFTU1
using Dates
import KernelAbstractions

# Read model parameters

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")


if length(ARGS) == 1
    infile = ARGS[1]
else
    infile = "main/Nf2/infile.in"
end
pdata = TOML.parsefile(infile)

const NFL = 2

wdir = pdata["Working directory"]["wdir"]
savename = pdata["Working directory"]["savename"]
configfile = wdir*"/"*savename*".bdio"
newconfigfile = wdir*"/"*savename*"_new"*".bdio"

beta = pdata["Model params"]["beta"]
mass = pdata["Model params"]["mass"]
lsize = pdata["Model params"]["L"]
tsize = pdata["Model params"]["T"]
BC = eval(Meta.parse(pdata["Model params"]["BC"]))

model = U1Nf2(
    Float64,
    beta = beta,
    am0 = mass,
    iL = (lsize, tsize),
    BC = PeriodicBC,
    device = KernelAbstractions.CPU(),
    )

randomize!(model)

ncfgs = LFTSampling.count_configs(configfile)
fb, model = read_cnfg_info(configfile, U1Nf2)

for i in 1:ncfgs-1
    LFTSampling.read_next_cnfg(fb, model)
    save_cnfg(newconfigfile, model)
end
close(fb)
mv(newconfigfile, configfile, force=true)
