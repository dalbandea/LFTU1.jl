# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using TOML

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")

if length(ARGS) == 1
    infile = ARGS[1]
else
    infile = "main/Nf2/infile.in"
end
pdata = TOML.parsefile(infile)
device = pdata["Model params"]["device"]
if device == "CUDA"
    import CUDA
    device = CUDA.device()
elseif device == "CPU"
    import KernelAbstractions
    device = KernelAbstractions.CPU()
else
    error("Only acceptable devices are CUDA or CPU")
end

using LFTSampling
using LFTU1
using Dates
using Logging

function create_simulation_directory(wdir::String, u1ws::U1Nf2)
    dt = Dates.now()
    wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
    fname = "Nf2sim-b$(u1ws.params.beta)-L$(model.params.iL[1])-m$(u1ws.params.am0)"*wdir_sufix
    fdir = joinpath(wdir, fname)
    configfile = joinpath(fdir, fname*".bdio")
    mkpath(fdir)
    cp(infile, joinpath(fdir,splitpath(infile)[end]))
    return configfile
end

# Read model parameters

beta = pdata["Model params"]["beta"]
mass = pdata["Model params"]["mass"]
lsize = pdata["Model params"]["L"]
BC = eval(Meta.parse(pdata["Model params"]["BC"]))

# Read HMC parameters

tau = pdata["HMC params"]["tau"]
nsteps = pdata["HMC params"]["nsteps"]
ntherm = pdata["HMC params"]["ntherm"]
ntraj = pdata["HMC params"]["ntraj"]
discard = pdata["HMC params"]["discard"]
integrator = eval(Meta.parse(pdata["HMC params"]["integrator"]))

# Working directory

wdir = pdata["Working directory"]["wdir"]


model = U1Nf2(
              Float64, 
              beta = beta, 
              am0 = mass, 
              iL = (lsize, lsize), 
              BC = PeriodicBC,
              device = device,
             )

randomize!(model)
smplr = HMC(integrator = integrator(tau, nsteps))
samplerws = LFTSampling.sampler(model, smplr)

@info "Creating simulation directory"

configfile = create_simulation_directory(wdir, model)
logio = open(dirname(configfile)*"/log.txt", "a+")
logger = SimpleLogger(logio)
global_logger(logger)

@info "U(1) NF=2 SIMULATION" model.params smplr

@info "Starting thermalization"

for i in 1:ntherm
    @info "THERM STEP $i"
    @time sample!(model, samplerws)
    flush(logio)
end

@info "Starting simulation"

@time for i in 1:ntraj
    @info "TRAJECTORY $i"
    for j in 1:discard
        @time sample!(model, samplerws)
    end
    @time sample!(model, samplerws)
    save_cnfg(configfile, model)
    flush(logio)
end

close(io)
