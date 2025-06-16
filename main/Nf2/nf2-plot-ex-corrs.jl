using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using Printf
using Statistics
using Plots
default(legend = false)
using LaTeXStrings
import DelimitedFiles
import ADerrors
import LsqFit


function read_trace_timeslices(filename)
    trace_timeslices = Vector{ADerrors.uwreal}(undef, tsize)
    filepath = joinpath(@__DIR__, "measurements", filename*".txt")
    data = DelimitedFiles.readdlm(filepath, ',')
    for t in 1:tsize
        trace_timeslices[t] = ADerrors.uwreal(data[:,t], "$filename at time $t")
    end
    return trace_timeslices
end


function get_pion_correlation_function(traces_timeslices)
    exconn11_timeslices = traces_timeslices[1]
    for t in 1:tsize
        ADerrors.uwerr(exconn11_timeslices[t])
    end
    return exconn11_timeslices
end


function get_etaprime_correlation_function(traces_timeslices)
    exconn11_timeslices = traces_timeslices[1]
    exdisc11_timeslices = traces_timeslices[2]
    result = exconn11_timeslices .- 2 .* exdisc11_timeslices
    for t in 1:tsize
        ADerrors.uwerr(result[t])
    end
    return result
end


function get_2pion_isospin2_channel_correlation_function(traces_timeslices)
    exD1111_timeslices = traces_timeslices[1]
    exC1111_timeslices = traces_timeslices[2]
    result = 2 .* (exD1111_timeslices .- exC1111_timeslices)
    for t in 1:tsize
        ADerrors.uwerr(result[t])
    end
    return result
end


function derivative_corr(temporal_sites, corr)
    result = Vector{ADerrors.uwreal}(undef, length(corr)-2)
    for tt in 1:length(corr)-2
        if temporal_sites[tt+1] > tsize÷2
            result[tt] = (corr[tt+2] - corr[tt]) / 2
        else
            result[tt] = -(corr[tt+2] - corr[tt]) / 2
        end
        ADerrors.uwerr(result[tt])
    end
    return temporal_sites[2:length(corr)-1], result
end


function effective_mass_function(temporal_sites, corr)
    result = Vector{ADerrors.uwreal}(undef, length(corr)-2)
    for tt in 1:length(corr)-2
        result[tt] = (corr[tt] + corr[tt+2]) / (2 * corr[tt+1])
        result[tt] = acosh(result[tt])
        ADerrors.uwerr(result[tt])
    end
    return temporal_sites[2:length(corr)-1], result
end


function plot_correlation_function(name, get_correlation_function, traces_timeslices)
    corr = get_correlation_function(traces_timeslices)
    expval_corr = Vector{Float64}(undef, tsize)
    error_corr = Vector{Float64}(undef, tsize)
    for t in 1:tsize
        expval_corr[t] = ADerrors.value(corr[t])
        error_corr[t] = ADerrors.err(corr[t])
    end
    plot(temporal_sites, expval_corr, yerror=error_corr, xlabel=L"t", ylabel=L"C(t)")
    savefig(joinpath(@__DIR__, "$(name)_corr.svg"))

    correlator_fit_function(t, p) = p[1] .* cosh.(p[2].*(t.-tsize./2)) .+ p[3]
    p0 = [0.1, 0.1, 0.1]
    fit_to_data = LsqFit.curve_fit(correlator_fit_function, temporal_sites, expval_corr, p0)
    p = fit_to_data.param
    plot(temporal_sites, expval_corr, xlabel=L"t", ylabel=L"C(t)", label="HMC", title="The fitted mass: $(@sprintf("%.3f", p[2]))")
    plot!(temporal_sites, correlator_fit_function(temporal_sites, p), linestyle=:dash, label="fit", legend=:top)
    savefig(joinpath(@__DIR__, "$(name)_corr_fit.svg"))

    if name in ["2pion_isospin2_channel"]
        ts, dcorr = derivative_corr(temporal_sites, corr)
        ts, effective_mass = effective_mass_function(ts, dcorr)
    else
        ts, effective_mass = effective_mass_function(temporal_sites, corr)
    end
    expval_effective_mass = Vector{Float64}(undef, length(ts))
    error_effective_mass = Vector{Float64}(undef, length(ts))
    for t in 1:length(ts)
        expval_effective_mass[t] = ADerrors.value(effective_mass[t])
        error_effective_mass[t] = ADerrors.err(effective_mass[t])
    end
    plot(
        ts[1:length(ts)÷2],
        expval_effective_mass[1:length(ts)÷2],
        yerror=error_effective_mass[1:length(ts)÷2],
        xlabel=L"t", ylabel=L"m_\mathrm{eff}(t)",
        title="The effective mass: "*@sprintf("%.3f",
        expval_effective_mass[tsize÷4+2])*L"\pm"*@sprintf("%.3f",
        error_effective_mass[tsize÷4+2]))
    savefig(joinpath(@__DIR__, "$(name)_eff-mass.svg"))
end


# Configurations
filename = "main"
fb, model = read_cnfg_info(joinpath(@__DIR__, filename*".bdio"), U1Nf2)
tsize = model.params.iL[2]
temporal_sites = [1:tsize;]


exconn11_timeslices = read_trace_timeslices("exconn-11_confs500-1499")
exdisc11_timeslices = read_trace_timeslices("exdisc-11_confs500-1499")

plot_correlation_function("pion", get_pion_correlation_function, [exconn11_timeslices])
plot_correlation_function("etaprime", get_etaprime_correlation_function, [exconn11_timeslices, exdisc11_timeslices])


exD1111_timeslices = read_trace_timeslices("exD-1111_confs500-1499")
exC1111_timeslices = read_trace_timeslices("exC-1111_confs500-1499")

plot_correlation_function("2pion_isospin2_channel", get_2pion_isospin2_channel_correlation_function, [exD1111_timeslices, exC1111_timeslices])
