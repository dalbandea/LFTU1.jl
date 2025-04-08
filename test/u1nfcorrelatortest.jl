import LinearAlgebra: norm, dot

let
    """
    Computes connected and disconnected (exact) traces with sink at t0 using
    point sources
    """
    function compute_trace(t0, corrws::U1Correlator, u1ws, ifl)
        S0 = corrws.S0
        S = corrws.S
        R = corrws.R
        lp = u1ws.params
        S0 .= zero(ComplexF64)
        trc = zeros(ComplexF64, 24)
        trd = zero(ComplexF64)
        for x in 1:24, s in 1:2
            S0[x,t0,s] = 1.0
            ## Solve g5D R = S0 for S for Flavor ifl
            iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0[ifl]), S0, model.sws, model)
            gamm5Dw!(R[ifl], S, model.params.am0[ifl], model)
            for t in 1:24
                trc[t] += dot(R[ifl][:,t,:], R[ifl][:,t,:]) / lp.iL[1]
            end
            trd += dot(S0, R[ifl]) / sqrt(lp.iL[1])
            S0[x,t0,s] = 0.0
        end
        return real.(trc), real(trd)
    end

    model = U1Nf(Float64,
                 beta = 4.0,
                 iL = (24, 24),
                 am0 = [0.02, 0.02],
                 BC = PeriodicBC,
                 # device = device,
                )

    N0 = model.params.iL[1]

    randomize!(model)

    smplr = HMC(integrator = OMF4(1.0, 4))
    samplerws = LFTSampling.sampler(model, smplr)

    for i in 1:10
        sample!(model, samplerws)
    end

    pws = U1exCorrelator(model)
    construct_invgD!(pws, model)

    P = zeros(Float64, length(pws.result))
    for it in 1:N0
        ex_connected_correlator(pws, model, it, 1, 1)
        for t in 1:N0
            tt=((t-it+N0)%N0+1);
            P[tt] += pws.result[t] / N0
        end
    end


    pws2 = U1Correlator(model)

    P2 = zeros(Float64, length(pws.result))
    Delta = zeros(Float64, N0)

    for it in 1:N0
        trc, trd = compute_trace(it, pws2, model, 1)
        Delta[it] += trd
        for t in 1:N0
            tt=((t-it+N0)%N0+1);
            P2[tt] += trc[t] / N0
        end
    end

    @testset verbose = true "Exact connected correlator" begin
        @test isapprox(norm(P2 .- P), 0, atol = 1e-7)
    end

    ex_disconnected_correlator(pws, model, 1)

    @testset verbose = true "Exact disconnected correlator" begin
        @test isapprox(norm(Delta .- pws.result), 0, atol = 1e-7)
    end
end




# function compute_trace_connected(t0, corrws::U1Correlator, u1ws, ifl, jfl)
#     S0 = corrws.S0
#     S = corrws.S
#     R = corrws.R
#     lp = u1ws.params
#     S0 .= zero(ComplexF64)
#     trc = zeros(ComplexF64, N0)
#     trd = zero(ComplexF64)
#     for x in 1:N0, s in 1:2
#         S0[x,t0,s] = 1.0
#         ## Solve g5D R = S0 for S for Flavor ifl
#         iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0[ifl]), S0, model.sws, model)
#         gamm5Dw!(R[ifl], S, model.params.am0[ifl], model)
#         iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0[jfl]), S0, model.sws, model)
#         gamm5Dw!(R[jfl], S, model.params.am0[jfl], model)
#         for t in 1:N0
#             trc[t] += dot(R[jfl][:,t,:], R[ifl][:,t,:]) / lp.iL[1]
#         end
#         trd += dot(S0, R[ifl]) / sqrt(lp.iL[1])
#         S0[x,t0,s] = 0.0
#     end
#     return real.(trc), real(trd)
# end


# model = U1Nf(Float64,
#              beta = 4.0,
#              iL = (6, 6),
#              am0 = [-0.02, 0.06],
#              BC = PeriodicBC,
#              # device = device,
#             )


# ns_rat = [6, 5]
# r_as = [0.06, 0.14]
# r_bs = [6.0, 6.0]
# model.rprm .= LFTU1.get_rhmc_params(ns_rat, r_as, r_bs)

# N0 = model.params.iL[1]

# randomize!(model)

# smplr = HMC(integrator = OMF4(1.0, 2))
# samplerws = LFTSampling.sampler(model, smplr)

# @time for i in 1:10
#     sample!(model, samplerws)
# end

# pws = U1exCorrelator(model)
# construct_invgD!(pws, model)

# P = zeros(Float64, length(pws.result))
# for it in 1:N0
#     ex_connected_correlator(pws, model, it, 1, 2)
#     for t in 1:N0
#         tt=((t-it+N0)%N0+1);
#         P[tt] += pws.result[t] / N0
#     end
# end


# pws2 = U1Correlator(model)

# P2 = zeros(Float64, length(pws2.result))
# Delta = zeros(Float64, N0)
# for it in ProgressBar(1:N0)
#     trc, trd = compute_trace_connected(it, pws2, model, 1, 2)
#     Delta[it] += trd
#     for t in 1:N0
#         tt=((t-it+N0)%N0+1);
#         P2[tt] += trc[t] / N0
#     end
# end

# P .- P2


# NFL = 2
# NSRC = 40000

# data = (
#     nc = 0,
#     P = zeros(Float64, NFL, NFL, N0),
#     disc = zeros(Float64, NFL, NFL, N0),
#     Delta = zeros(Float64, NFL, NSRC, N0)
# )


# function reset_data(data)
#     data.P .= 0.0
#     data.Delta .= 0.0
#     data.disc .= 0.0
#     return nothing
# end

# function correlators(data, corrws, u1ws, nsrc)
#     reset_data(data)
#     for isrc in ProgressBar(1:nsrc)
#         for it in 1:N0
#             random_source(it,corrws,u1ws)
#             for ifl in 1:2
#                 # disconnected_correlator(corrws, u1ws, ifl)
#                 # data.Delta[ifl,isrc,:] .+= corrws.result
#                 for jfl in ifl:2
#                     connected_correlator(corrws, u1ws, ifl, jfl)
#                     for t in 1:N0
#                         tt=((t-it+N0)%N0+1);
#                         data.P[ifl, jfl, tt] += corrws.result[t] ./ N0 ./ nsrc
#                     end
#                 end
#             end
#         end
#     end
# end


# correlators(data, pws2, model, NSRC)



# construct_gD!(pws, model, 2)

# gdc = copy(pws.gD)

# pws.gD .- gdc


# construct_invgD!(pws, model)

# pws.invgD[1] .- pws.invgD[2]



# cfile = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nfsim-b4.0-L24-m[0.02, 0.02]_D2024-04-18-18-42-12.272/Nfsim-b4.0-L24-m[0.02, 0.02]_D2024-04-18-18-42-12.272.bdio"

# cfile = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/results/2-non-degenerate/L64/Nfsim-b4.0-L64-m[-0.02,0.06]_D2024-04-30-13-10-42.216/Nfsim-b4.0-L64-m[-0.02,0.06]_D2024-04-30-13-10-42.216.bdio"


# fb, model = read_cnfg_info(cfile, U1Nf)

# LFTSampling.read_cnfg_n(fb, 1235, model)

# read_next_cnfg(fb, model)


# model.U


# NFL = 2
# NSRC = 40000

# N0 = model.params.iL[1]

# data = (
#     nc = 0,
#     P = zeros(Float64, NSRC, NFL, NFL, N0),
#     disc = zeros(Float64, NFL, NFL, N0),
#     Delta = zeros(Float64, NFL, NSRC, N0)
# )

# function correlators(data, corrws, u1ws, nsrc)
#     reset_data(data)
#     for isrc in ProgressBar(1:nsrc)
#         for it in 1:N0
#             random_source(it,corrws,u1ws)
#             for ifl in 1:2
#                 for jfl in ifl:2
#                     connected_correlator(corrws, u1ws, ifl, jfl)
#                     for t in 1:N0
#                         tt=((t-it+N0)%N0+1);
#                         data.P[isrc, ifl, jfl, tt] += corrws.result[t] ./ N0
#                     end
#                 end
#             end
#         end
#     end
# end

# correlators(data, pws2, model, NSRC)

# using ADerrors

# c0 = uwreal(data.P[:,1,2,1], "hi3")
# uwerr(c0)
# c0

# dif = c0 - P[1]
# uwerr(dif)
# dif

