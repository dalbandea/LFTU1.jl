
#########################################
# Stochastic / point source Correlators #
#########################################


"""
Correlator struct to compute correlators stochastically or with point sources
"""
struct U1Correlator <: LFTU1.AbstractU1Correlator
    name::String
    ID::String
    filepath::String
    R
    S
    S0
    result::Vector{Float64} # correlator
    history::Vector{Vector{Float64}}
    function U1Correlator(u1ws::LFTU1.U1; wdir::String = "./trash/", 
                               name::String = "U(1) correlator", 
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
        return new(name, ID, filepath, [R1, R2], S, S0, C, history)
    end
end
export U1Correlator


"""
Generate complex random normal source at time slice t0 storing it to corrws.S0, and solve γ₅D corrws.R[ifl] = corrws.S0 for each flavor ifl.
"""
function random_source(t0, corrws, u1ws::U1Nf)
    S0 = corrws.S0
    S = corrws.S
    R = corrws.R
    lp = u1ws.params
    S0 .= zero(ComplexF64)
    S0[:,t0,:] .= randn(ComplexF64, lp.iL[1],2)
    for ifl in 1:2
        ## Solve g5D R = S0 for S for Flavor ifl
        iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(u1ws.params.am0[ifl]), S0, u1ws.sws, u1ws)
        gamm5Dw!(R[ifl], S, u1ws.params.am0[ifl], u1ws)
    end
    return nothing
end
export random_source

"""
Generate complex random normal source at time slice t0 storing it to corrws.S0,
and solve γ₅D corrws.R[ifl] = corrws.S0 for each flavor ifl. Intended for use
with U1Nf2 only
"""
function random_source(t0, corrws, u1ws::U1Nf2)
    S0 = corrws.S0
    S = corrws.S
    R = corrws.R
    lp = u1ws.params
    S0 .= zero(ComplexF64)
    S0[:,t0,:] .= randn(ComplexF64, lp.iL[1],2)
    for ifl in 1:2
        ## Solve g5D R = S0 for S for Flavor ifl
        iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(u1ws.params.am0), S0, u1ws.sws, u1ws)
        gamm5Dw!(R[ifl], S, u1ws.params.am0, u1ws)
    end
    return nothing
end
export random_source

"""
Computes dot(corrws.R[ifl], corrws.R[jfl]), i.e. (D_ifl⁻¹ η, D_jfl⁻¹ η), at time slice t.
"""
function connected_correlator(corrws::U1Correlator, t, u1ws, ifl, jfl)
    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)

    # NOTE: this should be ultraslow. It may be better to put R1 and R2 into the
    # CPU prior to calling this function. For GPU, the best one can do is to
    # reduce columns of the GPU array.
    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.R[jfl][x,t,:]
        b = corrws.R[ifl][x,t,:]
        Ct += real(dot(b,a)) / lp.iL[1]
    end
    LFTU1.disallowscalar(u1ws.device)

    return Ct
end
export connected_correlator

"""
Computes dot(corrws.S0, corrws.R[ifl]), i.e. (η^†, D_ifl⁻¹ η), at time slice t.
"""
function disconnected_correlator(corrws, t, u1ws, ifl)
    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)

    # NOTE: this should be ultraslow. It may be better to put R1 and R2 into the
    # CPU prior to calling this function. For GPU, the best one can do is to
    # reduce columns of the GPU array.
    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.S0[x,t,:]
        b = corrws.R[ifl][x,t,:]
        Ct += real(dot(b,a)) / sqrt(lp.iL[1])
    end
    LFTU1.disallowscalar(u1ws.device)

    return Ct
end
export disconnected_correlator

function connected_correlator(corrws::U1Correlator, u1ws, ifl, jfl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = connected_correlator(corrws, t, u1ws, ifl, jfl) |> real
    end
end

function disconnected_correlator(corrws::U1Correlator, u1ws, ifl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = disconnected_correlator(corrws, t, u1ws, ifl) |> real
    end
end


#####################
# Exact Correlators #
#####################

"""
U1 structure to build exact correlators. It computes the inverse exactly
"""
struct U1exCorrelator <: LFTU1.AbstractU1Correlator
    name::String
    ID::String
    filepath::String
    gD
    invgD
    e1
    e2
    result::Vector{Float64} # correlator
    history::Vector{Vector{Float64}}
    function U1exCorrelator(u1ws::LFTU1.U1; wdir::String = "./trash/", 
                               name::String = "U(1) correlator", 
                               ID::String = "excorr_pion", 
                               mesdir::String = "measurements/", 
                               extension::String = ".txt")
        dt = Dates.now()
        wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
        lp = u1ws.params
        filepath = joinpath(wdir, mesdir, ID*wdir_sufix*extension)
        V = prod((u1ws.params.iL..., 2))
        gD = LFTU1.to_device(u1ws.device, zeros(complex(Float64), V, V))
        invgD1 = copy(gD)
        invgD2 = copy(gD)
        e1 = LFTU1.to_device(u1ws.device, zeros(complex(Float64), lp.iL..., 2))
        e2 = copy(e1)
        C = zeros(Float64, lp.iL[1])
        history = []
        mkpath(dirname(filepath))
        return new(name, ID, filepath, gD, [invgD1, invgD2], e1, e2, C, history)
    end
end
export U1exCorrelator

"""
Translates a 3-dimensional tensor index (il1,il2,is) into a 1-dimensional index
"""
function linear_index(il1, il2, is, L, S)
    il1 <= L && il2 <= L && is <= S || error("Out of range")
    return il1 + L * (il2 - 1) + L^2 * (is - 1)
end
export linear_index

"""
Translates a 1-dimensional index (il1,il2,is) into a 3-dimensional tensor index
"""
function inverse_linear_index(index, L, S)
    is = div(index - 1, L^2) + 1
    remainder = (index - 1) % L^2
    il2 = div(remainder, L) + 1
    il1 = remainder % L + 1
    return il1, il2, is
end
export inverse_linear_index

"""
It constructs γ₅D exactly for a mass `mass`
"""
function construct_gD!(corrws, u1ws::Union{U1Quenched,U1Nf2,U1Nf}, mass::Float64)
    x1 = corrws.e1
    x2 = corrws.e2
    gD = corrws.gD
    x1 .= 0.0
    x2 .= 0.0
    lsize = u1ws.params.iL[1]
    for is in 1:2, il2 in 1:lsize, il1 in 1:lsize
        ilinidx = linear_index(il1, il2, is, lsize, 2)
        x1[ilinidx] = 1.0
        gamm5Dw!(x2, x1, mass, u1ws)
        for js in 1:2, jl2 in 1:lsize, jl1 in 1:lsize
            jlinidx = linear_index(jl1, jl2, js, lsize, 2)
            gD[jlinidx,ilinidx] = x2[jlinidx]
        end
        x1[ilinidx] = 0.0
    end
end
export construct_gD!

"""
It constructs D exactly for a mass `mass`
"""
function construct_D!(corrws, u1ws::Union{U1Quenched,U1Nf2,U1Nf}, mass::Float64)
    x1 = corrws.e1
    x2 = corrws.e2
    gD = corrws.gD
    x1 .= 0.0
    x2 .= 0.0
    lsize = u1ws.params.iL[1]
    for is in 1:2, il2 in 1:lsize, il1 in 1:lsize
        ilinidx = linear_index(il1, il2, is, lsize, 2)
        x1[ilinidx] = 1.0
        Dw!(x2, x1, mass, u1ws)
        for js in 1:2, jl2 in 1:lsize, jl1 in 1:lsize
            jlinidx = linear_index(jl1, jl2, js, lsize, 2)
            gD[jlinidx,ilinidx] = x2[jlinidx]
        end
        x1[ilinidx] = 0.0
    end
end
export construct_D!

"""
Builds inverse of γ₅D using LinearAlgebra.inv for flavor `ifl`
"""
function get_invgD!(corrws, u1ws, ifl)
    corrws.invgD[ifl] .= inv(corrws.gD)
    return nothing
end
export get_invgD!


"""
Constructs the inverse of γ₅D for flavor `ifl`, storing it in `corrws.invgD[ifl]`
"""
function construct_invgD!(corrws, u1ws::U1Nf, ifl)
    construct_gD!(corrws, u1ws, u1ws.params.am0[ifl])
    get_invgD!(corrws, u1ws, ifl)
    return nothing
end

"""
Constructs the inverse of γ₅D for a mass `mass`, storing it in `corrws.invgD[ifl]`. Inteded for use with U1Quenched only
"""
function construct_invgD!(corrws, u1ws::U1Quenched, ifl, mass)
    construct_gD!(corrws, u1ws, mass)
    get_invgD!(corrws, u1ws, ifl)
    return nothing
end

"""
Constructs the inverse of γ₅D storing it in `corrws.invgD[ifl]`. Inteded for use with U1Nf2
"""
function construct_invgD!(corrws, u1ws::U1Nf2, ifl)
    construct_gD!(corrws, u1ws, u1ws.params.am0)
    get_invgD!(corrws, u1ws, ifl)
    return nothing
end

"""
Constructs the inverse of γ₅D for flavor `ifl`
"""
function construct_invgD!(corrws, u1ws::U1Nf)
    for ifl in 1:2
        construct_gD!(corrws, u1ws, u1ws.params.am0[ifl])
        get_invgD!(corrws, u1ws, ifl)
    end
    return nothing
end
export construct_invgD!

"""
Constructs the inverse of γ₅D for 2 flavors with mass `mass`. Inteded for use with U1Quenched only
"""
function construct_invgD!(corrws, u1ws::U1Quenched, mass)
    for ifl in 1:2
        construct_gD!(corrws, u1ws, mass)
        get_invgD!(corrws, u1ws, ifl)
    end
    return nothing
end
export construct_invgD!

"""
Constructs the inverse of γ₅D for 2 flavors. Inteded for use with U1Nf2 only
"""
function construct_invgD!(corrws, u1ws::U1Nf2)
    for ifl in 1:2
        construct_gD!(corrws, u1ws, u1ws.params.am0)
        get_invgD!(corrws, u1ws, ifl)
    end
    return nothing
end
export construct_invgD!

"""
Computes 1/N0 ∑_n,m tr[γ₅D⁻¹_ifl(n,t|m,t₀) γ₅D⁻¹_jfl(m,t₀|n,t)] with source at
time slice `t` and sink at time slice `t0`. (tr is over spin)
"""
function ex_connected_correlator_t0(corrws, t, t0, u1ws, ifl, jfl)
    gDinv1 = corrws.invgD[ifl]
    gDinv2 = corrws.invgD[jfl]
    lsize = u1ws.params.iL[1]
    Ct = 0.0
    for (is1, is2, ilm, iln) in Iterators.product(1:2, 1:2, 1:lsize, 1:lsize)
        Ct += gDinv1[linear_index(iln, t, is1, lsize, 2), linear_index(ilm, t0, is2, lsize, 2)] * gDinv2[linear_index(ilm, t0, is2, lsize, 2),linear_index(iln, t, is1, lsize, 2)]/lsize
    end
    return Ct
end
export ex_connected_correlator_t0

function ex_connected_correlator(corrws, u1ws, t0, ifl, jfl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = ex_connected_correlator_t0(corrws, t, t0, u1ws, ifl, jfl) |> real
    end
end
export ex_connected_correlator


"""
Computes 1/√N0 ∑_n tr[γ₅D⁻¹_ifl(n,t|n,t)] at time slice `t`. (tr is over spin)
"""
function ex_disconnected_correlator_t0(corrws::U1exCorrelator, t, u1ws, ifl)
    gDinv = corrws.invgD[ifl]
    lsize = u1ws.params.iL[1]
    Ct = 0.0
    for (is, iln) in Iterators.product(1:2, 1:lsize)
        Ct += gDinv[linear_index(iln, t, is, lsize, 2), linear_index(iln, t, is, lsize, 2)] / sqrt(lsize)
    end
    return Ct
end
export ex_disconnected_correlator_t0

function ex_disconnected_correlator(corrws, u1ws, ifl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = ex_disconnected_correlator_t0(corrws, t, u1ws, ifl) |> real
    end
end
export ex_disconnected_correlator

"""
Computes all two-to-two correlation functions
"""

function getId(combs, p1, p2)
    for i in 1:size(combs,2)
        if p1 == combs[1,i] && p2 == combs[2,i]
            return i
        end
    end
end
export getId

function multiplyPhase(Mmatrix, phase, L)
    id = 0

    res = zeros(complex(Float64), 2*L, 2*L)
    for y in 1:L
        res[1:L, y] =  phase .* Mmatrix[1:L, y]
        res[L+1:2*L, y] = phase .* Mmatrix[L+1:2*L, y]
        res[1:L, L+y] = phase .* Mmatrix[1:L, L+y]
        res[L+1:2*L, L+y] = phase .* Mmatrix[L+1:2*L, L+y]
    end
    return res
end
export multiplyPhase

function all_correlators1(correlator, g5_correlator, MR, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)
    T = u1ws.params.iL[2]
    L = u1ws.params.iL[1]

    for p in -abspmax:abspmax
        Threads.@threads for t in 1:T
            data.Delta[abspmax+p+1,t] = LinearAlgebra.tr(correlator[p+1+abspmax,t,t,:,:])
            data.Deltas[abspmax+p+1,t] = LinearAlgebra.tr(g5_correlator[p+1+abspmax,t,t,:,:])
        end
    end

    for p in 0:abspmax
        Threads.@threads for t in 0:T-1
            for t0 in 1:T
                data.P[p+1, t+1] += sum(transpose(correlator[1+abspmax-p, t0, 1+(t+t0-1) % T, :, :]) .* correlator[1+abspmax+p, 1+(t+t0-1) % T, t0, :, :]) / T
                data.Ps[p+1, t+1] += sum(transpose(g5_correlator[1+abspmax-p, t0, 1+(t+t0-1) % T, :, :]) .* g5_correlator[1+abspmax+p, 1+(t+t0-1) % T, t0, :, :]) / T
                data.disc[p+1, t+1] += data.Delta[abspmax-p+1,t0] * data.Delta[abspmax+p+1,1+(t+t0-1) % T] / T
                data.discs[p+1, t+1] += data.Deltas[abspmax-p+1,t0] * data.Deltas[abspmax+p+1,1+(t+t0-1) % T] / T
            end
        end
    end

    Threads.@threads for P in 0:Pmax
        for ini in 1:Onum
            q1 = mom_comb[P+1,ini,1]
            q2 = mom_comb[P+1,ini,2]
            for t in 1:T
                data.Vini[P+1, ini,t] = LinearAlgebra.tr(multiplyPhase(MR[1+abspmax-q2,t,t,:,:],phases[abspmax+1-q1],L))
                data.Vfin[P+1, ini,t] = LinearAlgebra.tr(multiplyPhase(MR[1+abspmax+q2,t,t,:,:],phases[abspmax+1+q1],L))
            end
        end
    end

    Threads.@threads for P in 0:Pmax
        id = 1
        for ini in 1:Onum
            q1 = mom_comb[P+1,ini,1]
            q2 = mom_comb[P+1,ini,2] #La simetrización en momento inicial para los triángulosno está implementada bien. Se puede reusar cambiando el significado de los tiempos.
            for fin in 1:Onum
                p1 = mom_comb[P+1,fin,1]
                p2 = mom_comb[P+1,fin,2]
                for t in 0:T-1
                    for t0 in 1:T
                        data.VV[P+1,Onum*(ini-1)+fin, t+1] += data.Vini[P+1, ini, t0] * data.Vfin[P+1, fin, 1+(t+t0-1) % T] / T
                        data.R[P+1, id, t+1] += sum(transpose(multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) .* multiplyPhase(MR[1+abspmax+p2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+p1],L)) / T
                    end
                end
                id += 1

                if p1 == p2
                    continue
                end

                p2 = mom_comb[P+1,fin,1]
                p1 = mom_comb[P+1,fin,2]
                for t in 0:T-1
                    for t0 in 1:T
                        data.R[P+1, id, t+1] += sum(transpose(multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) .* multiplyPhase(MR[1+abspmax+p2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+p1],L)) / T
                    end
                end
                id += 1
            end
        end

        id = 1
        for op in 1:Onum
            q1 = mom_comb[P+1,op,1]
            q2 = mom_comb[P+1,op,2]

            for t in 0:T-1
                for t0 in 1:T
                    data.Tsini[P+1, id, t+1] += sum(transpose(g5_correlator[1+abspmax-P, t0, 1+(t+t0-1) % T, :, :]) .* multiplyPhase(MR[1+abspmax+q2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+q1],L)) / T
                    data.Tsinidis[P+1, op, t+1] += data.Vfin[P+1, op, 1+(t+t0-1) % T] * data.Deltas[abspmax-P+1,t0] / T
                    data.Tsfin[P+1, id, t+1] += sum(transpose(g5_correlator[1+abspmax+P, 1+(t+t0-1) % T, t0, :, :]) .* multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) / T
                    data.Tsfindis[P+1, op, t+1] += data.Vini[P+1, op, t0] * data.Deltas[abspmax+P+1,1+(t+t0-1) % T] / T
                end
            end
            id += 1


            if q1 == q2
                continue
            end

            q2 = mom_comb[P+1,op,1]
            q1 = mom_comb[P+1,op,2]

            for t in 0:T-1
                for t0 in 1:T
                    data.Tsini[P+1, id, t+1] += sum(transpose(g5_correlator[1+abspmax-P, t0, 1+(t+t0-1) % T, :, :]) .* multiplyPhase(MR[1+abspmax+q2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+q1],L)) / T
                    data.Tsfin[P+1, id, t+1] += sum(transpose(g5_correlator[1+abspmax+P, 1+(t+t0-1) % T, t0, :, :]) .* multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) / T
                end
            end
            id += 1

        end
    end
    return nothing
end
export all_correlators1

function all_correlators3(correlator, mIg0_correlator, MR, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)
    T = u1ws.params.iL[2]
    L = u1ws.params.iL[1]

    for p in 0:abspmax
        Threads.@threads for t in 0:T-1
            for t0 in 1:T
                data.Pr[p+1, t+1] -= sum(transpose(mIg0_correlator[1+abspmax-p, t0, 1+(t+t0-1) % T, :, :]) .* mIg0_correlator[1+abspmax+p, 1+(t+t0-1) % T, t0, :, :]) / T
            end
        end
    end

    Threads.@threads for P in 0:Pmax
        id = 1
        for op in 1:Onum
            q1 = mom_comb[P+1,op,1]
            q2 = mom_comb[P+1,op,2]

            for t in 0:T-1
                for t0 in 1:T
                    data.Trini[P+1, id, t+1] -= sum(transpose(mIg0_correlator[1+abspmax-P, t0, 1+(t+t0-1) % T, :, :]) .* multiplyPhase(MR[1+abspmax+q2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+q1],L)) / T #The minus sign comes from the dagger operator, as gamma_1 and gamma_0 anticommute
                    data.Trfin[P+1, id, t+1] += sum(transpose(mIg0_correlator[1+abspmax+P, 1+(t+t0-1) % T, t0, :, :]) .* multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) / T
                end
            end
            id += 1


            if q1 == q2
                continue
            end

            q2 = mom_comb[P+1,op,1]
            q1 = mom_comb[P+1,op,2]

            for t in 0:T-1
                for t0 in 1:T
                    data.Trini[P+1, id, t+1] -= sum(transpose(mIg0_correlator[1+abspmax-P, t0, 1+(t+t0-1) % T, :, :]) .* multiplyPhase(MR[1+abspmax+q2,1+(t+t0-1) % T,t0,:,:],phases[abspmax+1+q1],L)) / T
                    data.Trfin[P+1, id, t+1] += sum(transpose(mIg0_correlator[1+abspmax+P, 1+(t+t0-1) % T, t0, :, :]) .* multiplyPhase(MR[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1-q1],L)) / T
                end
            end
            id += 1

        end
    end
    return nothing
end
export all_correlators3

function all_correlators2(MC, data, u1ws, Pmax, Onum, abspmax, mom_comb, phases)
    T = u1ws.params.iL[2]
    L = u1ws.params.iL[1]

    Threads.@threads for P in 0:Pmax
        id = 1
        for ini in 1:Onum
            q1 = mom_comb[P+1,ini,1]
            q2 = mom_comb[P+1,ini,2] #La simetrización en momento inicial para los triángulosno está implementada bien. Se puede reusar cambiando el significado de los tiempos.
            for fin in 1:Onum
                p1 = mom_comb[P+1,fin,1]
                p2 = mom_comb[P+1,fin,2]
                for t in 0:T-1
                    for t0 in 1:T
                        data.D[P+1,id, t+1] += LinearAlgebra.tr(multiplyPhase(MC[1+abspmax-q1,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p1],L)) * LinearAlgebra.tr(multiplyPhase(MC[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p2],L)) / T
                        data.C[P+1, id, t+1] += sum(transpose(multiplyPhase(MC[1+abspmax-q1,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p1],L)) .* multiplyPhase(MC[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p2],L)) / T
                    end
                end
                id += 1
                if p1 == p2
                    continue
                end

                p2 = mom_comb[P+1,fin,1]
                p1 = mom_comb[P+1,fin,2]
                Threads.@threads for t in 0:T-1
                    for t0 in 1:T
                        data.D[P+1,id, t+1] += LinearAlgebra.tr(multiplyPhase(MC[1+abspmax-q1,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p1],L)) * LinearAlgebra.tr(multiplyPhase(MC[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p2],L)) / T
                        data.C[P+1, id, t+1] += sum(transpose(multiplyPhase(MC[1+abspmax-q1,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p1],L)) .* multiplyPhase(MC[1+abspmax-q2,t0,1+(t+t0-1) % T,:,:],phases[abspmax+1+p2],L)) / T
                    end
                end
                id += 1
            end
        end
    end
    return nothing
end
export all_correlators2
