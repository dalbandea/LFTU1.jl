import LFTSampling: action

KernelAbstractions.@kernel function U1plaquette!(plx, U, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition

    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1
    
    plx[i1,i2] = cos(-im*log(U[i1,i2,1] *
                             U[iu1,i2,2] *
                             conj(U[i1,iu2,1]) *
                             conj(U[i1,i2,2])))

    if BC == OpenBC
        if i1 == Nx || i2 == Ny
            plx[i1, i2] = zero(eltype(plx))
        end
    end
    
end

KernelAbstractions.@kernel function U1qtop!(plx, U, Nx, Ny)

    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1
    
    plx[i1,i2] = angle(U[i1,i2,1] *
                            U[iu1,i2,2] *
                            conj(U[i1,iu2,1] *
                                 U[i1,i2,2]))
end


function force!(U1ws::U1Quenched, hmcws::AbstractHMC)
    return gauge_force!(U1ws, hmcws)
end

function gauge_force!(U1ws::U1, hmcws::AbstractHMC)
    lp = U1ws.params
    event = U1quenchedforce!(U1ws.device)(hmcws.frc1, hmcws.frc2, U1ws.U, lp.beta, lp.iL[1], lp.iL[2], lp.BC, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    wait(event)
    return nothing
end

KernelAbstractions.@kernel function U1quenchedforce!(frc1, frc2, U, beta, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition
    
    i1, i2 = @index(Global, NTuple)

    iu1 = mod(i1, Nx) + 1
    iu2 = mod(i2, Ny) + 1


    v = beta * imag(U[i1,i2,1] * U[iu1,i2,2] * conj(U[i1,iu2,1] * U[i1,i2,2]))

    if BC == OpenBC
        if i1 == Nx || i2 == Ny
            v = zero(eltype(v))
        end
    end

    frc1[i1,i2,1]  = -v 
    frc1[i1,i2,2]  =  v 
    frc2[iu1,i2,2] = -v 
    frc2[i1,iu2,1] =  v 

end

action(U1ws::U1Quenched, hmcws::AbstractHMC) = gauge_action(U1ws)
action(U1ws::U1Quenched) = gauge_action(U1ws)

gauge_action(U1ws::U1) = U1quenchedaction(U1ws)

function U1quenchedaction(U1ws::U1)
    lp = U1ws.params
    return U1quenchedaction(U1ws.U, lp.beta, lp.iL[1], lp.iL[2], lp.BC, U1ws.device, U1ws.kprm.threads, U1ws.kprm.blocks)
end

function U1quenchedaction(U, beta, Nx, Ny, BC, device, threads, blocks)
    plaquettes = to_device(device, zeros(eltype(U), Nx, Ny))
    return U1quenchedaction(plaquettes, U, beta, Nx, Ny, BC, device, threads, blocks)
end

function U1quenchedaction(plaquettes, U, beta, Nx, Ny, BC, device, threads, blocks)
    event = U1plaquette!(device)(plaquettes, U, Nx, Ny, BC, ndrange=(Nx, Ny), workgroupsize=threads)
    wait(event)

    if BC == OpenBC
        Nx = Nx - 1
        Ny = Ny - 1
    end
    S = beta * ( Nx * Ny - reduce(+, plaquettes) )

    return S
end

function top_charge(U1ws::U1)
    lp = U1ws.params
    return U1topcharge(U1ws.U, lp.beta, lp.iL[1], lp.iL[2], U1ws.device, U1ws.kprm.threads, U1ws.kprm.blocks)
end

function U1topcharge(U, beta, Nx, Ny, device, threads, blocks)
    plaquettes = to_device(device, zeros(Float64, Nx, Ny))
    return U1topcharge(plaquettes, U, beta, Nx, Ny, device, threads, blocks)
end

function U1topcharge(plaquettes, U, beta, Nx, Ny, device, threads, blocks)
    event = U1qtop!(device)(plaquettes, U, Nx, Ny, ndrange=(Nx, Ny), workgroupsize=threads)
    wait(event)
    Q = reduce(+, plaquettes) / (2.0*pi)
    return Q
end


# ======================= #
# ====== U1 Nf = 2 ====== #
# ======================= #


action(U1ws::U1Nf2, hmcws::AbstractHMC) = gauge_action(U1ws) + pfaction(U1ws, hmcws)

function pfaction(U1ws::U1Nf2, hmcws::AbstractHMC)
    invert!(hmcws.X, gamm5Dw_sqr_msq!, hmcws.F, U1ws.sws, U1ws)
    return real(LinearAlgebra.dot(hmcws.X, hmcws.F))
end

function force!(U1ws::U1Nf2, hmcws::AbstractHMC)
	# Solve DX = F for X
    iter = invert!(hmcws.X, gamm5Dw_sqr_msq!, hmcws.F, U1ws.sws, U1ws)

	# Apply gamm5D to X
    gamm5Dw!(hmcws.g5DX, hmcws.X, U1ws)
	
	# Get fermion part of the force in U1ws.pfrc
    pf_force!(U1ws, hmcws)

	# Get gauge part of the force in U1ws.frc1 and U1ws.frc2
    gauge_force!(U1ws, hmcws)

    return nothing
end
