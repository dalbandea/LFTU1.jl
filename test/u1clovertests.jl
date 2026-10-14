using LinearAlgebra

# Clover (Sheikholeslami-Wohlert) term: γ₅D_sw = γ₅D + (csw/2) F̂(x) on both spin components

# Dense matrix of γ₅D (g5 = true) or D on a small lattice, column by column
function dense_dirac(model, am0; g5 = true)
    L, T = model.params.iL
    V = 2L*T
    M = zeros(ComplexF64, V, V)
    e = zeros(ComplexF64, L, T, 2)
    o = similar(e)
    for k in 1:V
        e .= 0
        e[k] = 1
        g5 ? gamm5Dw!(o, e, am0, model) : Dw!(o, e, am0, model)
        M[:, k] .= vec(o)
    end
    return M
end

# F̂ written independently of the code: the four leaves as explicit link products around x
function clover_field_reference(U, L, T)
    F = zeros(L, T)
    p(i) = mod1(i, L)
    q(j) = mod1(j, T)
    for x in 1:L, y in 1:T
        l1 = U[x,y,1] * U[p(x+1),y,2] * conj(U[x,q(y+1),1]) * conj(U[x,y,2])
        l2 = U[x,y,2] * conj(U[p(x-1),q(y+1),1]) * conj(U[p(x-1),y,2]) * U[p(x-1),y,1]
        l3 = conj(U[p(x-1),y,1]) * conj(U[p(x-1),q(y-1),2]) * U[p(x-1),q(y-1),1] * U[x,q(y-1),2]
        l4 = conj(U[x,q(y-1),2]) * U[x,q(y-1),1] * U[p(x+1),q(y-1),2] * conj(U[x,y,1])
        F[x,y] = imag(l1 + l2 + l3 + l4) / 4
    end
    return F
end

# Constant field strength with topological charge Q on an L×L torus: every plaquette angle 2πQ/L²
function constant_field!(model, Q)
    L = model.params.iL[1]
    θ = 2π*Q/L^2
    for x in 1:L, y in 1:L
        model.U[x,y,1] = cis(-θ*(y-1))
        model.U[x,y,2] = y == L ? cis(θ*L*(x-1)) : one(ComplexF64)
    end
    return θ
end

let
    L, T = 6, 4
    csw = 1.3
    wilson = U1Nf2(Float64, iL = (L, T), beta = 2.0, am0 = 0.1, BC = PeriodicBC)
    clover = U1Nf2(Float64, iL = (L, T), beta = 2.0, am0 = 0.1, BC = PeriodicBC, csw = csw)
    LFTU1.randomize!(wilson)
    clover.U .= wilson.U
    Mw = dense_dirac(wilson, 0.1)
    Mc = dense_dirac(clover, 0.1)

    @testset "Clover γ₅D" begin
        @test norm(Mc - Mc') < 1e-13
        Δ = Mc - Mw
        @test norm(Δ - Diagonal(diag(Δ))) < 1e-13
        d = reshape(real(diag(Δ)), L, T, 2)
        F = clover_field_reference(wilson.U, L, T)
        @test maximum(abs, d[:,:,1] .- csw/2 .* F) < 1e-13
        @test maximum(abs, d[:,:,2] .- csw/2 .* F) < 1e-13
        LFTU1.random_gauge_trafo(clover)
        ev(M) = sort(eigvals(Hermitian(M)))
        @test norm(ev(dense_dirac(clover, 0.1)) - ev(Mc)) < 1e-11
    end

    # In a constant field, D(m=0) has |Q| real modes at the critical mass. Wilson: aF/2 = θ/2
    # at tree level; csw = 1 removes the O(a) term, leaving O(θ²)
    @testset "Clover O(a) improvement in a constant field" begin
        for L in (8, 12), Q in (1, 2)
            λ = Dict{Float64,Float64}()
            for csw in (0.0, 1.0)
                m = U1Nf2(Float64, iL = (L, L), beta = 1.0, am0 = 0.0, BC = PeriodicBC, csw = csw)
                θ = constant_field!(m, Q)
                @test isapprox(top_charge(m), Q)
                ev = eigvals(dense_dirac(m, 0.0; g5 = false))
                r = sort(real.(ev[abs.(imag.(ev)) .< 1e-8]))
                λ[csw] = r[1]
                @test isapprox(r[Q], r[1], atol = 1e-10)
            end
            θ = 2π*Q/L^2
            @test isapprox(λ[0.0], θ/2, rtol = 0.05)
            @test abs(λ[1.0]) < 0.1*θ^2
        end
    end

    @testset "Clover HMC force" begin
        for model in (U1Nf2(Float64, iL = (8, 8), beta = 5.0, am0 = 0.6, BC = PeriodicBC, csw = 1.0),
                      U1Nf(Float64, iL = (8, 8), beta = 5.0, am0 = [0.6, 0.3], BC = PeriodicBC,
                           csw = 1.0, ns_rat = [3, 5], r_as = [sqrt(0.45), sqrt(0.45)],
                           r_bs = [sqrt(22), sqrt(22)]))
            hmcws = LFTU1.sampler(model, HMC(integrator = Leapfrog(1.0, 10)))
            LFTU1.randomize!(model)
            LFTU1.generate_pseudofermions!(model, hmcws)
            ΔF = LFTSampling.force_test(model, hmcws, 1e-7)
            @test isapprox(zero(model.PRC), ΔF, atol = 1e-5)
        end
    end
end
