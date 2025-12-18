
k(eps) = sqrt(1.0-eps)
v(n, eps) = Elliptic.K(k(eps)^2)/(2.0*n+1.0)
a_r(r, n, eps) = Jacobi.cn(r*v(n,eps), k(eps)^2)^2/Jacobi.sn(r*v(n,eps), k(eps)^2)^2
c_r(r, n, eps) = Jacobi.sn(r*v(n,eps), k(eps)^2)^2


mu(j, n, eps, r_b) = r_b * a_r(2*j, n, eps)^(1/2)
nu(j, n, eps, r_b) = r_b * a_r(2*j-1, n, eps)^(1/2)

function rho_mu(j, k, l, n, eps, r_b)       # Lüscher eq. (3.5)
	if(j<k || j>l)
		throw("j is not between k and l")
	end

	res = ( nu(j, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2 )

	for m in k:l
		if m!=j
			res *= (nu(m, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2)/(mu(m, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2)
		end
	end

	return res
end

function rho_nu(j, k, l, n, eps, r_b)       # Lüscher eq. (3.9)
	if(j<k || j>l)
		throw("j is not between k and l")
	end

	res = ( mu(j, n, eps, r_b) - nu(j, n, eps, r_b) )

	for m in k:l
		if m!=j
			res *= (mu(m, n, eps, r_b) - nu(j, n, eps, r_b))/(nu(m, n, eps, r_b) - nu(j, n, eps, r_b))
		end
	end

	return res
end

function P(k, l, n, eps, r_b, Y)            # Lüscher eq. (3.4), not used in code
	dim = size(Y,2) # get dimensions of matrix
	res = LinearAlgebra.I(dim)

	# using LinearAlgebra.I(dim) makes it compute correctly when Y is just a
	# float number
	if(k!=0 && l!=0)
		for j in k:l
			res = res .+ rho_mu(j, k, l, n, eps, r_b)*(Y .+ mu(j, n, eps, r_b)^2*LinearAlgebra.I(dim) )^(-1)
		end
	end
	println(res)
	
	return res
end


function d(n, eps)
	res = k(eps)^(2*n+1)

	for i in 1:2:2*n+1
		res *= c_r(i, n, eps)^2
	end

	return res
end

delta(n, eps) = d(n, eps)^2 /(1+sqrt(1-d(n,eps)^2))^2

function A(n, eps)
	res = 2/(1+sqrt(1-d(n,eps)^2))

	for i in 1:2:2*n-1
		res *= c_r(i,n,eps)
	end

	for i in 2:2:2*n
		res *= 1/c_r(i,n,eps)
	end

	return(res)
end


function error(Y, Yapprox)
	return abs(1 .- sqrt(Y)*Yapprox[1])
end



function get_rhmc_params(n_rhmc::Int64, r_a_rhmc, r_b_rhmc; reweighting_N::Int64=1, reweighting_Taylor::Int64=5)

    eps_rhmc = ( r_a_rhmc/r_b_rhmc )^2
    mu_rhmc = Array{Float64}(undef, n_rhmc)
    nu_rhmc = Array{Float64}(undef, n_rhmc)
    rho_rhmc = Array{Float64}(undef, n_rhmc)
    A_rhmc = A(n_rhmc,eps_rhmc)
    delta_rhmc = delta(n_rhmc, eps_rhmc)

    for j in 1:n_rhmc
        mu_rhmc[j] = mu(j,n_rhmc,eps_rhmc, r_b_rhmc)
        nu_rhmc[j] = nu(j,n_rhmc,eps_rhmc, r_b_rhmc)
        rho_rhmc[j] = rho_mu(j,1,n_rhmc,n_rhmc,eps_rhmc, r_b_rhmc)
    end
    rprm = RHMCParm(r_b_rhmc, n_rhmc, eps_rhmc, A_rhmc, rho_rhmc, mu_rhmc, nu_rhmc, delta_rhmc, reweighting_N, reweighting_Taylor)

    return rprm
end

function get_rhmc_params(n_rhmc::Array, r_a_rhmc::Array, r_b_rhmc::Array; reweighting_N::Int64=1, reweighting_Taylor::Int64=4)

    rprms = Array{RHMCParm}(undef,length(n_rhmc))

    for j in 1:length(n_rhmc)
        rprms[j] = get_rhmc_params(n_rhmc[j], r_a_rhmc[j], r_b_rhmc[j])
    end

    return rprms
end


"""
    power_method(U, am0)

Given a gauge field `U` and a bare quark mass `am0`, return the maximum and
minimum eigenvalues of D^†D with 1000 iterations of the power method.

# Examples
```jldocs
lambda_min, lambda_max = power_method(U, am0)
```
"""

function power_method(U1ws::U1, am0; iter::Int64 = 1000)

    lp = U1ws.params

    b = to_device(U1ws.device, Random.randn(ComplexF64, lp.iL[1], lp.iL[2], 2)) # initial random fermionic field
    tmp = similar(b)

    shift = 2     # this shift helps convergence

    # Apply recursively b = Ab.
    # To help convergence, apply instead b = (A + shift) b = Ab + shift b.
    # Then λ_max will be ⟨b|A|b⟩/⟨b|b⟩ - shift.
    for i in 1:iter
        b_aux = copy(b)
        gamm5Dw_sqr_msq!(b_aux, tmp, b, am0, U1ws)
        b_aux .= b_aux .+ shift*b
        b = b_aux/sqrt(LinearAlgebra.dot(b_aux,b_aux))
    end
    bnext = copy(b)
    gamm5Dw_sqr_msq!(bnext, tmp, b, am0, U1ws)
    bnext .= bnext .+ shift*b
    lambda_max = LinearAlgebra.dot(b,bnext)/LinearAlgebra.dot(b,b) - shift

    println("TOL = ", LinearAlgebra.norm(bnext .- lambda_max*b - shift*b)/LinearAlgebra.norm(lambda_max*b))

    # Apply recursively b = (A-λ_max I) b = Ab - λ_max b
    # Then λ_min will be ⟨b|A|b⟩/⟨b|b⟩ + λ_max
    for i in 1:iter
        b_last = copy(b)
        gamm5Dw_sqr_msq!(b, tmp, b_last, am0, U1ws)
        b .= b .- lambda_max*b_last
        b = b/sqrt(LinearAlgebra.dot(b,b))
    end
    bnext = copy(b)
    gamm5Dw_sqr_msq!(bnext, tmp, b, am0, U1ws)
    bnext .= bnext .- lambda_max*b
    lambda_min = LinearAlgebra.dot(b,bnext)/LinearAlgebra.dot(b,b) + lambda_max

    return lambda_min, lambda_max

end


# Returns Z X_in, with Z = D†DR² - I
function LuscherZ(Z, X_in, am0, rprm::RHMCParm, u1ws::U1Nf)
    # Z = D†DR² X_in
    LFTU1.R(Z, X_in, am0, rprm, u1ws)
    tmp = copy(Z)
    LFTU1.R(Z, tmp, am0, rprm, u1ws)
    tmp .= Z
    gamm5Dw!(Z, tmp, am0, u1ws)
    tmp .= Z
    gamm5Dw!(Z, tmp, am0, u1ws)
    # Z = Z - X_in = (D†DR² - I) X_in
    Z .= Z .- X_in
    return nothing
end

# # Returns ZᵖX_in, with Z = D†DR² - I
# function LuscherZp(Z, p::Int64, U, X_in, am0, CGmaxiter, CGtol, rprm::RHMCParm, prm::LattParm, kprm::KernelParm)


#     tmp = copy(X_in)
#     # At the end, Z = (D†DR² - I)ᵖ X_in
#     for i in 1:p
#         LuscherZ(Z, U, tmp, am0, CGmaxiter, CGtol, rprm, prm, kprm)
#         tmp .= Z
#     end

#     return nothing

# end


# Compute reweighting factor W_N with N=1, eq. (4.1)
"""
    reweighting_factor(U, am0, prm::LattParm, kprm::KernelParm, rprm::RHMCParm)

Computes the reweighting factor ``W_N`` stochastically using `N` random normal fields up to order `rprm.reweighting_Taylor` in the Taylor expansion of ``(1-Z)^{-1/2}``. If `am0` and `rprm` are lists, returns product of ``W_N`` of all fermions.
"""
function reweighting_factor(am0, rprm::RHMCParm, u1ws::U1Nf)
    W_N = 0.0

    # estimate W_N with rprm.reweighting_N pseudofermions
    for j in 1:rprm.reweighting_N
        argexp = 0.0    # contains the argument of the exponentials of W_N for each j
        Tfactor = 1/2   # Taylor factor of expansion (1+Z)^(-1/2)
        X = to_device(u1ws.device, randn(complex(u1ws.PRC), u1ws.params.iL[1], u1ws.params.iL[2], 2))
        ZpX = similar(X)
        tmp = copy(X)
        for i in 1:rprm.reweighting_Taylor
            LFTU1.LuscherZ(ZpX, tmp, am0, rprm, u1ws)
            argexp += Tfactor * dot(X,ZpX) |> real
            Tfactor *= (-1)*(2*i+1)/2/factorial(i+1)*factorial(i)
            tmp .= ZpX
        end
        W_N += exp(argexp)/rprm.reweighting_N
    end

    @debug begin
        two_N_delta = 2*u1ws.params.iL[1]*u1ws.params.iL[2]*u1ws.rprm.delta
        # pritnln("W_$(rprm.reweighting_N) = $(W_N)")

        "if 2Nδ = $(two_N_delta) ≤ 0.01, W₁ is expected to deviate from 1 at most by 1%"
    end
    
    return W_N # if 2Nδ≤0.01, W₁ is expected to deviate from 1 at most by 1%
end

function reweighting_factor(u1ws::U1Nf)
    N_fermions = length(u1ws.params.am0)

    total_W_N = 1.0 # to return. will contain product of W_N of all fermions
    for j in 1:N_fermions
        total_W_N *= reweighting_factor(u1ws.params.am0[j], u1ws.rprm[j], u1ws)
    end

    return total_W_N
end



"""
    R(so, U, si, am0, maxiter, eps, A, rprm, prm, kprm)

Applies rational approximation ``R(A) = A^{-1/2}`` to the field `si`, storing it
in `so`, with `A` an operator.
"""
function R(so, si, am0, rprm::RHMCParm, U1ws::U1)
	aux = similar(so)
	so .= si  # this accounts for identity in partial fraction descomposition
	for j in 1:rprm.n
        iter = invert!(aux, gamm5Dw_sqr_musq_am0_mu!(am0, rprm.mu[j]), si, U1ws.sws, U1ws)
		so .= so .+ rprm.rho[j]*aux
	end
	so .= rprm.r_b^(-1) * rprm.A * so

    return nothing
end

