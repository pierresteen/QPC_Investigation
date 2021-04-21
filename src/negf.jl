module NEGF

using LinearAlgebra

include("system.jl")
using .System
include("types.jl")
using .Types

export system_solve

"""
	pickoutᵢ(λ_values, mode)

Returns an array of indices for eigenvalues in `λ_values` which correspond to:
* `R ->` right-propagating waves
* `L ->` left-propagating waves
* `E ->` evanescent waves
* `G ->` growing evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*
* `D ->` decaying evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*
"""
function pickoutᵢ(λ_values, mode)
	if mode == "R"
		# add the indices which equate to right-propagating eigenvals to $indices
		arrᵢ = findall(!iszero, imag(λ_values) .> 0)
	elseif mode == "L"
		# add the indices which equate to left-propagating eigenvals to $indices
		arrᵢ = findall(!iszero, imag(λ_values) .< 0)
	elseif mode == "E"
		# add the indices which equate to evanescent wave eigenvals to $indices
		arrᵢ = findall(!iszero, imag(λ_values) .== 0)
	elseif mode == "G"
		arrᵢ = findall(!iszero, abs.(λ_values) .> 1)
	elseif mode == "D"
		arrᵢ = findall(!iszero, abs.(λ_values) .< 1)
	end
	
	return Array(arrᵢ)
end

"""
	ψ_classify(λ_vec, λ_val)

Classifies eigenvectors and eigenvalues into categories to form the potential barrier-type problem to set up the NEGF form.
"""
function ψ_classify(λ_vec, λ_val)
	# sort waves by: right-moving, left-moving and evanescent
	Rᵢ = pickoutᵢ(λ_val, "R")
	Lᵢ = pickoutᵢ(λ_val, "L")
	Eᵢ = pickoutᵢ(λ_val, "E")
	
	# index evanescent waves which are growing: $Gᵢ or decaying: $Dᵢ
	Gᵢ = pickoutᵢ(λ_val[Eᵢ], "G")
	Dᵢ = pickoutᵢ(λ_val[Eᵢ], "D")
	
	# index $λ_vec to form ψ and E (evanescent) R-, L-mode & G-, D-mode wave arrays
	# which are a numerical representation of the system's wave functions
	ψ_R = λ_vec[:, Rᵢ]
	ψ_L = λ_vec[:, Lᵢ]
	E_G = λ_vec[:, Eᵢ][:, Gᵢ]
	E_D = λ_vec[:, Eᵢ][:, Dᵢ]
	
	return ψ_R, ψ_L, E_G, E_D
end

"""
	ψ_norms(ψR, ψL)

Calculates and applies norming factors to wavefunctions `ψ_R` and `ψ_L`.
"""
function ψ_norms(N, ψR, ψL)
	# evaluate wave function norms $ψₙ_R & $ψₙ_L
	ψₙ_R = norm(ψR[(N+1):2*N]).^2 - norm(ψR[1:N]).^2
	ψₙ_L = norm(ψL[1:N]).^2 - norm(ψL[(N+1):2*N]).^2
	
	# apply norming factors to wave funtions
	ψ_R = ψR ./ √(abs(ψₙ_R))
	ψ_L = ψL ./ √(abs(ψₙ_L))
	
	return ψ_R, ψ_L
end

"""
	build_Uᵣ(N, Sₜ, ψᵣ)

Builds the right-hand side terms of the NEGF form. 
"""
function build_Uᵣ(N, Sₜ, ψᵣ)
	Uᵣ_top = Sₜ.s_11 * ψᵣ[1:N, :] - ψᵣ[(N+1):(2*N), :]
	Uᵣ_bot = Sₜ.s_21 * ψᵣ[1:N, :]

	return vcat(Uᵣ_top, Uᵣ_bot)
end

"""
	build_Uₗ(N, Sₜ, ψᵣ, ψₗ, grow, decrease)

Builds the left-hand side terms of the NEGF form.
"""
function build_Uₗ(N, Sₜ, ψᵣ, ψₗ, grow, decrease)
	Uₗ_t1 = - Sₜ.s_12 * ψᵣ[(N+1):(2*N), :]
	Uₗ_t2 = grow[(N+1):(2*N), :] - (Sₜ.s_11 * grow[1:N, :])
	Uₗ_t3 = ψₗ[(N+1):(2*N), :] - (Sₜ.s_11 * ψₗ[1:N, :])
	Uₗ_t4 = - Sₜ.s_12 * decrease[(N+1):(2*N), :]

	Uₗ_b1 = ψᵣ[1:N,:] - (Sₜ.s_22 * ψᵣ[(N+1):(2*N), :])
	Uₗ_b2 = - Sₜ.s_21 * grow[1:N, :]
	Uₗ_b3 = - Sₜ.s_21 * ψₗ[1:N,:]
	Uₗ_b4 = decrease[1:N, :] - Sₜ.s_22 * decrease[(N+1):(2*N), :]

	Uₗ_top = hcat(Uₗ_t1, Uₗ_t2, Uₗ_t3, Uₗ_t4)
	Uₗ_bot = hcat(Uₗ_b1, Uₗ_b2, Uₗ_b3, Uₗ_b4)

	return vcat(Uₗ_top, Uₗ_bot)
end

"""
	system_intermediary(Uᵣ, Uₗ, ψ_L, E_D)

As the name implies, the function serves to simplify the numerical algorithm.

```julia
system_intermediary(Uᵣ, Uₗ, ψ_L, E_D) -> coeffs::Array{Complex{Float64},2}, count_in::Int, count_evan::Int
```
"""
function system_intermediary(Uᵣ, Uₗ, ψ_L, E_D)
	coeffs = inv(Uₗ) * Uᵣ		# coefficients matrix
	
	count_in	= size(ψ_L)[2]	# number of incident waves
	count_evan	= size(E_D)[2]	# number of evanescent waves
	
	return coeffs, count_in, count_evan
end

"""
	resolve(coeffs, count_in, count_ev)

Solutions resolver functions, final step in the numerical algorithm.
"""
function resolve(coeffs, count_in, count_ev)
	τ = coeffs[1:(count_in), :]
	α = coeffs[(count_in+1):(count_in+count_ev), :]
	r = coeffs[(count_in+count_ev+1):(2*count_in+count_ev), :]
	β = coeffs[(2*count_in+count_ev+1):(2*count_in+2*count_ev), :]
	
	G = norm(τ)^2 # final coductance calculation
	
	return G, τ, α, r, β
end

"""
	system_solve(μ, V, N, L, i)
	
Algorithm for solving the system of equations of the "Chalker-Coddington Network Model" of a 1D QPC...
"""
function system_solve(μ, V, N, L, i)
	# generate scattering matrices S_T::S_data
	S_T = gen_S_total(V, L)
	
	# extract eigenvectors & eigenvalues from T::T_data.self
	λ = eigen(T(μ, N).self, sortby=nothing, permute=true)
	
	# round eigen-components to 11 decimal places
	λ_vals = round.(λ.values, digits=10)
	λ_vecs = -1 .* round.(λ.vectors, digits=10)

	# sort and index: ψ_R & ψ_L (un-normed) + E_G & E_D (growing and decreasing)
	ψ_R, ψ_L, E_G, E_D = ψ_classify(λ_vecs, λ_vals)

	# evaluate and apply ψ norms
	ψ_R, ψ_L = ψ_norms(N, ψ_R, ψ_L)
	
	# form system of equation right- and left-hand side terms
	Uₗ = build_Uₗ(N, S_T, ψ_R, ψ_L, E_G, E_D)
	Uᵣ = build_Uᵣ(N, S_T, ψ_R)
	
	# evaluate coefficient and store in matrix form
	coeff, num_I, num_E = system_intermediary(Uᵣ, Uₗ, ψ_L, E_D)
	
	# evaluate system solutions
	solution_params = resolve(coeff, num_I, num_E)
	
	return solution_params
end

end
