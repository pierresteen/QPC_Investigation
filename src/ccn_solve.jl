# Chalker-Coddington Network Model Numerical Solver:
module ccnsolve

"""
	pickoutᵢ(λ_values, mode)

Returns an array of indices for eigenvalues in `λ_values` which correspond to:
- `R ->` right-propagating waves
- `L ->` left-propagating waves
- `E ->` evanescent waves
- `G ->` growing evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*
- `D ->` decaying evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*
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
	system_solve(μ, V, L, i, opt)
	
Algorithm for solving the system of equations of the "Chalker-Coddington Network Model" of a 1D QPC...
"""
function system_solve(μ, V, N, L, i, opt)
	# generate scattering matrices S_T::S_data, according to option 'opt'
	if opt == true
		S_T = gen_S_total_opt(V, L)
	else
		S_T = gen_S_total(V, L)
	end
	
	# extract eigenvectors & eigenvalues from T::T_data.self
	λ = eigen(T(μ, N).self, sortby=nothing)
	# round eigen-components to 11 decimal places
	λ_vals = round.(λ.values, digits=11)	
	λ_vecs = round.(λ.vectors, digits=11)
	
	# extract indices from:
	# 	forward & backward propagating waves
	# 	evanescent growing & decaying waves
	Rᵢ = pickoutᵢ(λ_vals, "R")
	Lᵢ = pickoutᵢ(λ_vals, "L")
	Eᵢ = pickoutᵢ(λ_vals, "E")
	
	# index evanescent waves which are growing: $Gᵢ or decaying: $Dᵢ
	Gᵢ = pickoutᵢ(λ_vals[Eᵢ], "G")
	Dᵢ = pickoutᵢ(λ_vals[Eᵢ], "D")
	
	# index $λ_vec to form ψ and E (evanescent) R-, L-mode & G-, D-mode wave arrays
	# which are a numerical representation of the system's wave functions
	ψ_R = λ_vecs[:, Rᵢ]
	ψ_L = λ_vecs[:, Lᵢ]
	E_G = λ_vecs[:, Eᵢ][:, Gᵢ]
	E_D = λ_vecs[:, Eᵢ][:, Dᵢ]
	
	# evaluate wave function norms $ψₙ_R & $ψₙ_L
	ψₙ_R = norm(ψ_R[(N+1):2*N]).^2 - norm(ψ_R[1:N]).^2
	ψₙ_L = norm(ψ_L[1:N]).^2 - norm(ψ_L[(N+1):2*N]).^2
	
	# apply norming factors to wave funtions
	ψ_R = ψ_R ./ √(abs(ψₙ_R))
	ψ_L = ψ_L ./ √(abs(ψₙ_L))
	#-- passes until here!!
	
	## formulate system of equations with grouped wave terms: ##
	
	# $Uₗ_top, create & append to fill 4N sized array
	lt1 = -S_T.s_12 * ψ_R[(N+1):(2*N),:]
	lt2 = E_G[(N+1):(2*N),:] - (S_T.s_11 * E_G[1:N,:])
	lt3 = ψ_L[(N+1):(2*N),:] - (S_T.s_11 * ψ_L[1:N,:])
	lt4 = -S_T.s_12 * E_D[(N+1):(2*N),:]
	Uₗ_top = hcat(lt1, lt2, lt3, lt4)
	#-- passes but Uₗ_top ≠ python(eqiuv. Uₗ_top)

	# $Uₗ_bot, create & append to fill 4N sized array
	Uₗ_bot = ψ_R[1:N,:] - (S_T.s_22 * ψ_R[(N+1):(2*N),:])
	Uₗ_bot = cat(Uₗ_bot, -S_T.s_21 * E_G[1:N,:], dims=2)
	Uₗ_bot = cat(Uₗ_bot, -S_T.s_21 * ψ_L[1:N,:], dims=2)
	Uₗ_bot = cat(Uₗ_bot, E_D[1:N,:] - S_T.s_22 * E_D[(N+1):(2*N),:], dims=2)
	#-- passes until here
	
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ, the total eq.-system matrix
	Uₗ = zeros(Complex{Float64}, 2*N, 2*N)
	Uₗ[1:N,:] 		  = Uₗ_top
	Uₗ[(N+1):(2*N),:] = Uₗ_bot
	#-- passses until here

	# $Uᵣ_top & $Uᵣ_bot create 4N sized arrays
	Uᵣ_top = S_T.s_11 * ψ_R[1:N,:] - ψ_R[(N+1):(2*N),:]
	Uᵣ_bot = S_T.s_21 * ψ_R[1:N,:]
	
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ, the total eq.-system matrix
	Uᵣ = vcat(Uᵣ_top, Uᵣ_bot)
	
	# evaluate coefficient and store in matrix form
	coeff = (Uₗ^1)' * Uᵣ
	
	# find the number of wave types from size of wavefunction and evanescent matrices
	num_I = size(ψ_L)[2]	# number of incident waves
	num_E = size(E_D)[2]	# number of evanescent waves
	
	# extract solution parameters from coefficients matrix
	τ = coeff[1:(num_I), :]
	α = coeff[(num_I):(num_I+num_E), :]
	r = coeff[(num_I+num_E):(2*num_I+num_E), :]
	β = coeff[(2*num_I+num_E):(2*num_I+2*num_E), :]
	
	# calculate conductance G:
	G = norm(τ)^2
	
	#test  = sum(abs.(τ).^2 + abs.(r).^2)
	#test2 = round(sum(test) / length(test), digits=5)
	return G, τ, α, r, β
end

export system_solve

end
