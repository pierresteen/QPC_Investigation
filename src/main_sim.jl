module main_sim

using DelimitedFiles
using LinearAlgebra
using QuadGK
using Plots
using CSV

# constsants
# const e 	= -1.602176634e-19 	# (C)
# const h 	= 6.62607015e-34 	# (Js)
# const ħ 	= 1.054571817e-34 	# (Js)
# const h_eV 	= abs(h/e) 		# (eVs)
# const ħ_eV 	= abs(ħ/e) 		# (eVs)

"""
Decomposed transfer properties of `T`, including self.
Used to avoid having to reindex to extract `T` block matrices `t_ij`.
"""
struct T_data
	self # transfer matrix $T
	# component block matrices
	t11
	t12
	t21
	t22
end

"""
	T(μ, N)

Generates a diagonal tranfer data type `T::T_data` for given bias potentail: `μ`.
`T` is a `2N`x`2N` matrix, with three main diagonals at `diagind[1] = 0, 5, -5`.
"""
function T(μ, N)
	# create complex float type matrix with 1im diagonal values
	im_mat = zeros(Complex{Float64}, N, N)
	im_mat[diagind(im_mat)] .= 1im
	v = ones(Float64, N-1)	# (N-1)-length array of Float 1s

	# create tight-binding Hamiltonian model
	H = diagm(-1 => -v) + diagm(0 => 4*ones(Float64, N) .- μ) + diagm(1 => -v)
	
	# form blocked transfer matrix blocks t_ij from $im_mat, $v and $H
	t11 = -im_mat .+ 0.5 * H
	t12 = Complex.(-0.5 * H)
	t21 = Complex.(-0.5 * H)
	t22 = im_mat .+ 0.5 * H
	
	# assemble transfer matrix blocks t_ij; into matrix T
	T = zeros(Complex{Float64}, 2*N, 2*N)
	T[1:N,1:N] 					= t11
	T[1:N,(N+1):(2*N)] 			= t12
	T[(N+1):(2*N),1:N] 			= t21
	T[(N+1):(2*N),(N+1):(2*N)] 	= t22
	
	return T_data(T, t11, t12, t21, t22) # return ::T_data
end

"""
Decomposed S-matrix properties of `S`, including self.
Used to avoid having to reindex to extract `S` block matrices `s_ij`.
"""
struct S_data
	self # transfer matrix $S
	# component block matrices
	s11
	s12
	s21
	s22
end

"""
	S(T, N)

Given a diagonal transfer matrix `T`, `S(T)`  constructs the correspondnig S-matrix for the Hamiltonian model.

The output `S` is also a `2N`x`2N` matrix of `complex`, `float64` type values.
"""
function S(T::T_data)
	N = size(T.t11)[1]
	# evaluate s_ij blocks
	s11 = -(T.t22)^-1 * T.t21
	s12 = (T.t22)^-1
	s21 = T.t11 - T.t12 * (T.t22)^-1 * T.t21
	s22 = T.t12 * (T.t22)^-1
	
	# assemble S-matrix from s_ij blocks
	S = zeros(Complex{Float64}, 2*N, 2*N)
	S[1:N,1:N] 					= s11
	S[1:N,(N+1):(2*N)] 			= s12
	S[(N+1):(2*N),1:N] 			= s21
	S[(N+1):(2*N),(N+1):(2*N)] 	= s22
	
	return S_data(S, s11, s12, s21, s22) # return ::S_data
end

"""
	sum_S(Sa, Sb)

Sums two S-matrix data types (`::S_data`)
"""
function sum_S(Sa::S_data, Sb::S_data)
	N = size(Sa.s11)[1] # later add size equality Sa <-> Sb check
	Id = 1 * Matrix(I, N, N)
	
	# intermediary variables for clarity of inverse calculations
	inter1 = (Id .- Sb.s11 * Sa.s22)^-1
	inter2 = (Id .- Sa.s22 * Sb.s11)^-1
	
	# evaluate new s_ij block values
	s11 = Sa.s11 + Sa.s12 * inter1 * Sb.s11 * Sa.s21
	s12 = Sa.s12 * inter1  * Sb.s12
	s21 = Sb.s21 * inter2  * Sa.s21
	s22 = Sb.s22 * Sb.s21 * inter2 * Sa.s22 * Sb.s12
	
	# assemble S-matrix from s_ij blocks
	S = zeros(Complex{Float64}, (2*N), (2*N))
	S[1:N, 1:N] 				= s11
	S[1:N, (N+1):(2*N)] 		= s12
	S[(N+1):(2*N), 1:N] 		= s21
	S[(N+1):(2*N), (N+1):(2*N)] = s22
	
	return S_data(S, s11, s12, s21, s22) # return next S::S_data
end

"""
	gen_S_total(V, L)

Generates all S-matrices for the column 'slices' of network model.
"""
function gen_S_total(V, L)
	# eval first column of S-matrices on first column of V
	S_T = S(T(V[:,1], size(V)[1]))
	for j in 2:L
		S_T = sum_S(S_T, S(T(V[:,j], size(V)[1])))
	end
	return S_T
end

"""
	prod_T(x::T_data, y::T_data)

Multiple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`.
"""
function prod_T(x::T_data, y::T_data)
	return T_data((x.self * y.self),
				  (x.t11 * y.t11), 
			 	  (x.t12 * y.t12),
				  (x.t21 * y.t21),
			 	  (x.t22 * y.t22))
end

"""
	gen_S_total_opt(V, L)

Multiple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`, optional method for use in the case of non multiples-of-ten sized networks.
"""
function gen_S_total_opt(V, L)
	# define transfer and scattering total matrices
	T_Topt = T(V[:,1], size(V)[1])
    S_Topt = S(T(V[:,1], size(V)[1]))
	
	@assert mod(L, 10) == 0
	for j in 1:L
		if mod(j, 10) != 0
			T_Topt = prod_T(T(V[:, j], size(V)[1]), T_Topt)
		else
			S_tot = sum_S(S_Topt, S(T_Topt))
            T_Topt = T(V[:, j], size(V)[1])
		end
	end
	
	return S_Topt
end

"""
	smooth_potential(μ, N, L, xL=1.,yL=1., h=1., prof=1)

Creates a smooth potential profile for the model.
The profile type can be changed bassed on the `prof` parameter.
"""
function smooth_potential(μ, N, L, xL=1.,yL=1., amp=1., profile=1)
	# create empty network domain, spaced xy-axis points
	V  = zeros(Float64, N, L)
	px = Float64.(range(-xL, xL, length=L))
	py = Float64.(range(-yL, yL, length=N))
	
	# populate V according to potential distribution model choice `prof`
	if profile == 1
		for i in 1:length(px)
			for j in 1:length(py)
				V[j,i] = abs(amp)*(px[i]^2 - py[j]^2) + μ
			end
		end
	elseif profile == 2
		for i in 1:length(px)
			for j in 1:length(py)
				V[j,i] = -0.5 * amp * (tanh(py[j]^2 - px[i]^2) + 1) + μ
			end
		end
	end
	
	return V
end

"""
	error_ϵ(S::S_data, T::T_data)

Method which evaluates the model error from `S:S_data` and `T::T_data`.
"""
function error_ϵ(S::S_data, T::T_data)
	return norm(S.self * conj(T.self) - (1 * Matrix(I, size(S.self)[1], size(S.self)[2])))
end

"""
	pickoutᵢ(λ_values, mode)

Returns an array of indices for eigenvalues in `λ_values` which correspond to:

`R ->` right-propagating waves

`L ->` left-propagating waves

`E ->` evanescent waves

`G ->` growing evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*

`D ->` decaying evanescent waves *(`λ_values` must be pre-indexed to `E` waves)*
"""
function pickoutᵢ(λ_values, mode)
	if mode == "R"
		# add the indices which equate to right-propagating eigenvals to $indices
		binᵢ = imag(λ_values) .> 0
		arrᵢ = findall(!iszero, binᵢ)
	elseif mode == "L"
		# add the indices which equate to left-propagating eigenvals to $indices
		binᵢ = imag(λ_values) .< 0
		arrᵢ = findall(!iszero, binᵢ)
	elseif mode == "E"
		# add the indices which equate to evanescent wave eigenvals to $indices
		binᵢ = imag(λ_values) .== 0
		arrᵢ = findall(!iszero, binᵢ)
	elseif mode == "G"
		binᵢ = abs.(λ_values) .> 1
		arrᵢ = findall(!iszero, binᵢ)
	elseif mode == "D"
		binᵢ = abs.(λ_values) .< 1
		arrᵢ = findall(!iszero, binᵢ)
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
	ψₙ_R = norm(ψ_R[(N+1):2*N])^2 - norm(ψ_R[1:N])^2
	ψₙ_L = norm(ψ_L[1:N])^2 - norm(ψ_L[(N+1):2*N])^2
	
	# apply norming factors to wave funtions
	ψ_R = ψ_R ./ √(abs(ψₙ_R))
	ψ_L = ψ_L ./ √(abs(ψₙ_L))
	#-- passes until here!!
	
	## formulate system of equations with grouped wave terms: ##
	
	# $Uₗ_top, create & append to fill 4N sized array
	Uₗ_top = -S_T.s_12 * ψ_R[(N+1):(2*N),:]
	Uₗ_top = cat(Uₗ_top, E_G[(N+1):(2*N),:] - (S_T.s_11 * E_G[1:N,:]), dims=2)
	Uₗ_top = cat(Uₗ_top, ψ_L[(N+1):(2*N),:] - (S_T.s_11 * ψ_L[1:N,:]), dims=2)
	Uₗ_top = cat(Uₗ_top, -S_T.s_12 * E_D[(N+1):(2*N),:], dims=2)
	#-- passes until here

	# $Uₗ_bot, create & append to fill 4N sized array
	Uₗ_bot = ψ_R[1:N,:] - (S_T.s_22 * ψ_R[(N+1):(2*N),:])
	Uₗ_bot = cat(Uₗ_bot, -S_T.s_21 * E_G[1:N,:], dims=2)
	Uₗ_bot = cat(Uₗ_bot, -S_T.s_21 * ψ_L[1:N,:], dims=2)
	Uₗ_bot = cat(Uₗ_bot, E_D[1:N,:] - S_T.s_22 * E_D[(N+1):(2*N),:], dims=2)
	#return Uₗ_bot
	#-- passes until here
	
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ, the total eq.-system matrix
	Uₗ = zeros(Complex{Float64}, 2*N, 2*N)
	Uₗ[1:N,:] 		  = Uₗ_top
	Uₗ[(N+1):(2*N),:] = Uₗ_bot
	#Uₗ = hcat()
	#return Uₗ
	#-- passses until here

	# $Uᵣ_top & $Uᵣ_bot create 4N sized arrays
	Uᵣ_top = S_T.s_11 * ψ_R[1:N,:] - ψ_R[(N+1):(2*N),:]
	Uᵣ_bot = S_T.s_21 * ψ_R[1:N,:]
	
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ, the total eq.-system matrix
	Uᵣ = zeros(Complex{Float64}, 2*N, size(Uᵣ_bot)[2])
	Uᵣ[1:N,:] 		  = Uᵣ_top
	Uᵣ[(N+1):(2*N),:] = Uᵣ_bot
	
	# evaluate coefficient and store in matrix form
	coeff = (Uₗ^1)' * Uᵣ
	#return coeff 
	
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

##		Testing		##
# enen = 0.5
# LL = 200
# NN = 40
# VV = smooth_potential(enen, NN, LL, 1., 1., .6, 2)
# a,b = system_solve(enen, VV, NN, LL, enen, false)
# #writedlm( "wavefunctionR_julia.csv",  a, ',')
# #writedlm( "wavefunctionL_julia.csv",  b, ',')
# # writedlm( "transfermat_julia.csv",  T(enen, NN).self, ',')
# #size(a), size(b)
# a,b

export T, S
end
