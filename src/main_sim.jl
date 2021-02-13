module main_sim

using DelimitedFiles
using LinearAlgebra
using QuadGK
using Plots
using CSV

# constsants
const e 	= -1.602176634e-19 	# (C)
const h 	= 6.62607015e-34 	# (Js)
const ħ 	= 1.054571817e-34 	# (Js)
const h_eV 	= abs(h/e) 		 	# (eVs)
const ħ_eV 	= abs(ħ/e) 			# (eVs)

"""
Decomposed transfer properties of `T`, including self.
Used to avoid having to reindex to extract `T` block matrices `t_ij`.
"""
struct T_data
	self # transfer matrix $T
	# component block matrices
	t_11
	t_12
	t_21
	t_22
end;

"""
	T(V_y)

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
	t_11 = -im_mat .+ 0.5 * H
	t_12 = Complex.(-0.5 * H)
	t_21 = Complex.(-0.5 * H)
	t_22 = im_mat .+ 0.5 * H
	
	# assemble transfer matrix blocks t_ij; into matrix T
	T = zeros(Complex{Float64}, 2*N, 2*N)
	T[1:N,1:N] 					= t_11
	T[1:N,(N+1):(2*N)] 			= t_12
	T[(N+1):(2*N),1:N] 			= t_21
	T[(N+1):(2*N),(N+1):(2*N)] 	= t_22
	
	return T_data(T, t_11, t_12, t_21, t_22) # return ::T_data
end;

"""
Decomposed S-matrix properties of `S`, including self.
Used to avoid having to reindex to extract `S` block matrices `s_ij`.
"""
struct S_data
	self # transfer matrix $S
	# component block matrices
	s_11
	s_12
	s_21
	s_22
end;

"""
	S(T, N)

Given a diagonal transfer matrix `T`, `S(T)`  constructs the correspondnig S-matrix for the Hamiltonian model.

The output `S` is also a `2N`x`2N` matrix of `complex`, `float64` type values.
"""
function S(T::T_data)
	N = size(T.t_11)[1]
	# evaluate s_ij blocks
	s_11 = -(T.t_22)^-1 * T.t_21
	s_12 = (T.t_22)^-1
	s_21 = T.t_11 - T.t_12 * (T.t_22)^-1 * T.t_21
	s_22 = T.t_12 * (T.t_22)^-1
	
	# assemble S-matrix from s_ij blocks
	S = zeros(Complex{Float64}, 2*N, 2*N)
	S[1:N,1:N] 					= s_11
	S[1:N,(N+1):(2*N)] 			= s_12
	S[(N+1):(2*N),1:N] 			= s_21
	S[(N+1):(2*N),(N+1):(2*N)] 	= s_22
	
	return S_data(S, s_11, s_12, s_21, s_22) # return ::S_data
end;

"""
	sum_S(Sa, Sb)

Sums two S-matrix data types (`::S_data`)
"""
function sum_S(Sa::S_data, Sb::S_data)
	N = size(Sa.s_11)[1] # later add size equality Sa <-> Sb check
	Id = 1 * Matrix(I, N, N)
	
	# intermediary variables for clarity of inverse calculations
	inter1 = (Id .- Sb.s_11 * Sa.s_22)^-1
	inter2 = (Id .- Sa.s_22 * Sb.s_11)^-1
	
	# evaluate new s_ij block values
	s_11 = Sa.s_11 + Sa.s_12 * inter1 * Sb.s_11 * Sa.s_21
	s_12 = Sa.s_12 * inter1  * Sb.s_12
	s_21 = Sb.s_21 * inter2  * Sa.s_21
	s_22 = Sb.s_22 * Sb.s_21 * inter2 * Sa.s_22 * Sb.s_12
	
	# assemble S-matrix from s_ij blocks
	S = zeros(Complex{Float64}, (2*N), (2*N))
	S[1:N, 1:N] 				= s_11
	S[1:N, (N+1):(2*N)] 		= s_12
	S[(N+1):(2*N), 1:N] 		= s_21
	S[(N+1):(2*N), (N+1):(2*N)] = s_22
	
	return S_data(S, s_11, s_12, s_21, s_22) # return next S::S_data
end;

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
end;

"""
	prod_T(x::T_data, y::T_data)

Multiple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`.
"""
function prod_T(x::T_data, y::T_data)
	return T_data((x.self * y.self),
				  (x.t_11 * y.t_11), 
			 	  (x.t_12 * y.t_12),
				  (x.t_21 * y.t_21),
			 	  (x.t_22 * y.t_22))
end;

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
end;

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
	
	λ_vals = round.(λ.values, digits=11) 	 # round eigenvalues to 10 decimal places
	λ_vecs = round.(λ.vectors, digits=11) # ""	  eigenvectors ""

	#return λ_vals, λ_vecs
	
	# extract indices from:
	# 	forward & backward propagating waves
	# 	evanescent growing & decaying waves
	Rᵢ = pickoutᵢ(λ_vals, "R")
	Lᵢ = pickoutᵢ(λ_vals, "L")
	Eᵢ = pickoutᵢ(λ_vals, "E")
	
	# index evanescent waves which are growing: $Gᵢ or decaying: $Dᵢ
	Gᵢ = pickoutᵢ(λ_vals[Eᵢ], "G")
	Dᵢ = pickoutᵢ(λ_vals[Eᵢ], "D")
	
	#return Rᵢ, Lᵢ, Eᵢ, Gᵢ, Dᵢ
	
	# index $λ_vec to form ψ and E (evanescent) R-, L-mode & G-, D-mode wave arrays
	# which are a numerical representation of the system's wave functions
	ψ_R = λ_vecs[:, Rᵢ]
	ψ_L = λ_vecs[:, Lᵢ]
	E_G = λ_vecs[:, Eᵢ][:, Gᵢ]
	E_D = λ_vecs[:, Eᵢ][:, Dᵢ]
	
	return E_G, E_D
	
	#return ψ_R, ψ_L, E_G, E_D
	# evaluate wave function norms $ψₙ_R & $ψₙ_L
	ψₙ_R = norm(ψ_R[(N+1):2*N])^2 - norm(ψ_R[1:N])^2
	ψₙ_L = norm(ψ_L[1:N])^2 - norm(ψ_L[(N+1):2*N])^2
	
	#return ψ_R, ψ_L
	
	# apply norming factors to wave funtions
	ψ_R /= √(abs(ψₙ_R))
	ψ_L /= √(abs(ψₙ_L))
	
	#return ψ_R, ψ_L
	
	#-formulate system of equations with grouped wave terms:----#
	# $Uₗ_top, create & append to fill 4N sized array
	Uₗ_top1 = -S_T.s_12 * ψ_R[(N+1):(2*N)]
	Uₗ_top2 = E_G[(N+1):(2*N)] - (S_T.s_11 * E_G[1:N])
	
	#return Uₗ_top1, Uₗ_top2
	
	Uₗ_top3 = ψ_L[(N+1):(2*N)] - (S_T.s_11 * ψ_L[1:N])
	Uₗ_top4 = -S_T.s_12 * E_D[(N+1):(2*N)]
	
	#return Uₗ_top1, Uₗ_top2, Uₗ_top3, Uₗ_top4
	# return S_T.s_11, S_T.s_12, S_T.s_21, S_T.s_22
	# $Uₗ_bot, create & append to fill 4N sized array
	Uₗ_bot = ψ_R[1:N] - (S_T.s_22 * ψ_R[(N+1):(2*N)])
	append!(Uₗ_bot, -S_T.s_21 * E_G[1:N])
	append!(Uₗ_bot, -S_T.s_21 * ψ_L[1:N])
	append!(Uₗ_bot, E_D[1:N] - S_T.s_22 * E_D[(N+1):(2*N)])
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ, the total eq.-system matrix
	Uₗ = zeros(Complex{Float64}, 4*N, 2)
	Uₗ[:,1] = Uₗ_top
	Uₗ[:,2] = Uₗ_bot

	# $Uᵣ_top & $Uᵣ_bot create 4N sized arrays
	Uᵣ_top = S_T.s_11 * ψ_R[1:N] - ψ_R[(N+1):(2*N)]
	Uᵣ_bot = S_T.s_21 * ψ_R[1:N]
	# assemble $Uₗ_top & $Uₗ_bot into $Uₗ
	# the total eq.-system matrix
	Uᵣ = zeros(Complex{Float64}, 4*N, 2)
	Uᵣ[:,1] = Uₗ_top
	Uᵣ[:,2] = Uₗ_bot
	#-----------------------------------------------------------#
	
# 	# evaluate coefficient and store in matrix form
# 	#coeff = (Uₗ^1)' * Uᵣ
# 	return Uᵣ, Uₗ
end

##		Testing		##

enen = 0.5
LL = 200
NN = 40
VV = smooth_potential(enen, NN, LL, 1., 1., .6, 2)
a,b = system_solve(enen, VV, NN, LL, enen, false)
#writedlm( "wavefunctionR_julia.csv",  a, ',')
#writedlm( "wavefunctionL_julia.csv",  b, ',')
# writedlm( "transfermat_julia.csv",  T(enen, NN).self, ',')
#size(a), size(b)
a,b

end
