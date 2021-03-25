# Chalker-Coddington Network Model Scattering Functions:
module ccnscattering

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
end

"""
	S(T)

Given a diagonal transfer matrix `T`, `S(T)`  constructs the correspondnig S-matrix for the Hamiltonian model.

The output `S` is also a `2N`x`2N` matrix of `complex`, `float64` type values.
"""
function S(T::T_data)
	# evaluate s_ij blocks
	s_11 = -(inv(T.t_22) * T.t_21)
	s_12 = inv(T.t_22)
	s_21 = T.t_11 - T.t_12 * inv(T.t_22) * T.t_21
	s_22 = T.t_12 * inv(T.t_22)
	
	return S_data([s_11 s_12; s_21 s_22], s_11, s_12, s_21, s_22)
end

"""
	sum_S(Sa, Sb)

Sums two S-matrix data types (`::S_data`)
"""
function sum_S(Sa::S_data, Sb::S_data)
	I = UniformScaling(1.) # unity Float64 scaling operator
	
	s_11 = Sa.s_11 + Sa.s_12 * inv(I - (Sb.s_11 * Sa.s_22)) * Sb.s_11 * Sa.s_21
	s_12 = Sa.s_12 * inv(I - Sb.s_11 * Sa.s_22) * Sb.s_12
	s_21 = Sb.s_21 * inv(I - Sa.s_22 * Sb.s_11) * Sa.s_21
	s_22 = Sb.s_22 + Sb.s_21 * inv(I - Sa.s_22 * Sb.s_11) * Sa.s_22 * Sb.s_12
	
	return S_data([s_11 s_12; s_21 s_22], s_11, s_12, s_21, s_22)
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
				  (x.t_11 * y.t_11), 
			 	  (x.t_12 * y.t_12),
				  (x.t_21 * y.t_21),
			 	  (x.t_22 * y.t_22))
end

export T, S, sum_S, gen_S_total, prod_T

end
