### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 9cabcdd6-5e68-11eb-0613-9785eb761d6d
begin
	using PlutoUI
	using QuadGK
	using Plots
	using CSV
	using DataFrames
	using LinearAlgebra
	using DelimitedFiles
end

# ╔═╡ 7ee2fb54-433c-11eb-1f9b-3528ac7148a4

md"""
# Numerical solutions to the quantum transport problem in a 1D system

The quantum transport system we will be investigating in this notebook is a quantum point contact (QPC).
This is simple system which displays quasi-one-dimensional quantum transport phenomena such as the characteristic quantisation of conductance.

In the name of simplifying the numerical solving process, we treat the device's constriction with a **saddle-point approximation** of the confinement potential energy.

We then frame the transport problem using the **non-equilibrium Green function formalism** (NGEF), and obtain solutions for the conductance $G$ for varying bias potentials.
"""

# ╔═╡ e616e6b0-61fe-11eb-398b-4fde45cba90f


# ╔═╡ 3ab951aa-5f2d-11eb-24d3-9d64610bf050
md"## Packages:"

# ╔═╡ ca57a27e-61cd-11eb-0057-a7a89cb2f828


# ╔═╡ 8ce21aca-5cad-11eb-0d3e-53ee628dd525
md"""
## The QPC device

When a negative voltage split-gate is applied on top of a 2DEG; the conductance of the QPC, in the unconfined electron momentum dimension, was found to be quantised in multiple of:

$G_0 = \frac{2e^2}{h}$

**This is classic result associated with ballistic transport of electrons across a one-dimensional channel**.
The figure below illustrates the heterojunction geometry used to achieve this, notice that the system can sufficiently be described by 2 dimensions globally and behaves as a quasi-one-dimensional system in the constriction channel.

![Imgur](https://imgur.com/h1KrE1H.png)


Bias potentials $V_{s}$ and $V_{d}$, applied to the source and drain respectively,  create a potential differnce $V_{sd}$, which is dropped along the path between the terminals $S$ and $D$.
We shall cover the different ways in which this potential energy can be distributed along the dimension in which electron momentum is free later in our analysis.

The negative voltages $V_{sg1}$ and $V_{sg2}$, applied at the split-gate terminals $SG1$ and $SG2$, provide a confinement potential for the electrons crossing the channel ballistically.

Recalling the behaviour of single electrons confined in quantum wells, we can say that the number of conducting bands and sub-bands in the channel will depend on the profile of the applied potential in the lateral (restricted) dimension 'Y'.

Cartesian profile of the system:

> Z -- dimension in which the **3DEG** is confined to form a **2DEG**
>
> Y -- dimension in which the **2DEG** is confined to form a **1DEG** (constriction)
>
> X -- free dimension in which quantum transport is considered **ballistic**
"""

# ╔═╡ 46f71c54-5f2d-11eb-3a79-c96ae093a6cc


# ╔═╡ b4e0828e-6110-11eb-2cca-2bcb5a409caf
md"""
## The saddle-point constriction model

A simplification to the potential energy profile of the constriction is achieved by  treating it as a saddle-point potential.
**This simplification only applies when the channel between the split-gates is short enough**.

We gain an advantage from treating the potential profile in this way because the saddle-point surface can simply be modelled as a function of $x$ and $y$.

The shape of the saddle-point potential model is illustrated below.
"""

# ╔═╡ 5ecf3cf4-6116-11eb-11c5-db4253ba8d7f
begin
	plotly()
	x = range(-5., 5., length=50)
	y = range(-5., 5., length=50)
	
	out = zeros(Float64, length(x), length(y))
	for i in 1:length(x)
		for j in 1:length(y)
			out[i,j] = x[i]^2 - y[j]^2
		end
	end
	
	p_saddle = surface(x, y, out, ticks=false, xlabel="X", ylabel="Y", zlabel="Z")
	title!(p_saddle, "General saddle-point constriction potential profile")
	plot!(p_saddle, camera=(40,40))
end

# ╔═╡ c8c5010c-5f6d-11eb-020f-b5a3fa043f1e


# ╔═╡ dc91ea8c-5f6e-11eb-20ed-318a74d2f404
md"## Constants:"

# ╔═╡ ef273a10-5f6e-11eb-386e-4df51c71d0b5
begin
	const e 	= -1.602176634e-19 	# (C)
	const h 	= 6.62607015e-34 	# (Js)
	const ħ 	= 1.054571817e-34 	# (Js)
	const h_eV 	= abs(h/e) 		 	# (eVs)
	const ħ_eV 	= abs(ħ/e) 			# (eVs)
end;

# ╔═╡ 3d636042-61ff-11eb-1b22-9555285fe9af


# ╔═╡ 3e467742-61ff-11eb-3640-8f313ff08354
md"""
## The scattering problem approach

### Setup:
For this approach, we want to treat the system as a **scattering problem**, in the context of a **tight-binding, discretised** network-like system.
We assume 'zero-mode' dispersion of the wave functions in the system, which means the 2DEG at the leads (labelled $\text{I}$ and $\text{II}$ on the device schematic), can be characterised by the following set of wave functions:

> the wave function incident on the saddle-point constriction from $\text{I}$:
>
> $\psi^i_{inc}$ 
>
> the wave function from $\text{I}$ reflected by the barrier back into $\text{I}$:
>
> $\psi_{L,out}$
>
> the wave function from $\text{I}$ which crosses the barrier into $\text{II}$:
>
> $\psi_{R,out}$

These form a system of equations as:

$\psi_L = \psi^i_{inc} + \Sigma_j r_{ji}\psi_{L,out} + \Sigma_k \beta_{ki}\psi_{L,ev}$
$\psi_R = \Sigma_j t_{ji}\psi_{R,out} + \Sigma_k \alpha_{ki}\psi_{L,ev}$

where $\psi_{L,ev}$ is the wavefunction that describes $\text{I}$ at rest.

**We use the *Chalker-Coddigton Network Model* to discretise our system**, and now consider the system as a network of localised nodes.
This allows us to evaluate the wave function of electrons in the system for the each column along the free dimension.

A crude interpretaion of this struture is presented below:

```julia
( ) = lead nodes with no scattering
(X) = scattering nodes (QPC area)
```

```julia
+---------------------------------------------------------> (X)
|		...	n-4	n-3	n-2	n-1  n 	n+1 n+2 n+3 n+4 ...
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
| 	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |	 |
|	( )	( )	( )	(X)	(X)	(X)	(X)	(X)	(X)	(X) ( )	( )	( )
|
|	left lead 	 	scattering region 		right lead
|		I 				(barrier)				II
V
(Y)
```

The wave functions of each column are represented by:

$\psi_n(y)$
"""

# ╔═╡ adaf3546-72f4-11eb-0b21-e7466c2d81be
md"""
### Data Structures
"""

# ╔═╡ 4dedeecc-6246-11eb-00c7-014b87b08c32
"""
**Transfer matrix data type**

Decomposed transfer properties of `T`, including self.
Used to avoid having to reindex to extract `T` block matrices `t_ij`.
"""
struct T_data
	self # transfer matrix T
	t_11
	t_12
	t_21
	t_22
end;

# ╔═╡ b9d7ddd8-624a-11eb-1084-35320b3f9afb
"""
**Scattering matrix data type**

Decomposed S-matrix properties of `S`, including self.
Used to avoid having to reindex to extract `S` block matrices `s_ij`.
"""
struct S_data
	self # transfer matrix S
	s_11
	s_12
	s_21
	s_22
end;

# ╔═╡ b06e326c-72f6-11eb-204a-ef48d6cbf876
struct Sys_sol
	G
	τ
	α
	r
	β
end;

# ╔═╡ 7158aa22-72f5-11eb-246b-3bb16136c59c


# ╔═╡ 70e21632-72f5-11eb-341e-0f121cf27eac
md"""
### Simulation Functions
"""

# ╔═╡ 06038796-6234-11eb-3dd3-cf25a7095963
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

# ╔═╡ 41a9c7cc-6245-11eb-148b-3791b3fb504c
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
end;

# ╔═╡ faedfda0-72d7-11eb-0b80-7d63e962468d
md"""
To reference: `sum_S(Sa, Sb)` see equation §B6:

[Calculation of the conductance of a graphene sheet using the Chalker-Coddingtonnetwork model](https://journals-aps-org.libproxy.ucl.ac.uk/prb/pdf/10.1103/PhysRevB.78.045118).
"""

# ╔═╡ fce9afc0-624a-11eb-09e2-c38456a1fe35
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
end;

# ╔═╡ d03c2ac6-6253-11eb-0483-596dd3d5e5a4
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

# ╔═╡ 095be506-64e5-11eb-3ac8-6dbf5a7f5f9e
"""
	prod_T(x::T_data, y::T_data)

Mutliple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`.
"""
function prod_T(x::T_data, y::T_data)
	return T_data((x.self * y.self),
				  (x.t_11 * y.t_11), 
			 	  (x.t_12 * y.t_12),
				  (x.t_21 * y.t_21),
			 	  (x.t_22 * y.t_22))
end;

# ╔═╡ 08169170-64e3-11eb-3fbe-6b50b31ee02f
"""
	gen_S_total_opt(V, L)

Mutliple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`, optional method for use in the case of non multiples-of-ten sized networks.
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

# ╔═╡ fe16d518-64e7-11eb-04f5-bb25ed0a9eea
function smooth_potential_broken(μ, N, L, xL=1.,yL=1., amp=1., profile=1)
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
	
	return px, py, V
end

# ╔═╡ 212d911a-7dc3-11eb-11ee-333220a641e5
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# ╔═╡ 9ff7af7e-7dc2-11eb-17b8-e7fe576888c4
"""
	smooth_potential(μ, N, L, xL=1.,yL=1., h=1., prof=1)

Creates a smooth potential profile for the model.
"""
function smooth_potential(μ, N, L, xL=1.,yL=1., amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	
	return (-0.5 * amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
end

# ╔═╡ 6b63b052-64eb-11eb-1a62-33262062ece1
"""
	error_ϵ(S::S_data, T::T_data)

Method which evaluates the model error from `S:S_data` and `T::T_data`.
"""
function error_ϵ(S::S_data, T::T_data)
	return norm(S.self * conj(T.self) - (1 * Matrix(I, size(S.self)[1], size(S.self)[2])))
end

# ╔═╡ 8ef3c6e0-72f3-11eb-3244-1f406e6bde18
md"Potential plot beneath ↓"

# ╔═╡ 629a9616-625c-11eb-0e76-536b5de36ab7
begin
# 	Vsg = -0.1
# 	# network dimensions and potential ranges
# 	V 	= zeros(Float64, 40, 100)
# 	px 	= range(-5., 5., length=100)
# 	py 	= range(-5., 5., length=40)
	
# 	# map saddle-point potential profile to network matrix V
# 	for i in 1:length(px)
# 		for j in 1:length(py)
# 			V[j,i] = abs(Vsg)*(px[i]^2 - py[j]^2)
# 		end
# 	end
	#ptest = surface(px, py, V, xlabel="Y", ylabel="X", zlabel="Z", title="Potential barrier profile")
	# uncomment above to show plot
end

# ╔═╡ 2fd2a6c8-6256-11eb-2b61-1deb1e2e4c77
md"""
### Solving numerically

The matrix $S$  is calcuated by solving the following system of equations:

$[U_{R,out}\quad U_{R,ev} \quad - S_{tot}\cdot U_{out} \quad - S_{tot}\cdot U_{ev}][t\quad \alpha\quad r\quad \beta]^T = S_{tot}\cdot U_{in}$

For each propagation mode $i$, the sum of transmissions and reflections must sum to unity.
This is condition we enforce to obtain solutions:

$\Sigma^{N}_{j=1} |r_{ij}|^2 + |s_{ij}|^2$

The conductance of the system is given by Landauer's eq.:

$G = G_0\ \Sigma_{i,j} |s_{ij}|^2$

"""

# ╔═╡ 210393f2-65ad-11eb-3dc0-0bcab1b97c73
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

# ╔═╡ 5ad541b0-64eb-11eb-0782-a59689a23af5
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
	# which are a numerical representation of the system's wave fucntions
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

# ╔═╡ 7c8dd648-7dad-11eb-2d5e-69d011342fe8
## results test
begin
	enen = 0.5
	LL = 200
	NN = 40
	
	energies = 0.3:0.0141:1.
	G = zeros(Float64, length(energies))
	
	for i in 1:length(energies)
		VV = smooth_potential(energies[i], NN, LL, 1., 1., .6)
		G[i] = system_solve(enen, VV, NN, LL, enen, false)[1]
	end
	plot(energies, G)
	# close but not quite there yet, something about the oscillation of G isn't right
end

# ╔═╡ 6bde2f84-6258-11eb-0e07-af0a2275fd79


# ╔═╡ 7545065e-72e7-11eb-1db0-3df6683bcbeb
md"""
## Debugging

To be able to quickly compare the numerial results obtained by *julia* and *numpy* at each step if the simulation, the functions below:

```julia
complexclean(string_in::String)

csvcomplexparse(file_dir::String)
```
have been created.
These allow us to quickly import a `.csv` numpy-complex-type array to julia.
A conversion between the native numpy complex type using `j` and the *julian*:

```julia
Complex{Float64}
```

type is done through a string array intermediary, which is mapped to a clean julia-parsing compatible clean form array by:

```julia
complexclean(string_in::String)
```

The result of this is a grouped function that performs:

```julia
file_dir::String -> csvcomplexparse(file_dir)::Array{Complex{Float64},2}
```

"""

# ╔═╡ b1c556a8-72e3-11eb-1299-8b52ae0c19b7
"""
	complexclean(string_in::String)

Cleans incoming *numpy complex* formatted string and returns a clean string that can be **parsed** to a *julia* `Complex` type using:

```
parse(Complex{Float64}, string_parse)
```
"""
function complexclean(string_in::String)
	let
		clean = ""
		for char in string_in
			if char == ' '
				continue
			elseif char == '(' || char == ')'
				continue
			elseif char == 'j'
				clean *= "im"
			else
				clean *= char
			end
		end
		return clean
	end
end

# ╔═╡ 5e9936fa-72e5-11eb-078f-bd9e193eda1a
"""
	csvcomplexparse(file_dir::String)

Maps an imported `Array{String,2}` type, from numpy via csv intermediary, cleaning and parsing, to an `Array{Complex{Float64},2}` type.

`file_dir::String ->` location of csv stored numpy array
"""
function csvcomplexparse(file_dir::String)
	strs = CSV.File(file_dir, header=0) |> Tables.matrix |> x -> map(complexclean, x)
	cmps = parse.(Complex{Float64}, strs)
	
	return cmps
end

# ╔═╡ d8cbc6e4-7dbd-11eb-378e-8bf7a5d244f2
function csvfloatparse(file_dir::String)
	return CSV.File(file_dir, header=0) |> Tables.matrix
end

# ╔═╡ 43b40838-72f4-11eb-068b-b7152a884e66
md"""
### Incremental test comparison
In this section, we make use of the import functions defined above, we incrementally compare the numerical results obtained in the python simulation to those obtained in this simulation.
"""

# ╔═╡ e27d74fe-6e6c-11eb-08d5-b988732170d0
begin
	# defs
	μ = 0.5
	L = 200
	N = 40
end;

# ╔═╡ 65a0f08e-7dc4-11eb-1676-8ba451a5574a
smooth_potential(μ, N, L)

# ╔═╡ cd720bf4-7dad-11eb-0e06-3fc6fa6c91c9
md"""
#### Testing via python matrix import comparison
"""

# ╔═╡ 93dccf8a-7dac-11eb-25f8-cd19f446b7a3
# test & compare T(μ)
begin
	T_compare = T(μ, N)			# julia T constructor function
	T_julia = T_compare.self 	# 'self' T matrix
	
	@assert T_julia[1:N,1:N] == T_compare.t_11
	@assert T_julia[1:N,(N+1):(2*N)] == T_compare.t_12
	@assert T_julia[(N+1):(2*N),1:N] == T_compare.t_21
	@assert T_julia[(N+1):(2*N),(N+1):(2*N)] == T_compare.t_22
	# T sub matrices are correct
	
	T_python =  csvcomplexparse("matrices/T_python.csv")
	
	@assert T_julia == T_python # tests true!
end
# PASS

# ╔═╡ 168b4c16-7dc8-11eb-3bbe-9d52f6558742
T_python_D = T_data(T_python,
				  T_python[1:N,1:N],
				  T_python[1:N,(N+1):(2*N)],
				  T_python[(N+1):(2*N),1:N],
				  T_python[(N+1):(2*N),(N+1):(2*N)])

# ╔═╡ 3391ed90-7dae-11eb-054d-db7a314ab852
# test & compare S(T(μ))
begin
	S_compare = S(T_compare)	# julia S constructor function
	S_julia = S_compare.self	# 'self' S matrix
	#writedlm("matrices/S_julia.csv",  S_julia, ',')

	@assert S_julia[1:N,1:N] == S_compare.s_11
	@assert S_julia[1:N,(N+1):(2*N)] == S_compare.s_12
	@assert S_julia[(N+1):(2*N),1:N] == S_compare.s_21
	@assert S_julia[(N+1):(2*N),(N+1):(2*N)] == S_compare.s_22
	# S sub matrices are correct

	S_python =  csvcomplexparse("matrices/S_python.csv")

	for el in abs.(S_julia - S_python)
		@assert isapprox(el, 0, atol = 0.000001)
	end # abs. error margin is for els. in T_0.000001
end
# PASS, JULIA AND PYTHON METHODS ARE EQUAL BUT FOR ROUNDING ERRORS

# ╔═╡ 76344974-7db4-11eb-19c0-3b2116705950
S_python_D = S_data(S_python,
				  S_python[1:N,1:N],
				  S_python[1:N,(N+1):(2*N)],
				  S_python[(N+1):(2*N),1:N],
				  S_python[(N+1):(2*N),(N+1):(2*N)])

# ╔═╡ 0743b716-7db4-11eb-1fcd-13f95d1daf8b
# test & compare sum_S(S, S)
begin
	sumS_compare_j = sum_S(S_compare, S_compare)
	sumS_compare_p = sum_S(S_python_D, S_python_D)
	
	verif = isapprox.(sumS_compare_j.self, sumS_compare_p.self, atol = 0.00001)
	length(findall(verif .== 1))
	# sumS_compare_j and sumS_compare_p are approx equal to atol = 0.00001
	
	
	sumS_python = csvcomplexparse("matrices/addS_python.csv")
	
	verif = isapprox.(sumS_compare_j.self, sumS_python, atol = 0.1)
	length(findall(verif .== 1))
	# sumS_compare_j and sumS_compare_p are approx equal to atol = 0.00001
	
	#writedlm("matrices/addS_julia.csv",  sumS_compare_j.self, ',')
	
	## compare the sub matrices of sumS_julia and sumS_python
	sumS_julia = sumS_compare_j.self
	
	@assert isapprox(sumS_julia[1:N,1:N], sumS_python[1:N,1:N], atol=0.0001)
	@assert isapprox(sumS_julia[1:N,(N+1):(2*N)], sumS_python[1:N,(N+1):(2*N)], atol=0.0001)
	@assert isapprox(sumS_julia[(N+1):(2*N),1:N], sumS_python[(N+1):(2*N),1:N], atol=0.0001)
	@assert isapprox(sumS_julia[(N+1):(2*N),(N+1):(2*N)], sumS_python[(N+1):(2*N),(N+1):(2*N)], atol=0.0001)
end
# FAIL, SOMETHING IS WRONG WITH S_22 OF THE JULIA SUM_S ALGORITH
# s_22 = Sb.s_22 + Sb.s_21 * inv(I - Sa.s_22 * Sb.s_11) * Sa.s_22 * Sb.s_22
# BUG FOUND, TYPO FIXED FROM: `... * Sb.s_22` to `... * Sb.s_12`
# PASS, DEGREE OF ACCURACY TOLERANCE: atol = 0.0001 = 1e-4

# ╔═╡ 3dc4eab8-7dbd-11eb-11a5-d5b1c33b5541
# test & compare smooth_potential (cell 1)
begin
	px 	= range(-1., 1., length=200)
	py 	= range(-1., 1., length=40)
	sP_python = csvfloatparse("matrices/sP_python.csv")
	#pythontest = surface(px, py, sP_python, xlabel="Y", ylabel="X", zlabel="Z", title="Potential barrier profile")
end;

# ╔═╡ 5fd3174a-7dbf-11eb-22c0-13725c493e74
# test & compare smooth_potential (cell 2)
begin
	sP_julia = smooth_potential(μ, N, L)
	#juliatest = surface(px, py, sP_julia, xlabel="Y", ylabel="X", zlabel="Z", title="Potential barrier profile")
end;
# PASS

# ╔═╡ a7cda810-7dbc-11eb-2434-31d93e7651e0
# test & compare gen_S_total(V, L)
begin
	genSt_python = csvcomplexparse("matrices/genSt_python.csv")
	
	genSt_compare = gen_S_total(sP_python, L)
	genSt_julia = genSt_compare.self
	
	@assert isapprox(genSt_python, genSt_julia, atol = 0.0001)
end
# PASS, DEGREE OF ACCURACY TOLERANCE: atol = 0.0001 = 1e-4

# ╔═╡ 0190c7c0-7dc6-11eb-2f0a-3dcdd3e7fa2c
# test & compare gen_S_total_opt(V, L)
begin
	genStopt_python = csvcomplexparse("matrices/genStopt_python.csv")
	
	genStopt_compare = gen_S_total_opt(sP_python, L)
	genStopt_julia = genStopt_compare.self
	
	@assert isapprox(genStopt_python, genSt_julia, atol = 4)
end
# FAIL (LEAVE FOR NOW SINCE FUNCTIONS ONLY PROVIDES AN OPTIONAL SPEED INCREASE)

# ╔═╡ 64b90c6c-7dc7-11eb-2de4-fd1f990d6860
# test error_ϵ(S::S_data, T::T_data)
begin
	errorS_julia = error_ϵ(S_python_D, T_python_D)
end

# ╔═╡ f74d6a68-61e9-11eb-0ed8-8bdd85177922
#= md"""
**TO BE DEVELOPED ON FURTHER**

## Non-equilibrium Green function formalism (NEGF)

The non-equilibrium Green function formalism approach to describing quantum transport in a channel which can be crossed ballistically, revolves around the following relationship:

$E[\psi] = [H][\psi] = [\Sigma][\psi] + [s]$

which in fact a compact form of the following matrices:

![NGEF1](https://imgur.com/vBMNpYr.png)

Here, $H$ is the Hamiltonian operator and $\Sigma$
""" =#

# ╔═╡ f195457c-7dce-11eb-1326-83ed59d18879
md"""
## Experimental Impurity Data:
The data represents a shift in the QPC channel from left to right which is obtained by applying a differential voltage on left and right split gates, here, -0.1 V and 0.1 V respectively.
Data is seperated into **`clean`** & **`noisy`**.

---
**Labels:**

**`splitgate_V ->`** voltage applied across the split-gate to bias channel to the 'right'.

**`conductance ->`** **G** (μS) measured across QPC channel

---
**Properties:**

**`clean ->`** no impurities in QPC channel

**`noisy ->`** impurities present in channel (varying types, further analyis on this needed)

---
"""

# ╔═╡ 552aa3e0-7dd2-11eb-399e-ad5fc208fbc5
md"""
#### Data Import:
"""

# ╔═╡ d5a18c24-7dd1-11eb-30d4-6dcb0c8c9c4e
begin
	cleanV = csvfloatparse("data/splitgateV_clean.csv")
	cleanG = csvfloatparse("data/conductance_clean.csv")

	noisyV = csvfloatparse("data/splitgateV_noisy.csv")
	noisyG = csvfloatparse("data/conductance_noisy.csv")
	
	plotly()
end

# ╔═╡ 5056efdc-7dd6-11eb-21d3-e13768d765d9
clean_plot = plot(cleanV[1:(size(cleanV)[1]),1:(size(cleanV)[2])],
				  cleanG[1:(size(cleanG)[1]), 1:(size(cleanG)[2])],
				  title="Channel Conductance of a Clean QPC",
				  xlabel="Split Gate Voltage (V)",
				  ylabel="Channel Conductance G (μS)")

# ╔═╡ fa5d996e-7dd5-11eb-3010-8935856e0b68
noisy_plot = plot(noisyV[1:(size(noisyV)[1]), 1:(size(noisyV)[2])],
				  noisyG[1:(size(noisyG)[1]), 1:(size(noisyG)[2])],
				  title="Channel Conductance of a QPC Containing Impurities",
				  xlabel="Split Gate Voltage (V)",
				  ylabel="Channel Conductance G (μS)")

# ╔═╡ Cell order:
# ╟─7ee2fb54-433c-11eb-1f9b-3528ac7148a4
# ╟─e616e6b0-61fe-11eb-398b-4fde45cba90f
# ╟─3ab951aa-5f2d-11eb-24d3-9d64610bf050
# ╠═9cabcdd6-5e68-11eb-0613-9785eb761d6d
# ╟─ca57a27e-61cd-11eb-0057-a7a89cb2f828
# ╟─8ce21aca-5cad-11eb-0d3e-53ee628dd525
# ╟─46f71c54-5f2d-11eb-3a79-c96ae093a6cc
# ╟─b4e0828e-6110-11eb-2cca-2bcb5a409caf
# ╟─5ecf3cf4-6116-11eb-11c5-db4253ba8d7f
# ╟─c8c5010c-5f6d-11eb-020f-b5a3fa043f1e
# ╟─dc91ea8c-5f6e-11eb-20ed-318a74d2f404
# ╠═ef273a10-5f6e-11eb-386e-4df51c71d0b5
# ╟─3d636042-61ff-11eb-1b22-9555285fe9af
# ╟─3e467742-61ff-11eb-3640-8f313ff08354
# ╟─adaf3546-72f4-11eb-0b21-e7466c2d81be
# ╠═4dedeecc-6246-11eb-00c7-014b87b08c32
# ╠═b9d7ddd8-624a-11eb-1084-35320b3f9afb
# ╠═b06e326c-72f6-11eb-204a-ef48d6cbf876
# ╟─7158aa22-72f5-11eb-246b-3bb16136c59c
# ╟─70e21632-72f5-11eb-341e-0f121cf27eac
# ╠═06038796-6234-11eb-3dd3-cf25a7095963
# ╠═41a9c7cc-6245-11eb-148b-3791b3fb504c
# ╟─faedfda0-72d7-11eb-0b80-7d63e962468d
# ╠═fce9afc0-624a-11eb-09e2-c38456a1fe35
# ╠═d03c2ac6-6253-11eb-0483-596dd3d5e5a4
# ╠═095be506-64e5-11eb-3ac8-6dbf5a7f5f9e
# ╠═08169170-64e3-11eb-3fbe-6b50b31ee02f
# ╟─fe16d518-64e7-11eb-04f5-bb25ed0a9eea
# ╠═212d911a-7dc3-11eb-11ee-333220a641e5
# ╠═9ff7af7e-7dc2-11eb-17b8-e7fe576888c4
# ╠═65a0f08e-7dc4-11eb-1676-8ba451a5574a
# ╠═6b63b052-64eb-11eb-1a62-33262062ece1
# ╟─8ef3c6e0-72f3-11eb-3244-1f406e6bde18
# ╟─629a9616-625c-11eb-0e76-536b5de36ab7
# ╟─2fd2a6c8-6256-11eb-2b61-1deb1e2e4c77
# ╠═210393f2-65ad-11eb-3dc0-0bcab1b97c73
# ╠═5ad541b0-64eb-11eb-0782-a59689a23af5
# ╠═7c8dd648-7dad-11eb-2d5e-69d011342fe8
# ╟─6bde2f84-6258-11eb-0e07-af0a2275fd79
# ╟─7545065e-72e7-11eb-1db0-3df6683bcbeb
# ╠═b1c556a8-72e3-11eb-1299-8b52ae0c19b7
# ╠═5e9936fa-72e5-11eb-078f-bd9e193eda1a
# ╠═d8cbc6e4-7dbd-11eb-378e-8bf7a5d244f2
# ╟─43b40838-72f4-11eb-068b-b7152a884e66
# ╠═e27d74fe-6e6c-11eb-08d5-b988732170d0
# ╟─cd720bf4-7dad-11eb-0e06-3fc6fa6c91c9
# ╠═93dccf8a-7dac-11eb-25f8-cd19f446b7a3
# ╠═168b4c16-7dc8-11eb-3bbe-9d52f6558742
# ╠═3391ed90-7dae-11eb-054d-db7a314ab852
# ╠═76344974-7db4-11eb-19c0-3b2116705950
# ╠═0743b716-7db4-11eb-1fcd-13f95d1daf8b
# ╠═3dc4eab8-7dbd-11eb-11a5-d5b1c33b5541
# ╠═5fd3174a-7dbf-11eb-22c0-13725c493e74
# ╠═a7cda810-7dbc-11eb-2434-31d93e7651e0
# ╠═0190c7c0-7dc6-11eb-2f0a-3dcdd3e7fa2c
# ╠═64b90c6c-7dc7-11eb-2de4-fd1f990d6860
# ╟─f74d6a68-61e9-11eb-0ed8-8bdd85177922
# ╟─f195457c-7dce-11eb-1326-83ed59d18879
# ╟─552aa3e0-7dd2-11eb-399e-ad5fc208fbc5
# ╠═d5a18c24-7dd1-11eb-30d4-6dcb0c8c9c4e
# ╟─5056efdc-7dd6-11eb-21d3-e13768d765d9
# ╟─fa5d996e-7dd5-11eb-3010-8935856e0b68
