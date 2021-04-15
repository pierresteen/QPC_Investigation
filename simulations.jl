### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 9cabcdd6-5e68-11eb-0613-9785eb761d6d
begin
	using CSV
	using Plots
	using Images
	using PlutoUI
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

# ╔═╡ f43c9da7-397f-49ec-a280-499c857c807c
TableOfContents()

# ╔═╡ ca57a27e-61cd-11eb-0057-a7a89cb2f828


# ╔═╡ 8ce21aca-5cad-11eb-0d3e-53ee628dd525
md"""
## The QPC device

When a negative voltage split-gate is applied on top of a 2DEG; the conductance of the QPC, in the unconfined electron momentum dimension, was found to be quantised in multiple of:

$G_0 = \frac{2e^2}{h}$

**This is classic result associated with ballistic transport of electrons across a one-dimensional channel**.
The figure below illustrates the heterojunction geometry used to achieve this, notice that the system can sufficiently be described by 2 dimensions globally and behaves as a quasi-one-dimensional system in the constriction channel.

![Imgur](https://imgur.com/h1KrE1H.png)


Bias potentials $V_{s}$ and $V_{d}$, applied to the source and drain respectively,  create a potential difference $V_{sd}$, which is dropped along the path between the terminals $S$ and $D$.
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

# ╔═╡ dc91ea8c-5f6e-11eb-20ed-318a74d2f404
md"## Constants:"

# ╔═╡ ef273a10-5f6e-11eb-386e-4df51c71d0b5
begin
	const e 	= -1.602176634e-19 	# (C)
	const h 	= 6.62607015e-34 	# (Js)
	const ħ 	= 1.054571817e-34 	# (Js)
	const h_eV 	= abs(h/e) 		 	# (eVs)
	const ħ_eV 	= abs(ħ/e) 			# (eVs)
	const ε_0	= 8.8541878128e-12	# (Fm^-1)
end;

# ╔═╡ 3d636042-61ff-11eb-1b22-9555285fe9af


# ╔═╡ f74d6a68-61e9-11eb-0ed8-8bdd85177922
md"""
## Non-equilibrium Green function formalism (NEGF)

The non-equilibrium Green function formalism approach to describing quantum transport in a channel which can be crossed ballistically, revolves around the following relationship:

```math
E[\psi] = [H][\psi] + [\Sigma][\psi] + [s]
```

which in fact a compact form of the following matrices:

![NGEF1](https://imgur.com/vBMNpYr.png)

Here, $H$ is the Hamiltonian operator and $\Sigma$ is the self-energy matrix of the connecting leads (drain and source).

Import for us to remember, is that ``s`` is the __scattering matrix__.

We will be using ``s`` along with numerically generated Hamiltonian operators to solve for the wavefuntions of the system and therefore characterise the quantum transport, finally solving for the conductance ``G`` (more on this later).
"""

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
|		...	L-4	L-3	L-2	L-1  N 	L+1 L+2 L+3 L+4 ...
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
"""

# ╔═╡ adaf3546-72f4-11eb-0b21-e7466c2d81be
md"""
# Simulation
## Custom Data Types
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
"""
**Solution parameter type**
"""
struct Sys_sol
	G
	τ
	α
	r
	β
end;

# ╔═╡ 3cf41550-834e-11eb-1997-99d861892e35
md"""
## System 'Constructor' Functions 

### Transfer Matrices

The matrix ``T`` operates on discrete wave functions as:

```math
\begin{bmatrix}
	\psi_{n+1}\\
	t\psi_{n}\\
\end{bmatrix}
= T
\begin{bmatrix}
	\psi_{n}\\
	t\psi_{n-1}\\
\end{bmatrix}
```
where:
```math
T =
\begin{bmatrix}
	t^{-1}H & -1\\
	t & 0\\
\end{bmatrix}
```

Transfer matrices can be evaluated locally and for the entire system without the need for iteration over all nodes.
We can subdivide the transfer matrix into its sub-elements as:

```math
T = 
\begin{bmatrix}
t_{11} & t_{12}\\
t_{21} & t_{22}\\
\end{bmatrix}
```
where:
```math
t_{11} = - iI + \frac{1}{2} H
\quad\quad
t_{12} = - \frac{1}{2} iH 
```
```math
t_{21} = - \frac{1}{2} iH
\quad\quad
t_{22} = iI + \frac{1}{2} H
```

and the Hamiltonian of the system is given by:

```math
H = -t I_{-1} + (4 - \mu) \epsilon I_{0} + -t I_{1}
```
```math
H =
\begin{bmatrix}
4t - \mu & -t & 0 & \dots\\
-t & 4t - \mu & -t & \dots\\
0 & -t & 4t - \mu & \dots\\
\vdots & \vdots & \vdots & \ddots\\
\end{bmatrix}
```
where ``t = 1``, the coupling constant, ``\epsilon = (4 - \mu)``
```math
I_{-1} =
\begin{bmatrix}
0 & 0 & 0 & \dots\\
1 & 0 & 0 & \dots\\
0 & 1 & 0 & \dots\\
\vdots & \vdots & \vdots & \ddots\\
\end{bmatrix}
\quad
I_{0} =
\begin{bmatrix}
1 & 0 & 0 & \dots\\
0 & 1 & 0 & \dots\\
0 & 0 & 1 & \dots\\
\vdots & \vdots & \vdots & \ddots\\
\end{bmatrix}
\quad
I_{1} =
\begin{bmatrix}
0 & 1 & 0 & \dots\\
0 & 0 & 1 & \dots\\
0 & 0 & 0 & \dots\\
\vdots & \vdots & \vdots & \ddots\\
\end{bmatrix}
```

__Example:__

For ``N = 4`` and ``\mu = 1``, the transfer matrix ``T`` is given by: 
```math
T =
\begin{bmatrix}
1.5-1.0i & -0.5-0.0i & -1.5+0.0i & 0.5+0.0i\\
-0.5-0.0i & 1.5-1.0i & 0.5+0.0i & -1.5+0.0i\\
-1.5+0.0i & 0.5+0.0i & 1.5+1.0i & -0.5+0.0i\\
0.5+0.0i & -1.5+0.0i & -0.5+0.0i & 1.5+1.0i\\
\end{bmatrix}
```
"""

# ╔═╡ c2559046-8e3b-11eb-3061-5dee42c3e621
"""
	H_mod()

Builds a Hamiltonian (tight-binding model) for the QPC system.
Used later in generating the transfer matrix of the system and the numerical algorithm for solving the quantum transport of the system.
"""
function H_mod(N, μ, t)
	v = ones(Float64, N-1)	# (N-1)-length array of Float 1s
	H = diagm(-1 => -v) + diagm(0 => (4 * t) * ones(Float64, N) .- μ) + diagm(1 => -v)
	
	return H
end

# ╔═╡ 06038796-6234-11eb-3dd3-cf25a7095963
"""
	T(μ, N)

Generates a diagonal tranfer data type `T::T_data` for given bias potentail: `μ`.
`T` is a `2N`x`2N` matrix, with three main diagonals at `diagind[1] = 0, 5, -5`.
"""
function T(μ, N)
	# create complex float type matrix with 1im diagonal values
	im_mat = zeros(Complex{Float64}, N, N)
	im_mat[diagind(im_mat)] .= 1im

	# create tight-binding Hamiltonian model
	H = H_mod(N, μ, 1)
	
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

# ╔═╡ d15c21b8-8350-11eb-0a17-916ab9ab4c48
md"""
### Scattering Matrices

This matrix relates the initial state to the final state by:

```math
\begin{bmatrix}
	B\\
	C\\
\end{bmatrix}
=
\begin{bmatrix}
	S_{11} & S_{12}\\
	S_{21} & S_{22}\\
\end{bmatrix}
\begin{bmatrix}
	A\\
	D\\
\end{bmatrix}
```

Where ``S`` is the complete scattering matrix in the relation:

```math
S =
\begin{bmatrix}
	S_{11} & S_{12}\\
	S_{21} & S_{22}\\
\end{bmatrix}
\quad
\Psi_{out} =
\begin{bmatrix}
	B\\
	C\\
\end{bmatrix}
\quad
\Psi_{in} =
\begin{bmatrix}
	A\\
	D\\
\end{bmatrix}
```
```math
\Psi_{out} = S \Psi_{in}
```
"""

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
end

# ╔═╡ faedfda0-72d7-11eb-0b80-7d63e962468d
md"""
Reference to: `sum_S(Sa, Sb)` function, see equation §B6:

[Calculation of the conductance of a graphene sheet using the Chalker-Coddington network model](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.78.045118).
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
end

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
end

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
end

# ╔═╡ b3fb891c-8d83-11eb-31d8-3fea8e634889
md"""
## System Energy Calculations
"""

# ╔═╡ 212d911a-7dc3-11eb-11ee-333220a641e5
"""
	meshgrid(x, y)

Generates a 2D meshgrid, same functionality as MATLAB meshgrid function.
"""
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# ╔═╡ 9ff7af7e-7dc2-11eb-17b8-e7fe576888c4
"""
	smooth_potential_broken(μ, N, L, xL=1.,yL=1., amp=1.)

Generates a smooth saddle-point potential profile for system ef dimensions `width = N` and `length = L`.
"""
function smooth_potential(μ, N, L, xL=1., yL=1., amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	
	return (-0.5 * amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
end

# ╔═╡ bba2787c-8db9-11eb-0566-7d3b8f46e43d
# surface(smooth_potential(0.5, 40, 100, 1., 1., 1.),
# 		#ticks=false,
# 		xlabel="X",
# 		ylabel="Y",
# 		zlabel="Z")

# ╔═╡ a1f94578-8d84-11eb-1de6-03bab5d1e34e
md"""
### Scattering Error
"""

# ╔═╡ 6b63b052-64eb-11eb-1a62-33262062ece1
"""
	error_ϵ(S::S_data, T::T_data)

Method which evaluates the model error from `S:S_data` and `T::T_data`.
"""
function error_ϵ(S::S_data)
	return norm(S.self * conj(S.self') - UniformScaling(1.))
end

# ╔═╡ deb49ea2-8d85-11eb-34ed-7b71e4b3cef8


# ╔═╡ 2fd2a6c8-6256-11eb-2b61-1deb1e2e4c77
md"""
## Numerical Transport Calculation

The matrix $S$  is calcuated by solving the following system of equations:

$[U_{R,out}\quad U_{R,ev} \quad - S_{tot}\cdot U_{out} \quad - S_{tot}\cdot U_{ev}][t\quad \alpha\quad r\quad \beta]^T = S_{tot}\cdot U_{in}$

For each propagation mode $i$, the sum of transmissions and reflections must sum to unity.
This is condition we enforce to obtain solutions:

$\Sigma^{N}_{j=1} |r_{ij}|^2 + |s_{ij}|^2$

The conductance of the system is given by Landauer's eq.:

$G = G_0\ \Sigma_{i,j} |s_{ij}|^2$

"""

# ╔═╡ c2f6b348-8d84-11eb-2b07-d585477a2f50
md"""
### Numerical Solve Block Functions
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

# ╔═╡ ce242db8-8d84-11eb-1f4d-532062e2cb6d
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
	# which are a numerical representation of the system's wave fucntions
	ψ_R = λ_vec[:, Rᵢ]
	ψ_L = λ_vec[:, Lᵢ]
	E_G = λ_vec[:, Eᵢ][:, Gᵢ]
	E_D = λ_vec[:, Eᵢ][:, Dᵢ]
	
	return ψ_R, ψ_L, E_G, E_D
end

# ╔═╡ 76e232ba-8d85-11eb-1e66-d7243264b5ed
"""
	build_Uᵣ(N, Sₜ, ψᵣ)

Builds the right-hand side terms of the NEGF form. 
"""
function build_Uᵣ(N, Sₜ, ψᵣ)
	Uᵣ_top = Sₜ.s_11 * ψᵣ[1:N, :] - ψᵣ[(N+1):(2*N), :]
	Uᵣ_bot = Sₜ.s_21 * ψᵣ[1:N, :]

	return vcat(Uᵣ_top, Uᵣ_bot)
end

# ╔═╡ 8c858216-8d85-11eb-27d5-710e5153ba7a
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

# ╔═╡ 2954131a-8d85-11eb-1862-bdef8e49a509
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

# ╔═╡ 7186dc7e-8d85-11eb-3429-3bbc1f4ab65b
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

# ╔═╡ c3d2dafc-8d85-11eb-1927-0ffa6df786db
md"""
## Numerical Solve Algorithm
"""

# ╔═╡ 0c4227d0-9069-11eb-2ffc-9b8d4d9dc3cf
md"""
## Poster Results
"""

# ╔═╡ 47f47e16-8d87-11eb-2734-fbe36fd94431
begin
	N = 40
	L = 100
	μ = 0.5
end;

# ╔═╡ 0a306d9c-8d85-11eb-3ceb-737958085066
"""
	ψ_norms(ψR, ψL)

Calculates and applies norming factors to wavefunctions `ψ_R` and `ψ_L`.
"""
function ψ_norms(ψR, ψL)
	# evaluate wave function norms $ψₙ_R & $ψₙ_L
	ψₙ_R = norm(ψR[(N+1):2*N]).^2 - norm(ψR[1:N]).^2
	ψₙ_L = norm(ψL[1:N]).^2 - norm(ψL[(N+1):2*N]).^2
	
	# apply norming factors to wave funtions
	ψ_R = ψR ./ √(abs(ψₙ_R))
	ψ_L = ψL ./ √(abs(ψₙ_L))
	
	return ψ_R, ψ_L
end

# ╔═╡ cffd655e-8d85-11eb-262c-c35e8d38a7d1
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
	ψ_R, ψ_L = ψ_norms(ψ_R, ψ_L)
	
	# form system of equation right- and left-hand side terms
	Uₗ = build_Uₗ(N, S_T, ψ_R, ψ_L, E_G, E_D)
	Uᵣ = build_Uᵣ(N, S_T, ψ_R)
	
	# evaluate coefficient and store in matrix form
	coeff, num_I, num_E = system_intermediary(Uᵣ, Uₗ, ψ_L, E_D)
	
	# evaluate system solutions
	solution_params = resolve(coeff, num_I, num_E)
	
	return solution_params
end

# ╔═╡ 754fd72a-8f2b-11eb-381b-c19ea1fed40a
begin
	V_test = smooth_potential(0.4, 40, 100, 1., 1., 0.5)
	system_solve(0.4, V_test, 40, 100, 0.4)
end

# ╔═╡ 96452cb8-9071-11eb-191e-434f5df7af07
# savefig(fig_clean, "dense_clean_quantised_G.png") # uncommment to save plot above

# ╔═╡ f0f515f7-a11b-4ec3-b882-92b934e5d87a
# savefig(fig_impurity, "dense_dirty_quantised_G.svg") # uncomment to save plot above

# ╔═╡ 6af5daff-fc93-443e-ae0a-002f7b943e5b
md"""
The graphs above, especially the impurity conductance trace, seems too cluttered, this prevents us being able to really distinguish between traces and observe the oscillations in the conductance.

__To rectify this -- plot traces in steps of ``10`` from `G_im`.__
"""

# ╔═╡ c6a7a888-8506-442d-9234-ee38bf53cc1d
md"""
#### Impurity Testing
"""

# ╔═╡ cfecb1d8-8fc5-11eb-3d49-3f7edba7de29
function local_sin_imp(μ, N, L, imp_A, x_spread, y_spread, xtune, ytune, smooth_A, xL=1., yL=1.)
	px = Float64.(range(-(x_spread/2), (x_spread/2), length=Int((x_spread/N)*N)))
	py = Float64.(range(-(y_spread/2), (x_spread/2), length=Int((y_spread/L)*L)))
	
	X, Y = meshgrid(px, py)

	add_imp = - imp_A .* (cos.(xtune .* X) + cos.(ytune .* Y))
	nom_pot = smooth_potential(μ, N, L, xL, yL, smooth_A)
	
	xlower_i = Int(N/2 - x_spread/2)
	xupper_i = Int(N/2 + x_spread/2 - 1)
	ylower_i = Int(L/2 - y_spread/2)
	yupper_i = Int(L/2 + y_spread/2 - 1)
	
	nom_pot[xlower_i:xupper_i, ylower_i:yupper_i] .+= add_imp
	
	return nom_pot
end

# ╔═╡ 47f94af8-9071-11eb-04c5-2d2fc8cf6035
function conductance_gen2(N, L, gate_min, gate_max, barrier_height, precision)
	G = []
	gate_energies = range(gate_min, gate_max, length=precision)
	
	for g_en in gate_energies
		V_en = local_sin_imp(g_en, 40, 100, -0.05, 20, 20, 0.5, 0.1, barrier_height, 1., 1.)
		push!(G, system_solve(g_en, V_en, N, L, g_en)[1])
	end
	
	return G, gate_energies
end

# ╔═╡ 971ea8f8-9071-11eb-0d85-b1186f4d3aea
begin
	G_im = []
	μ_im = []
	
	for b_height in 0.3:0.01:1.0
		G_loc, μ_loc = conductance_gen2(N, L, 0.1, 0.9, b_height, 200)
		push!(G_im, G_loc)
		push!(μ_im, μ_loc)
	end
end

# ╔═╡ f3987e2a-06d4-4f96-86ee-991c395226c8
begin
	gr()
	fig_impurity = plot(μ_im[1], G_im[1])
	
	for i in 2:length(G_im)
		plot!(μ_im[i], G_im[i])
	end
	
	plot!(
		# title & labels
		title="Impurity Simulated Quantised Conductance",
		xlabel="Potential μ",
		ylabel="Conductance (Gₒ)",
		# title & labels geometry
		titlefontsize=14,
		guidefontsize=14,
		tickfontsize=14,
		# annotations
		leg=false,
		# x & y axis
		yticks=0.0:1.0:10.0,
		xticks=0.0:0.1:1.0
	)
	fig_impurity
end

# ╔═╡ a758ef64-039d-4b6e-ac92-a01e4a41f4ee
length(G_im)

# ╔═╡ 36f78314-f54b-4334-85cc-7ebe9424cd17
begin
	gr()
	fig_space_impurity = plot(μ_im[1], G_im[1])
	
	for i in 4:5:length(G_im)
		plot!(μ_im[i], G_im[i])
	end
	
	plot!(
		# title & labels
		title="Impurity Spaced Simulated Quantised Conductance",
		xlabel="Potential μ",
		ylabel="Conductance (Gₒ)",
		# title & labels geometry
		titlefontsize=14,
		guidefontsize=14,
		tickfontsize=14,
		# annotations
		leg=false,
		# x & y axis
		yticks=0.0:1.0:10.0,
		xticks=0.0:0.1:1.0,
		# plot geometry
	)
end

# ╔═╡ 6108e688-e2e1-4548-96c3-a44080ba3b53
plotly(); plot(μ_im[30], G_im[30], yticks=0.0:1.0:10.0)

# ╔═╡ 3875774a-8d87-11eb-321a-0f74a8dc4c73
md"""
## Simulation Results
"""

# ╔═╡ d5a61ac2-9074-11eb-1948-934b10dcb64c
surf_clean = surface(smooth_potential(0.5, N, L, 1., 1., 0.6),
		title="Reference Potential Profile (Clean)",
		titlefontsize=18,
		xlabel="Y",
		ylabel="X",
		guidefontsize=15,
		tickfontsize=12)

# ╔═╡ 6b75a64e-9075-11eb-0ade-294cd8ee447a
savefig(surf_clean, "surf_clean.svg")

# ╔═╡ 878ec3c4-9075-11eb-0e07-a3696ce0e839
surf_dirty = surface(local_sin_imp(0.5, 40, 100, -0.05, 20, 20, 0.02, 0.1, 0.6, 1., 1.),
	title="Reference Potential Profile (Impurity)",
		titlefontsize=18,
		xlabel="Y",
		ylabel="X",
		guidefontsize=15,
		tickfontsize=12)

# ╔═╡ c53f3960-9075-11eb-32b3-17b4dd27707f
savefig(surf_dirty, "surf_dirty.svg")

# ╔═╡ ef295a00-8db7-11eb-1f02-db93c59bd24f
function conductance_gen(N, L, gate_min, gate_max, barrier_height, precision)
	G = []
	gate_energies = range(gate_min, gate_max, length=precision)
	
	for g_en in gate_energies
		V_en = smooth_potential(g_en, N, L, 1., 1., barrier_height)
		push!(G, system_solve(g_en, V_en, N, L, g_en)[1])
	end
	
	return G, gate_energies
end

# ╔═╡ 5205f976-906a-11eb-329b-c9937537fa31
# Clean quantised conductance
begin
	G_test, ens_test = conductance_gen(N, L, 0.2, 0.7, 0.5, 50)
	G_test2, ens_test2 = conductance_gen(N, L, 0.2, 0.7, 0.6, 50)
	plot(ens_test, G_test, leg=false)
	plot!(ens_test2, G_test2, leg=false)
end

# ╔═╡ 50f917b4-906c-11eb-1037-cfbbf1cb3bc0
begin
	G_t = []
	μ_t = []
	
	for b_height in 0.3:0.01:1.0
		G_loc, μ_loc = conductance_gen(N, L, 0.1, 0.9, b_height, 200)
		push!(G_t, G_loc)
		push!(μ_t, μ_loc)
	end
end

# ╔═╡ 34f22ce4-906d-11eb-28f5-4d84d9fecd6d
begin
	barrier_height = 0.3:0.01:1.0
	b1 = "μ = " * string(barrier_height[1])
	fig_clean = plot(μ_t[1], G_t[1], label=b1)
	for i in 2:length(G_t)
		μ_base = "μ = " * string(barrier_height[i])
		plot!(μ_t[i], G_t[i], label=μ_base)
	end
	plot!(
		# title & labels
		title="Clean Simulated Quantised Conductance",
		xlabel="Potential μ",
		ylabel="Conductance (Gₒ)",
		# title & labels geometry
		titlefontsize=14,
		guidefontsize=14,
		tickfontsize=14,
		# annotations
		leg=false,
		# x & y axis
		xticks=0.1:0.1:0.8,
		yticks=0.0:1.0:10.0
	)
	
	fig_clean
end

# ╔═╡ 602d89d4-8dbc-11eb-3095-553df225ff7d
begin
	G_traces = []
	for i in 0.2:0.2:0.8
		G_i, ens = conductance_gen(40, 100, 0.1, 1.5, i, 20)
		push!(G_traces, [G_i ens])
	end
end

# ╔═╡ f09e2df2-8e37-11eb-3537-ad7491c66146
md"""
## Simulating Impurity

So far, we have used a tight-binding Hamiltonian model, given by:
```julia
v = ones(Float64, N-1)	# (N-1)-length array of Float 1s
H = diagm(-1 => -v) + diagm(0 => 4*ones(Float64, N) .- μ) + diagm(1 => -v)
```

### Impurity Modified Potential Map 
"""

# ╔═╡ 91b4b744-8328-11eb-017b-6153bb61cfb2
"""
	place_impurity(grid_pot, loc_x, loc_y)

Generates a point charge impurity at `(loc_x, loc_y)` in an otherwise smooth potential  profile domain.
"""
function place_impurity(grid_pot, loc_x, loc_y, spread, A_imp)
	map_size = size(grid_pot)
	mesh_size = zeros(Float64, map_size[1], map_size[2])
	
	# check that impurity locations provided are within network graph
	@assert all((loc_x, loc_y) .<= map_size)
	@assert all((loc_x, loc_y) .>= (1,1))
	for i in 1:map_size[1], j in 1:map_size[2]
		r = sqrt((loc_x - i)^2 + (loc_y - j)^2)
		V = abs(1/(4*pi*ε_0) * (spread * e/r))
		V = A_imp * cos(spread * r)
		if V == Inf
			mesh_size[i,j] = 0.05
		else
			mesh_size[i,j] = V
		end
	end
	
	return grid_pot + mesh_size
end

# ╔═╡ 1c98b9ae-8fca-11eb-1fb4-2f5b5826949b
begin
	gr()
	p_smooth = surface(place_impurity(smooth_potential(0.4, N, L, 1., 1., 0.6), 20, 50, 0.01, 0.1),
			title="QPC saddle-point potential profile",
			titlefontsize=18,
			xlabel="Y",
			ylabel="X",
			zlabel="E")
end

# ╔═╡ 7bf5e4d2-905e-11eb-324d-674214dcb962
savefig(p_smooth, "p_smooth.svg")

# ╔═╡ 0b8b2316-8fc7-11eb-25af-930765cb3f88
begin
	gr()
	surface(local_sin_imp(0.4, 40, 100, -0.05, 20, 20, 0.1, 0.1, 0.5, 1., 1.),
			title="QPC modified saddle-point potential profile")
end

# ╔═╡ fe42a016-9029-11eb-2bfa-2f90a7a38816
begin
	G3 = []
	energies3 = range(0.1, 0.5, length=100)
	for en in energies3
		#V = impurity_potential(en, N, L, 0.0001, 10, 10, 1., 1.)
		V3 = local_sin_imp(0.4, N, L, -0.05, 20, 20, 0.1, 0.5, en, 1., 1.)
		push!(G3, system_solve(en, V3, N, L, en)[1])
	end
end

# ╔═╡ ed59c378-9054-11eb-2090-0f8beb23b855
begin
	G4 = []
	energies4 = range(0.1, 0.8, length=200)
	for en in energies4
		#V = impurity_potential(en, N, L, 0.0001, 10, 10, 1., 1.)
		V4 = smooth_potential(μ, N, L, 1., 1., 0.5)
		push!(G4, system_solve(en, V4, N, L, en)[1])
	end
end

# ╔═╡ fda8c6a4-9068-11eb-0b67-0b0d9a0da15d
surface(smooth_potential(μ, N, L, 1., 1., 0.5))

# ╔═╡ 3df5e43a-902b-11eb-1501-4dd67f1bc4b0
begin
	surface(local_sin_imp(μ, N, L, -0.05, 20, 20, 0.5, 0.1, 0.5, 1., 1.))
end

# ╔═╡ 55c634ea-8f82-11eb-259a-3b7f881c4d80
function impurity_potential3(μ, N, L, tune_y, sub_y, smooth_A, xL=1., yL=1.)
	smp = smooth_potential(μ, N, L, xL, yL, smooth_A)
	imp = impurity_potential2(μ, N, L, tune_y, sub_y, xL, yL)
	
	return smp + imp
end

# ╔═╡ 69e57db4-8f2f-11eb-04fa-052bf0433dea
function impurity_potential(μ, N, L, A, xtune, ytune, smooth_A, xL=1., yL=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)

	add_imp = - A .* (cos.(xtune .* X) + cos.(ytune .* Y))
	
	return smooth_potential(μ, N, L, xL, yL, smooth_A) + add_imp
end

# ╔═╡ fd1585d4-8f2f-11eb-21c5-f96dc290073c
begin
	gr()
	surface(impurity_potential(μ, N, L, 0.1, 20, 20, 0.6, 1., 1.),
			title="QPC noisy saddle-point potential profile")
end

# ╔═╡ 83d9e9fa-8fc3-11eb-2a8a-f90d743fd9b4
function local_impurity(b, x_loc, x_mod, y_loc, y_mod, xL=1., yL=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	
	local_imp = b .* (((X ./ x_mod) .- x_loc).^2 + ((Y ./ y_mod) .- y_loc).^2)
	
	return local_imp
end

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
### Data Import:
"""

# ╔═╡ 3fd81744-8d87-11eb-3366-3d51632a9043
begin
	G = []
	energies = range(0.3, 0.6, length=50)
	for en in energies
		#V = impurity_potential(en, N, L, 0.0001, 10, 10, 1., 1.)
		V = smooth_potential(en, N, L, 1., 1., 0.6)
		push!(G, system_solve(en, V, N, L, en)[1])
	end
end

# ╔═╡ 53ddbc9c-8d87-11eb-050e-11c509947dbf
plot(energies, G, leg=false)

# ╔═╡ 97ea8df4-8f47-11eb-1f43-1771fffdbf0d
begin
	G2 = []
	energies2 = range(0.01, 0.5, length=100)
	#energies2 = 0:0.0014028056112224449:0.6
	for en in energies2
		#V = impurity_potential3(en, N, L, 0.1, 0.3, 0.6, 1., 1.)
		#V = local_sin_imp(μ, N, L, -0.05, 20, 50, 0.1, 0.1, 0.6, 1., 1.)
		V = place_impurity(smooth_potential(en, N, L, 1., 1., 0.6), 20, 50, 0.01)
		push!(G2, system_solve(en, V, N, L, en)[1])
	end
end

# ╔═╡ b6128210-8fc8-11eb-0e3d-a11dc7195e78
surface(
	place_impurity(
		smooth_potential(
			0.4,
			40, 
			100,
			1.,
			1.,
			0.6
		),
		20,
		50,
		0.01
	)
)

# ╔═╡ 44e76071-4ced-450f-b815-4ee277570980
smooth_potential(0.4, N, L, 1., 1., 0.6)

# ╔═╡ a22e007a-8f47-11eb-3af2-e9ce0a3eafbe
plot(energies2, G2, leg=false)

# ╔═╡ 854b06e2-87fc-11eb-1d4a-058609b638d3
md"""
## Aims and Objectives:
__I want to be able to:__
1. plot simulated 'clean' conductance of a QPC alongside the experimental results
   - deduce the errors in the simulated 'clean' results from the experimental ones
   - introduce compensation term into 'clean' sim to _tune_ results for accuracy
2. plot simulated 'noisy' conductance of a QPC alongside the experimental results
   - deduce the errors in the simulated 'noisy' results from the experimental ones
   - introduce compensation term into 'noisy' sim to _tune_ results for accuracy
"""

# ╔═╡ 8e92f6f4-8353-11eb-30ca-c3d79597943a


# ╔═╡ e63f0386-8353-11eb-0334-3b77fcc24f3a
md"""
## Electron Energy in Confined Systems

In a system with in-plane dispersion in the xy-plane, and confinement in the z-axis, as illustrated below:

$(load("./mdsrc/ipd.png"))

the in-plane wavefunctions of an electron: ``\psi_{x,y}`` will solve (IIF it is assumed that the plan is infite) to reflect the current flow properties as:

```math
\psi_{x,y}(x,y) = \frac{1}{A}\exp[i(k_{x} x + k_{y} y)]
```

In the restricted dimension, along the z-axis, the wavefunction ``\psi_{z}`` takes the form of a sinusoid when cosidering the confining potential as perfect i.e infinite outside the plane and zero inside the plane gap.
Further consideration of the boundary problem refines the solution to:

```math
\psi_{n}(z) = \sqrt{\frac{2}{l_w}}\sin\left(\frac{\pi n z}{l_w}\right)
```

Observe how the wavefunction in the confined dimension has become discretised in the quantum number ``n``.

We can also find the energy components in the ``x`` and ``y`` dimensions as:

```math
E_{x} = \frac{\hbar^2 k_{x}^2}{2m}
```
```math
E_{y} = \frac{\hbar^2 k_{y}^2}{2m}
```

and for the confined dimension:

```math
E_{z} = E_{n} = \frac{\hbar^2 \pi^2 n^2}{2 m l_w^2}.
```

This results in a total system (electron) energy given as:

```math
E = E_{n} + \frac{\hbar^2 | \vec{k}_{x,y} |^2}{2m^*}
```

in accordance to the effective mass theorem which gives us the electron mass in the material as ``m^*``.

"""

# ╔═╡ f6c12ba2-8d86-11eb-3060-77aa104ab877


# ╔═╡ 7545065e-72e7-11eb-1db0-3df6683bcbeb
md"""
# Debugging Utils

This section contains functions that were built and used in order to debug the above algorthims, using imported and converted python generated results as ground-truths to test against.
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

__Complex *matrix* debug import tool!__

Maps an imported `Array{String,2}` type, from numpy via csv intermediary, cleaning and parsing, to an `Array{Complex{Float64},2}` type.

`file_dir::String ->` location of csv stored numpy array.
"""
function csvcomplexparse(file_dir::String)
	strs = CSV.File(file_dir, header=0) |> Tables.matrix |> x -> map(complexclean, x)
	cmps = parse.(Complex{Float64}, strs)
	
	return cmps
end

# ╔═╡ a6f19760-8d81-11eb-2e7e-6dae07c47af9
"""
	csvcomplexparse(file_dir::String)

__Complex *vector* debug import tool!__

Maps an imported `Array{String,1}` type, from numpy via csv intermediary, cleaning and parsing, to an `Array{Complex{Float64},1}` type.

`file_dir::String ->` location of csv stored numpy array.
"""
function csvcomplexparse2(file_dir::String)
	t_vec = CSV.File(file_dir, header=0) |> Tables.matrix
	strs = map(complexclean, t_vec[:,2])
	cmps = parse.(Complex{Float64}, strs)
	
	return cmps
end

# ╔═╡ d8cbc6e4-7dbd-11eb-378e-8bf7a5d244f2
"""
	csvfloatparse(file_dir::String)

__Floating point *matrix* debug import tool!__

`file_dir::String ->` location of csv stored numpy array.
"""
function csvfloatparse(file_dir::String)
	return CSV.File(file_dir, header=0) |> Tables.matrix
end

# ╔═╡ d5a18c24-7dd1-11eb-30d4-6dcb0c8c9c4e
begin
	cleanV = csvfloatparse("data/splitgateV_clean.csv")
	cleanG = csvfloatparse("data/conductance_clean.csv")

	noisyV = csvfloatparse("data/splitgateV_noisy.csv")
	noisyG = csvfloatparse("data/conductance_noisy.csv")
end;

# ╔═╡ 5056efdc-7dd6-11eb-21d3-e13768d765d9
clean_plot = plot(
	cleanV[1:(size(cleanV)[1]),1:(size(cleanV)[2])],
	cleanG[1:(size(cleanG)[1]), 1:(size(cleanG)[2])],
	title="Channel Conductance of a Clean QPC",
	xlabel="Split Gate Voltage (V)",
	ylabel="Channel Conductance G (μS)",
	leg=false
)

# ╔═╡ fa5d996e-7dd5-11eb-3010-8935856e0b68
noisy_plot = plot(noisyV[1:(size(noisyV)[1]), 1:(size(noisyV)[2])],
				noisyG[1:(size(noisyG)[1]), 1:(size(noisyG)[2])],
				title="Channel Conductance of a QPC Containing Impurities",
				xlabel="Split Gate Voltage (V)",
				ylabel="Channel Conductance G (μS)",
				leg=false)

# ╔═╡ f7b17256-8331-11eb-1542-0d28cdf6478f
# scale conductance plots by G_0
begin
	G_0 = (2 * e^2) / h
	
	cleanG_scaled = (cleanG .* 1e-6) ./ G_0
	noisyG_scaled = (noisyG .* 1e-6) ./ G_0
end;

# ╔═╡ 3988a6f4-8332-11eb-0bbc-61d88181a812
begin
clean_plot2 = plot(cleanV[1:(size(cleanV)[1]),1:(size(cleanV)[2])],
				cleanG_scaled[1:(size(cleanG)[1]), 1:(size(cleanG)[2])])
	plot!(title="Clean Experimental Quantised Conductance",
		titlefontsize=18,
		xlabel="Split Gate Voltage (V)",
		ylabel="Conductance (Gₒ)",
		guidefontsize=16,
		tickfontsize=14,
		yticks=0.0:1.0:10.0,
		leg=false,
		legendfontsize=13)
	savefig(clean_plot2, "clean_exp.svg")
	@show clean_plot2
end

# ╔═╡ 561742b2-8332-11eb-0c5f-d14c9b23709c
begin
	noisy_plot2 = plot(
		noisyV[1:(size(noisyV)[1]), 1:(size(noisyV)[2])],
		noisyG_scaled[1:(size(noisyG)[1]), 1:(size(noisyG)[2])],
		title="Impurity Experimental Quantied Conductance",
		titlefontsize=18,
		xlabel="Split Gate Voltage (V)",
		ylabel="Channel Conductance (Gₒ)",
		guidefontsize=16,
		tickfontsize=14,
		yticks=0.0:1.0:10.0,
		leg=false,
		legendfontsize=13
	)
	savefig(noisy_plot2, "dirty_exp.svg")
		@show noisy_plot2
end

# ╔═╡ 9bb30ee2-b6de-431e-8e7d-3761a2084fe4


# ╔═╡ Cell order:
# ╟─7ee2fb54-433c-11eb-1f9b-3528ac7148a4
# ╟─e616e6b0-61fe-11eb-398b-4fde45cba90f
# ╟─3ab951aa-5f2d-11eb-24d3-9d64610bf050
# ╠═9cabcdd6-5e68-11eb-0613-9785eb761d6d
# ╠═f43c9da7-397f-49ec-a280-499c857c807c
# ╟─ca57a27e-61cd-11eb-0057-a7a89cb2f828
# ╟─8ce21aca-5cad-11eb-0d3e-53ee628dd525
# ╟─46f71c54-5f2d-11eb-3a79-c96ae093a6cc
# ╟─b4e0828e-6110-11eb-2cca-2bcb5a409caf
# ╟─dc91ea8c-5f6e-11eb-20ed-318a74d2f404
# ╠═ef273a10-5f6e-11eb-386e-4df51c71d0b5
# ╟─3d636042-61ff-11eb-1b22-9555285fe9af
# ╟─f74d6a68-61e9-11eb-0ed8-8bdd85177922
# ╟─3e467742-61ff-11eb-3640-8f313ff08354
# ╟─adaf3546-72f4-11eb-0b21-e7466c2d81be
# ╠═4dedeecc-6246-11eb-00c7-014b87b08c32
# ╠═b9d7ddd8-624a-11eb-1084-35320b3f9afb
# ╠═b06e326c-72f6-11eb-204a-ef48d6cbf876
# ╟─3cf41550-834e-11eb-1997-99d861892e35
# ╠═c2559046-8e3b-11eb-3061-5dee42c3e621
# ╠═06038796-6234-11eb-3dd3-cf25a7095963
# ╟─d15c21b8-8350-11eb-0a17-916ab9ab4c48
# ╠═41a9c7cc-6245-11eb-148b-3791b3fb504c
# ╠═faedfda0-72d7-11eb-0b80-7d63e962468d
# ╠═fce9afc0-624a-11eb-09e2-c38456a1fe35
# ╠═d03c2ac6-6253-11eb-0483-596dd3d5e5a4
# ╠═095be506-64e5-11eb-3ac8-6dbf5a7f5f9e
# ╟─b3fb891c-8d83-11eb-31d8-3fea8e634889
# ╠═212d911a-7dc3-11eb-11ee-333220a641e5
# ╠═9ff7af7e-7dc2-11eb-17b8-e7fe576888c4
# ╠═bba2787c-8db9-11eb-0566-7d3b8f46e43d
# ╠═a1f94578-8d84-11eb-1de6-03bab5d1e34e
# ╠═6b63b052-64eb-11eb-1a62-33262062ece1
# ╟─deb49ea2-8d85-11eb-34ed-7b71e4b3cef8
# ╟─2fd2a6c8-6256-11eb-2b61-1deb1e2e4c77
# ╟─c2f6b348-8d84-11eb-2b07-d585477a2f50
# ╠═210393f2-65ad-11eb-3dc0-0bcab1b97c73
# ╠═ce242db8-8d84-11eb-1f4d-532062e2cb6d
# ╠═0a306d9c-8d85-11eb-3ceb-737958085066
# ╠═76e232ba-8d85-11eb-1e66-d7243264b5ed
# ╠═8c858216-8d85-11eb-27d5-710e5153ba7a
# ╠═2954131a-8d85-11eb-1862-bdef8e49a509
# ╠═7186dc7e-8d85-11eb-3429-3bbc1f4ab65b
# ╟─c3d2dafc-8d85-11eb-1927-0ffa6df786db
# ╠═cffd655e-8d85-11eb-262c-c35e8d38a7d1
# ╠═754fd72a-8f2b-11eb-381b-c19ea1fed40a
# ╟─0c4227d0-9069-11eb-2ffc-9b8d4d9dc3cf
# ╠═47f47e16-8d87-11eb-2734-fbe36fd94431
# ╟─5205f976-906a-11eb-329b-c9937537fa31
# ╠═50f917b4-906c-11eb-1037-cfbbf1cb3bc0
# ╠═34f22ce4-906d-11eb-28f5-4d84d9fecd6d
# ╠═96452cb8-9071-11eb-191e-434f5df7af07
# ╠═971ea8f8-9071-11eb-0d85-b1186f4d3aea
# ╠═f3987e2a-06d4-4f96-86ee-991c395226c8
# ╠═f0f515f7-a11b-4ec3-b882-92b934e5d87a
# ╟─6af5daff-fc93-443e-ae0a-002f7b943e5b
# ╠═a758ef64-039d-4b6e-ac92-a01e4a41f4ee
# ╠═36f78314-f54b-4334-85cc-7ebe9424cd17
# ╟─c6a7a888-8506-442d-9234-ee38bf53cc1d
# ╠═6108e688-e2e1-4548-96c3-a44080ba3b53
# ╠═47f94af8-9071-11eb-04c5-2d2fc8cf6035
# ╠═cfecb1d8-8fc5-11eb-3d49-3f7edba7de29
# ╟─3875774a-8d87-11eb-321a-0f74a8dc4c73
# ╠═d5a61ac2-9074-11eb-1948-934b10dcb64c
# ╠═6b75a64e-9075-11eb-0ade-294cd8ee447a
# ╠═878ec3c4-9075-11eb-0e07-a3696ce0e839
# ╠═c53f3960-9075-11eb-32b3-17b4dd27707f
# ╠═ef295a00-8db7-11eb-1f02-db93c59bd24f
# ╠═602d89d4-8dbc-11eb-3095-553df225ff7d
# ╠═3988a6f4-8332-11eb-0bbc-61d88181a812
# ╠═561742b2-8332-11eb-0c5f-d14c9b23709c
# ╠═f09e2df2-8e37-11eb-3537-ad7491c66146
# ╠═91b4b744-8328-11eb-017b-6153bb61cfb2
# ╠═1c98b9ae-8fca-11eb-1fb4-2f5b5826949b
# ╠═7bf5e4d2-905e-11eb-324d-674214dcb962
# ╠═fd1585d4-8f2f-11eb-21c5-f96dc290073c
# ╠═0b8b2316-8fc7-11eb-25af-930765cb3f88
# ╠═fe42a016-9029-11eb-2bfa-2f90a7a38816
# ╠═ed59c378-9054-11eb-2090-0f8beb23b855
# ╠═fda8c6a4-9068-11eb-0b67-0b0d9a0da15d
# ╠═3df5e43a-902b-11eb-1501-4dd67f1bc4b0
# ╠═55c634ea-8f82-11eb-259a-3b7f881c4d80
# ╠═69e57db4-8f2f-11eb-04fa-052bf0433dea
# ╠═83d9e9fa-8fc3-11eb-2a8a-f90d743fd9b4
# ╟─f195457c-7dce-11eb-1326-83ed59d18879
# ╟─552aa3e0-7dd2-11eb-399e-ad5fc208fbc5
# ╠═d5a18c24-7dd1-11eb-30d4-6dcb0c8c9c4e
# ╠═5056efdc-7dd6-11eb-21d3-e13768d765d9
# ╠═fa5d996e-7dd5-11eb-3010-8935856e0b68
# ╠═3fd81744-8d87-11eb-3366-3d51632a9043
# ╠═53ddbc9c-8d87-11eb-050e-11c509947dbf
# ╠═97ea8df4-8f47-11eb-1f43-1771fffdbf0d
# ╠═b6128210-8fc8-11eb-0e3d-a11dc7195e78
# ╠═44e76071-4ced-450f-b815-4ee277570980
# ╠═a22e007a-8f47-11eb-3af2-e9ce0a3eafbe
# ╠═f7b17256-8331-11eb-1542-0d28cdf6478f
# ╟─854b06e2-87fc-11eb-1d4a-058609b638d3
# ╟─8e92f6f4-8353-11eb-30ca-c3d79597943a
# ╟─e63f0386-8353-11eb-0334-3b77fcc24f3a
# ╟─f6c12ba2-8d86-11eb-3060-77aa104ab877
# ╟─7545065e-72e7-11eb-1db0-3df6683bcbeb
# ╠═b1c556a8-72e3-11eb-1299-8b52ae0c19b7
# ╠═5e9936fa-72e5-11eb-078f-bd9e193eda1a
# ╠═a6f19760-8d81-11eb-2e7e-6dae07c47af9
# ╠═d8cbc6e4-7dbd-11eb-378e-8bf7a5d244f2
# ╠═9bb30ee2-b6de-431e-8e7d-3761a2084fe4
