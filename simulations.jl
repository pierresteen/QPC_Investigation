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

Multiple dispatch for multiplying two objects `::T_data` composed of a `self` matrix and four sub-matrix blocks `T_ij`.
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
## Clean System Potentials
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

# ╔═╡ 0a306d9c-8d85-11eb-3ceb-737958085066
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
	
	count_in = size(ψ_L)[2]	# number of incident waves
	count_evan = size(E_D)[2]	# number of evanescent waves
	
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

# ╔═╡ e125a71d-37df-4f47-bebe-62ea8bccf3e2
function resolve_R(coeffs, count_in, count_ev)
	τ = coeffs[1:(count_in), :]
	α = coeffs[(count_in+1):(count_in+count_ev), :]
	r = coeffs[(count_in+count_ev+1):(2*count_in+count_ev), :]
	β = coeffs[(2*count_in+count_ev+1):(2*count_in+2*count_ev), :]
	
	R = norm(r)^2 # final coductance calculation
	
	return R
end

# ╔═╡ a3adb388-be9b-49ec-99c6-537e87c57cee
function system_solve_R(μ, V, N, L, i)
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
	solution_params = resolve_R(coeff, num_I, num_E)
	
	return solution_params
end

# ╔═╡ 973c91dd-58e7-4462-a792-85ad18eca925
md"""
⚠️ __Algorithm validity check__ ⚠️

Ok if number below = 2.97874...
"""

# ╔═╡ 754fd72a-8f2b-11eb-381b-c19ea1fed40a
begin
	V_test = smooth_potential(0.4, 40, 100, 1., 1., 0.5)
	system_solve(0.4, V_test, 40, 100, 0.4)[1]
end

# ╔═╡ f09e2df2-8e37-11eb-3537-ad7491c66146
md"""
## Simulating Impurity

In this section we attempt to simulate impurities in the QPC device, in form of exited ions.
We will achieve this by producing a modified barrier potential to map to the network nodes.

Localised impurities should be somewhat well modelled by local periodic potentials, added to the hyperbolic paraboloid base.
"""

# ╔═╡ 3875774a-8d87-11eb-321a-0f74a8dc4c73
md"""
### Impurity System Potentials
"""

# ╔═╡ 69e57db4-8f2f-11eb-04fa-052bf0433dea
function impurity_potential(μ, N, L, A, xtune, ytune, smooth_A, xL=1., yL=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)

	add_imp = - A .* (cos.(xtune .* X) + cos.(ytune .* Y))
	
	return smooth_potential(μ, N, L, xL, yL, smooth_A) + add_imp
end

# ╔═╡ 442a3b4f-632d-4a2a-8813-912aa14c3031
md"""
#### Gaussian Impurity Function

Using a 2D gaussian function to model the potential dropped by the exited ionic impurity.
The barrier potential will be a sum of: the hyperbolic paraboloid and the gaussian potential.
We can control the location of the gaussian's maxima by setting the 'mean' for the `x = N` and `y = L` axes.

2D Gaussian function:

```math
\text{gaussian}(x,y) =
A \cdot
\exp\left(
	-\left(
		\frac{(x - x_0)^2}{2\sigma_x^2}
		+
		\frac{(y - y_0)^2}{2\sigma_y^2}
	\right)
\right)
```

"""

# ╔═╡ 2250cb4c-0065-4ce4-8c5a-205392be0909
"""
	gaussian_impurity_potential(μ, N, L; x_pos=(N/2), y_pos=(L/2), imp_amp=0.2, xL=1., yL=1., Vg_amp=1.)

Generates a impure saddle-point potential, with a gaussian pulse modelling an exited ionic impurity for system dimensions `width = N` and `length = L`.
"""
function gaussian_impurity_potential(μ, N, L; x_pos=(N/2), y_pos=(L/2), σ_x=0.2, σ_y=0.2, imp_amp=0.2, xL=1., yL=1., gate_amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	x_0 = px[Int(x_pos)]
	y_0 = py[Int(y_pos)]
	
	saddle = (-0.5 * gate_amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
	
	gaussian = imp_amp * exp.(-(((X .- x_0).^2 ./ (2 * σ_x^2)) + ((Y .- y_0).^2 ./ (2 * σ_y^2))))
	
	return gaussian .+ saddle
end

# ╔═╡ ef780500-363f-4f76-a59e-e4220afe344d
function gaussian(μ, N, L; x_pos=(N/2), y_pos=(L/2), σ_x=0.2, σ_y=0.2, imp_amp=0.2, xL=1., yL=1., gate_amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	x_0 = px[Int(x_pos)]
	y_0 = py[Int(y_pos)]
	
	saddle = (-0.5 * gate_amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
	
	gaussian = imp_amp * exp.(-(((X .- x_0).^2 ./ (2 * σ_x^2)) + ((Y .- y_0).^2 ./ (2 * σ_y^2))))
	
	return gaussian
end

# ╔═╡ 047a102e-3936-4269-93fb-d5530aaa495d
function tester(μ, N, L)
	px = Float64.(range(-1., 1., length=N))
	py = Float64.(range(-1., 1., length=L))
	
	X, Y = meshgrid(px, py)
	
	return Y.^2 - X.^2 .+ μ
end

# ╔═╡ e67b1c41-efb8-448a-86a7-9918af1df1f8


# ╔═╡ bf2b0abe-b2f8-4f88-a82f-22d839b33a27
surface(tester(0.4, 40, 100))

# ╔═╡ be491790-b3a8-42f2-8fbf-20d8a3d0c0f0
md"""
__Gaussian impurity - conductance and reflectance results__
"""

# ╔═╡ 3738fea4-522d-4160-a96d-c798ff0f4786
md"""
__Stronger Impurity simulation:__
"""

# ╔═╡ 6482750b-35b7-42da-94e4-a38e69a67bd6
md"""
__Clean experimental spaced --conductance and reflectance results__
"""

# ╔═╡ 57613f4f-9faf-4e9a-9951-8bd675edc72f


# ╔═╡ 0c4227d0-9069-11eb-2ffc-9b8d4d9dc3cf
md"""
## Simulated Conductance Results
In this section, we use all the components developed above to obtain numerical results for the quantised conductance of a quantum point contact.
"""

# ╔═╡ 65d696f4-d49a-41b4-aacd-4ac5be4de1ed
md"""
### Conductance Generators
These functions enable us to automate the creation of multiple conductance traces.
"""

# ╔═╡ ef295a00-8db7-11eb-1f02-db93c59bd24f
"""
	conductance_gen(N, L, gate_min, gate_max, barrier_height, precision)

Generate an array of quantised conductance results, simulated using a NEGF numerical algorithm.
Also returns the array of gate energies used in generating G, as specified by the user via: `gate_max`, `gate_min` and `barrier_height`.
"""
function conductance_gen(N, L, gate_min, gate_max, barrier_height, precision)
	G = []
	gate_energies = range(gate_min, gate_max, length=precision)
	
	for g_en in gate_energies
		V_en = smooth_potential(g_en, N, L, 1., 1., barrier_height)
		push!(G, system_solve(g_en, V_en, N, L, g_en)[1])
	end
	
	return G, gate_energies
end

# ╔═╡ e260588a-240c-49c4-aa16-02fb9c4fb741
"""
	conductance_gen3(N, L, gate_min, gate_max, barrier_height, precision)

Impure system, quantised conductance generating function.
"""
function conductance_gen3(N, L, gate_min, gate_max, barrier_height, precision)
	G = []
	gate_energies = range(gate_min, gate_max, length=precision)
	
	for g_en in gate_energies
		V_en = gaussian_impurity_potential(
			g_en, N, L;
			x_pos=20,
			y_pos=50,
			σ_x=0.1,
			σ_y=0.1,
			imp_amp=0.2,
			xL=1.,
			yL=1.,
			gate_amp=barrier_height
		)
		push!(G, system_solve(g_en, V_en, N, L, g_en)[1])
	end
	
	return G, gate_energies
end

# ╔═╡ 7ef77e84-c92c-4a21-9e44-8bcb9989c287
function conductance_gen4(N, L, gate_min, gate_max, barrier_height, precision)
	G = []
	gate_energies = range(gate_min, gate_max, length=precision)
	
	for g_en in gate_energies
		V_en = gaussian_impurity_potential(
			g_en, N, L;
			x_pos=20,
			y_pos=50,
			σ_x=0.2,
			σ_y=0.2,
			imp_amp=0.3,
			xL=1.,
			yL=1.,
			gate_amp=barrier_height
		)
		push!(G, system_solve(g_en, V_en, N, L, g_en)[1])
	end
	
	return G, gate_energies
end

# ╔═╡ 56777301-68ca-4840-b6a5-86f4bb3dc81a
md"""
#### Dense Conductance Plots

Commented out below are the densely simulated (high accuracy) conductance results.
They are left commented to avoid running a 2000 seconds long cell upon notebook start-up.
"""

# ╔═╡ 47f47e16-8d87-11eb-2734-fbe36fd94431
begin
	N = 40
	L = 100
	μ = 0.5
end;

# ╔═╡ d5a61ac2-9074-11eb-1948-934b10dcb64c
surf_clean = surface(smooth_potential(0.5, N, L, 1., 1., 0.6),
	title="Hyperbolic Paraboloid - Saddle-Point Potential Profile",
	titlefont=(13, "times"),
	xlabel="Y",
	ylabel="X",
	guidefont=(12,"times"),
	tickfont=(9,"times"),
	legendfontsize=12
)

# ╔═╡ 20312b33-55c7-4fb6-8fe6-4aa386fddb61
savefig(surf_clean, "clean_saddle_map.png")

# ╔═╡ fd1585d4-8f2f-11eb-21c5-f96dc290073c
begin
	gr()
	f_dumb = surface(
		impurity_potential(μ, N, L, 0.1, 20, 20, 0.6, 1., 1.),
		title="QPC Impurity Potential Mapping - Too Aggressive",
		titlefont=(13,"times")
	)
end

# ╔═╡ 5077c151-fa25-4312-9e93-d52e8301e1aa
savefig(f_dumb, "bad_impurity_pot.png")

# ╔═╡ 180d8892-8cd9-4d3f-bf69-8aeac076fc57
gauss_surf = surface(
	gaussian(μ, N, L),
	title="Gaussian Localised Impurity Potential Approximation",
	titlefont=(13, "times"),
	xlabel="Y",
	ylabel="X",
	guidefont=(12,"times"),
	tickfont=(9,"times"),
	legendfontsize=12
)

# ╔═╡ 88d9513a-c8ee-451b-ac18-11836c0877bb
savefig(gauss_surf, "gaussian_potential.png")

# ╔═╡ 753d2cfa-df10-4216-85df-7292ef44e692
impu_surf = surface(
	gaussian_impurity_potential(μ, N, L),
	title="Combined Impurity Saddle-Point Potential Approximation",
	titlefont=(13, "times"),
	xlabel="Y",
	ylabel="X",
	guidefont=(12,"times"),
	tickfont=(9,"times"),
	legendfontsize=12
)

# ╔═╡ 0857e99b-a08b-43d7-906d-32107c0977c9
savefig(impu_surf, "impurity_potential_example.png")

# ╔═╡ 015176cd-e00f-419b-aeae-a4e3db0e10da
begin
	gr()
	surface(
		gaussian_impurity_potential(
			μ, N, L;
			x_pos=20,
			y_pos=60,
			σ_x=0.05,
			σ_y=0.05,
			imp_amp=0.1
		)
	)
	# gaussian_impurity_potential(μ, N, L)
end

# ╔═╡ 9492ef77-6124-44f3-baad-039ec1ddd99c
G_gauss, μ_gass = conductance_gen3(N, L, 0.1, 0.9, 0.5, 200)

# ╔═╡ 5bb0a920-9906-43d9-9c9d-3d9ce44deacf
begin
	plot(
		μ_gass,
		G_gauss,
		lab="Vg=0.5",
		leg=:topleft
	)
end

# ╔═╡ 7e2fdc41-a1b5-4a44-8c87-08f15dc60e4b
begin
	μ_gaussian = []
	G_gaussian = []
	
	barrier_ens = 0.2:0.1:1.0
	
	for b_en in barrier_ens
		G_loc, μ_loc = conductance_gen3(N, L, 0.1, 0.9, b_en, 200)
		push!(G_gaussian, G_loc)
		push!(μ_gaussian, μ_loc)
	end
end

# ╔═╡ a1b4a714-88e9-43a9-a594-fbe401039663
## gaussian ionic impurity model conductance plot
begin
	using Plots.PlotMeasures
	
	fig_gauss_imp = plot(μ_gaussian[1], G_gaussian[1])
	
	for i in 2:length(G_gaussian)
		plot!(μ_gaussian[i], G_gaussian[i])
	end
	
	plot!(
		# title & labels
		title="Gaussian-approximated ionic impurity conductance",
		xlabel="Potential μ [eV]",
		ylabel="Conductance [Gₒ = (2e²/h)]",
		# title & labels geometry
		titlefont=(14,"times"),
		guidefont=(12,"times"),
		tickfont=(12,"times"),
		# annotations
		leg=false,
		# x & y axis
		yticks=0.0:1.0:10.0,
		xticks=0.0:0.1:1.0,
		bottom_margin=5mm
	)
end

# ╔═╡ 00fa5126-a4b2-49ec-82df-e09a0d888dca
savefig(fig_gauss_imp, "gaussian_impurity_conductance.png")

# ╔═╡ bd9b0c3c-4e08-4dae-bab7-64ffff5954ee
begin
	sample_impurity = surface(
		gaussian_impurity_potential(
			0.5, N, L;
			x_pos=20,
			y_pos=50,
			σ_x=0.1,
			σ_y=0.1,
			imp_amp=0.2,
			xL=1.,
			yL=1.,
			gate_amp=0.5
		),
		title="Impurity QPC Scattering Region Potential Profile",
		xlabel="L",
		ylabel="N",
		zlabel="E (eV)",
		titlefonts=(14, "times"),
		guidefonts=(12, "times"),
		ticksfonts=(12, "times")
	)
end

# ╔═╡ dfc72ded-0c44-4004-a334-31731558ff8b
savefig(sample_impurity, "sampled_impurity_pot1.png")

# ╔═╡ 8b41d412-dc91-42ef-b119-8420a28c9be9
begin
	μ_gaussian2 = []
	G_gaussian2 = []
	
	for b_en in barrier_ens
		G_loc, μ_loc = conductance_gen4(N, L, 0.1, 0.9, b_en, 300)
		push!(G_gaussian2, G_loc)
		push!(μ_gaussian2, μ_loc)
	end
end

# ╔═╡ c0ba6c21-5908-49a3-9da0-12fc1db6e6ab
## gaussian ionic impurity model conductance plot
begin	
	fig_gauss2_imp = plot(μ_gaussian2[1], G_gaussian2[1])
	
	for i in 2:length(G_gaussian2)
		plot!(μ_gaussian2[i], G_gaussian2[i])
	end
	
	plot!(
		# title & labels
		title="Gaussian-approximated ionic impurity conductance (V2)",
		xlabel="Potential μ [eV]",
		ylabel="Conductance [Gₒ = (2e²/h)]",
		# title & labels geometry
		titlefont=(14,"times"),
		guidefont=(12,"times"),
		tickfont=(12,"times"),
		# annotations
		leg=false,
		# x & y axis
		yticks=0.0:1.0:10.0,
		xticks=0.0:0.1:1.0,
		bottom_margin=5mm
	)
end

# ╔═╡ 654e760c-fd53-48c5-9531-3c10d5917664
savefig(fig_gauss2_imp, "gaussian_impurity_conductance_noisy.png")

# ╔═╡ 844ad86a-e9db-4469-a582-bf352ae8a09d
begin
	sample_impurity2 = surface(
		gaussian_impurity_potential(
			0.5, N, L;
			x_pos=20,
			y_pos=50,
			σ_x=0.2,
			σ_y=0.2,
			imp_amp=0.3,
			xL=1.,
			yL=1.,
			gate_amp=0.5
		),
		title="Impurity QPC Scattering Region Potential Profile (V2)",
		xlabel="L",
		ylabel="N",
		zlabel="E (eV)",
		titlefonts=(14, "times"),
		guidefonts=(12, "times"),
		ticksfonts=(12, "times")
	)
end

# ╔═╡ 67f63163-bbeb-41d8-ba26-b0d52ec1cf90
savefig(sample_impurity2, "big_impurity_sample.png")

# ╔═╡ 62382e8b-03e1-4fc6-95fd-b10184c369d4
begin
	μ_clean1 = []
	G_clean1 = []
	
	for b_en in barrier_ens
		G_loc, μ_loc = conductance_gen(N, L, 0.1, 0.9, b_en, 200)
		push!(G_clean1, G_loc)
		push!(μ_clean1, μ_loc)
	end
end

# ╔═╡ 096918e8-e23a-4951-a531-04428565c60b
## impurity-free conductance plot
begin
	fig_clean_pot = plot(μ_clean1[1], G_clean1[1])
	
	for i in 2:length(G_clean1)
		plot!(μ_clean1[i], G_clean1[i])
	end
	
	plot!(
		# title & labels
		title="Impurity-Free conductance",
		xlabel="Potential μ [eV]",
		ylabel="Conductance [Gₒ = (2e²/h)]",
		# title & labels geometry
		titlefont=(14,"times"),
		guidefont=(12,"times"),
		tickfont=(12,"times"),
		# annotations
		leg=false,
		# x & y axis
		yticks=0.0:1.0:10.0,
		xticks=0.0:0.1:1.0,
		bottom_margin=5mm
	)
end

# ╔═╡ f1b62b34-0a1e-4889-a30d-a874784a2a3a
savefig(fig_clean_pot, "impurity-free_conductance.png")

# ╔═╡ d6ef330e-0b64-4a86-9711-d3a4f69eb82b
begin
	sample_clean = surface(
		smooth_potential(0.5, N, L, 1., 1., 0.5),
		title="Impurity-Free QPC Scattering Region Potential Profile",
		xlabel="L",
		ylabel="N",
		zlabel="E (eV)",
		titlefonts=(14, "times"),
		guidefonts=(12, "times"),
		ticksfonts=(12, "times")
	)
end

# ╔═╡ 9a2d34f5-ba42-41f0-a4cb-9aacb5395d92
savefig(sample_clean, "sampled_impurity-free_pot1.png")

# ╔═╡ 9e995c1f-47e0-441b-b478-5deb057a85c6
md"""
__↓ Clean QPC dense conductance simulation ↓__
"""

# ╔═╡ 50f917b4-906c-11eb-1037-cfbbf1cb3bc0
# begin
# 	G_t = []
# 	μ_t = []
	
# 	for b_height in 0.2:0.1:1.0
# 		G_loc, μ_loc = conductance_gen(N, L, 0.1, 0.9, b_height, 200)
# 		push!(G_t, G_loc)
# 		push!(μ_t, μ_loc)
# 	end
# end

# ╔═╡ 104741eb-104a-4c05-bf43-9f0e9d3a0863
md"""
__↓ Clean QPC dense conductance plot ↓__
"""

# ╔═╡ 34f22ce4-906d-11eb-28f5-4d84d9fecd6d
# begin
# 	barrier_height = 0.3:0.01:1.0
# 	b1 = "μ = " * string(barrier_height[1])
# 	fig_clean = plot(μ_t[1], G_t[1], label=b1)
# 	for i in 2:length(G_t)
# 		μ_base = "μ = " * string(barrier_height[i])
# 		plot!(μ_t[i], G_t[i], label=μ_base)
# 	end
# 	plot!(
# 		# title & labels
# 		title="Clean Simulated Quantised Conductance",
# 		xlabel="Potential μ",
# 		ylabel="Conductance (Gₒ)",
# 		# title & labels geometry
# 		titlefontsize=14,
# 		guidefontsize=14,
# 		tickfontsize=14,
# 		# annotations
# 		leg=false,
# 		# x & y axis
# 		xticks=0.1:0.1:0.8,
# 		yticks=0.0:1.0:10.0
# 	)
	
# 	fig_clean
# end

# ╔═╡ 96452cb8-9071-11eb-191e-434f5df7af07
# savefig(fig_clean, "dense_clean_quantised_G.png") # uncommment to save plot above

# ╔═╡ 589ecae2-39d4-4570-832d-58365a32f606
md"""
__↓ Dirty (impurity) QPC dense conductance simulation ↓__
"""

# ╔═╡ 971ea8f8-9071-11eb-0d85-b1186f4d3aea
# begin
# 	G_im = []
# 	μ_im = []
	
# 	for b_height in 0.3:0.1:1.0
# 		G_loc, μ_loc = conductance_gen2(N, L, 0.1, 0.9, b_height, 200)
# 		push!(G_im, G_loc)
# 		push!(μ_im, μ_loc)
# 	end
# end

# ╔═╡ 3cb02428-cd91-4379-a658-daeefc420e57
md"""
__↓ Dirty (impurity) QPC dense conductance plot ↓__
"""

# ╔═╡ f3987e2a-06d4-4f96-86ee-991c395226c8
# begin
# 	gr()
# 	fig_impurity = plot(μ_im[1], G_im[1])
	
# 	for i in 2:length(G_im)
# 		plot!(μ_im[i], G_im[i])
# 	end
	
# 	plot!(
# 		# title & labels
# 		title="Impurity Simulated Quantised Conductance",
# 		xlabel="Potential μ",
# 		ylabel="Conductance (Gₒ)",
# 		# title & labels geometry
# 		titlefontsize=14,
# 		guidefontsize=14,
# 		tickfontsize=14,
# 		# annotations
# 		leg=false,
# 		# x & y axis
# 		yticks=0.0:1.0:10.0,
# 		xticks=0.0:0.1:1.0
# 	)
# 	fig_impurity
# end

# ╔═╡ 6af5daff-fc93-443e-ae0a-002f7b943e5b
md"""
The graphs above, especially the impurity conductance trace, seems too cluttered, this prevents us being able to really distinguish between traces and observe the oscillations in the conductance.

__To rectify this -- plot traces in steps of ``10`` from `G_im`.__
"""

# ╔═╡ fd1bb926-4ebc-4a7c-aeaf-8c7c11d813a5
md"""
__↓ Visually spaced -- dirty (impurity) QPC dense conductance plot ↓__
"""

# ╔═╡ 36f78314-f54b-4334-85cc-7ebe9424cd17
# begin
# 	gr()
# 	fig_space_impurity = plot(μ_im[1], G_im[1])
	
# 	for i in 4:5:length(G_im)
# 		plot!(μ_im[i], G_im[i])
# 	end
	
# 	plot!(
# 		# title & labels
# 		title="Impurity Spaced Simulated Quantised Conductance",
# 		xlabel="Potential μ",
# 		ylabel="Conductance (Gₒ)",
# 		# title & labels geometry
# 		titlefontsize=14,
# 		guidefontsize=14,
# 		tickfontsize=14,
# 		# annotations
# 		leg=false,
# 		# x & y axis
# 		yticks=0.0:1.0:10.0,
# 		xticks=0.0:0.1:1.0,
# 		# plot geometry
# 	)
# end

# ╔═╡ f195457c-7dce-11eb-1326-83ed59d18879
md"""
## Experimental Conductance Data

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

# ╔═╡ 847b4707-45a6-476c-b3e2-855c96a4d564


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
noisy_plot = plot(
	noisyV[1:(size(noisyV)[1]), 1:(size(noisyV)[2])],
	noisyG[1:(size(noisyG)[1]), 1:(size(noisyG)[2])],
	title="Channel Conductance of a QPC Containing Impurities",
	xlabel="Split Gate Voltage (V)",
	ylabel="Channel Conductance G (μS)",
	# title & labels geometry
	titlefont=(14,"times"),
	guidefont=(12,"times"),
	tickfont=(12,"times"),
	# annotations
	leg=false,
	bottom_margin=5mm
)

# ╔═╡ f7b17256-8331-11eb-1542-0d28cdf6478f
# scale conductance plots by G_0
begin
	G_0 = (2 * e^2) / h
	
	cleanG_scaled = (cleanG .* 1e-6) ./ G_0
	noisyG_scaled = (noisyG .* 1e-6) ./ G_0
end;

# ╔═╡ ee73b04f-fe55-4eec-99bb-2ef5c45b468e
clean_plot_scaled = plot(
	cleanV[1:(size(cleanV)[1]),1:(size(cleanV)[2])],
	cleanG_scaled[1:(size(cleanG_scaled)[1]), 1:(size(cleanG_scaled)[2])],
	title="Channel Conductance of a Clean QPC (Scaled)",
	xlabel="Split Gate Voltage (V)",
	ylabel="Channel Conductance G [G₀ = (2e²/h)]",
	# title & labels geometry
	titlefont=(14,"times"),
	guidefont=(12,"times"),
	tickfont=(12,"times"),
	# annotations
	leg=false,
	bottom_margin=5mm
)

# ╔═╡ 6cdec916-aa89-4a0d-ba7e-93ed1a09075a
savefig(clean_plot_scaled, "clean_experimental_scaled.png")

# ╔═╡ 5f7eee43-19a2-4bfa-8d46-0bab82dd7268
noisy_plot_scaled = plot(
	noisyV[1:(size(noisyV)[1]), 1:(size(noisyV)[2])],
	noisyG_scaled[1:(size(noisyG_scaled)[1]), 1:(size(noisyG_scaled)[2])],
	title="Channel Conductance of a QPC Containing Impurities (Scaled)",
	xlabel="Split Gate Voltage [V]",
	ylabel="Channel Conductance [G₀ = (2e²/h)] ",
	# title & labels geometry
	titlefont=(14,"times"),
	guidefont=(12,"times"),
	tickfont=(12,"times"),
	# annotations
	leg=false,
	bottom_margin=5mm,
	right_margin=5mm
)

# ╔═╡ fd233a8d-e7b1-41be-8631-845402e0ecc2
savefig(noisy_plot_scaled, "noisy_experimental_scaled.png")

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
# ╟─faedfda0-72d7-11eb-0b80-7d63e962468d
# ╠═fce9afc0-624a-11eb-09e2-c38456a1fe35
# ╠═d03c2ac6-6253-11eb-0483-596dd3d5e5a4
# ╠═095be506-64e5-11eb-3ac8-6dbf5a7f5f9e
# ╟─b3fb891c-8d83-11eb-31d8-3fea8e634889
# ╠═212d911a-7dc3-11eb-11ee-333220a641e5
# ╠═9ff7af7e-7dc2-11eb-17b8-e7fe576888c4
# ╟─a1f94578-8d84-11eb-1de6-03bab5d1e34e
# ╠═6b63b052-64eb-11eb-1a62-33262062ece1
# ╟─deb49ea2-8d85-11eb-34ed-7b71e4b3cef8
# ╠═2fd2a6c8-6256-11eb-2b61-1deb1e2e4c77
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
# ╠═e125a71d-37df-4f47-bebe-62ea8bccf3e2
# ╠═a3adb388-be9b-49ec-99c6-537e87c57cee
# ╟─973c91dd-58e7-4462-a792-85ad18eca925
# ╟─754fd72a-8f2b-11eb-381b-c19ea1fed40a
# ╟─f09e2df2-8e37-11eb-3537-ad7491c66146
# ╟─3875774a-8d87-11eb-321a-0f74a8dc4c73
# ╠═d5a61ac2-9074-11eb-1948-934b10dcb64c
# ╠═20312b33-55c7-4fb6-8fe6-4aa386fddb61
# ╠═fd1585d4-8f2f-11eb-21c5-f96dc290073c
# ╠═5077c151-fa25-4312-9e93-d52e8301e1aa
# ╠═69e57db4-8f2f-11eb-04fa-052bf0433dea
# ╟─442a3b4f-632d-4a2a-8813-912aa14c3031
# ╠═2250cb4c-0065-4ce4-8c5a-205392be0909
# ╠═ef780500-363f-4f76-a59e-e4220afe344d
# ╠═180d8892-8cd9-4d3f-bf69-8aeac076fc57
# ╠═753d2cfa-df10-4216-85df-7292ef44e692
# ╠═0857e99b-a08b-43d7-906d-32107c0977c9
# ╠═88d9513a-c8ee-451b-ac18-11836c0877bb
# ╠═047a102e-3936-4269-93fb-d5530aaa495d
# ╠═e67b1c41-efb8-448a-86a7-9918af1df1f8
# ╠═bf2b0abe-b2f8-4f88-a82f-22d839b33a27
# ╠═015176cd-e00f-419b-aeae-a4e3db0e10da
# ╠═9492ef77-6124-44f3-baad-039ec1ddd99c
# ╠═5bb0a920-9906-43d9-9c9d-3d9ce44deacf
# ╟─be491790-b3a8-42f2-8fbf-20d8a3d0c0f0
# ╠═7e2fdc41-a1b5-4a44-8c87-08f15dc60e4b
# ╠═bd9b0c3c-4e08-4dae-bab7-64ffff5954ee
# ╠═dfc72ded-0c44-4004-a334-31731558ff8b
# ╠═a1b4a714-88e9-43a9-a594-fbe401039663
# ╠═00fa5126-a4b2-49ec-82df-e09a0d888dca
# ╟─3738fea4-522d-4160-a96d-c798ff0f4786
# ╠═8b41d412-dc91-42ef-b119-8420a28c9be9
# ╠═844ad86a-e9db-4469-a582-bf352ae8a09d
# ╠═67f63163-bbeb-41d8-ba26-b0d52ec1cf90
# ╠═c0ba6c21-5908-49a3-9da0-12fc1db6e6ab
# ╠═654e760c-fd53-48c5-9531-3c10d5917664
# ╟─6482750b-35b7-42da-94e4-a38e69a67bd6
# ╠═62382e8b-03e1-4fc6-95fd-b10184c369d4
# ╠═d6ef330e-0b64-4a86-9711-d3a4f69eb82b
# ╠═9a2d34f5-ba42-41f0-a4cb-9aacb5395d92
# ╠═096918e8-e23a-4951-a531-04428565c60b
# ╠═f1b62b34-0a1e-4889-a30d-a874784a2a3a
# ╟─57613f4f-9faf-4e9a-9951-8bd675edc72f
# ╟─0c4227d0-9069-11eb-2ffc-9b8d4d9dc3cf
# ╟─65d696f4-d49a-41b4-aacd-4ac5be4de1ed
# ╠═ef295a00-8db7-11eb-1f02-db93c59bd24f
# ╠═e260588a-240c-49c4-aa16-02fb9c4fb741
# ╠═7ef77e84-c92c-4a21-9e44-8bcb9989c287
# ╟─56777301-68ca-4840-b6a5-86f4bb3dc81a
# ╠═47f47e16-8d87-11eb-2734-fbe36fd94431
# ╟─9e995c1f-47e0-441b-b478-5deb057a85c6
# ╠═50f917b4-906c-11eb-1037-cfbbf1cb3bc0
# ╟─104741eb-104a-4c05-bf43-9f0e9d3a0863
# ╠═34f22ce4-906d-11eb-28f5-4d84d9fecd6d
# ╠═96452cb8-9071-11eb-191e-434f5df7af07
# ╟─589ecae2-39d4-4570-832d-58365a32f606
# ╠═971ea8f8-9071-11eb-0d85-b1186f4d3aea
# ╟─3cb02428-cd91-4379-a658-daeefc420e57
# ╠═f3987e2a-06d4-4f96-86ee-991c395226c8
# ╟─6af5daff-fc93-443e-ae0a-002f7b943e5b
# ╟─fd1bb926-4ebc-4a7c-aeaf-8c7c11d813a5
# ╠═36f78314-f54b-4334-85cc-7ebe9424cd17
# ╟─f195457c-7dce-11eb-1326-83ed59d18879
# ╟─552aa3e0-7dd2-11eb-399e-ad5fc208fbc5
# ╠═d5a18c24-7dd1-11eb-30d4-6dcb0c8c9c4e
# ╟─5056efdc-7dd6-11eb-21d3-e13768d765d9
# ╟─ee73b04f-fe55-4eec-99bb-2ef5c45b468e
# ╠═6cdec916-aa89-4a0d-ba7e-93ed1a09075a
# ╟─fa5d996e-7dd5-11eb-3010-8935856e0b68
# ╟─5f7eee43-19a2-4bfa-8d46-0bab82dd7268
# ╠═fd233a8d-e7b1-41be-8631-845402e0ecc2
# ╠═f7b17256-8331-11eb-1542-0d28cdf6478f
# ╟─847b4707-45a6-476c-b3e2-855c96a4d564
# ╟─7545065e-72e7-11eb-1db0-3df6683bcbeb
# ╠═b1c556a8-72e3-11eb-1299-8b52ae0c19b7
# ╠═5e9936fa-72e5-11eb-078f-bd9e193eda1a
# ╠═a6f19760-8d81-11eb-2e7e-6dae07c47af9
# ╠═d8cbc6e4-7dbd-11eb-378e-8bf7a5d244f2
