### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ aec08616-8123-11eb-3b8e-a5db3738fdf0
using Images

# ╔═╡ 1ad5470c-8202-11eb-3c31-2940b4217974
md"""
## Read Next:

~~ In P. Harrison _Computational Physics of Quantum Ws..._~~

__Solving Schrodinger's Eq.__
- page 138 	-- 2.23+	-- Poisson's Equation & Discretised Version
- page 142 	-- 3.25 	-- self-consistent Schrodinger-Poisson Solution

__Impurities in Heterostructures__
- page 169 	-- 5+ 		-- __Impurities__ (important for project!)  [X] turns out no

__Models of Quantum Wires__
- page ...	-- 8.2		-- Schrodinger's equation in quantum wires 	[✓]
- page ...	-- 8.4		-- Quantum wire approximation
- page ...	-- 8.4		-- Matrix approaches

__Carrier Scattering__
- page ...	-- 10.25	-- __Impurity Scattering__ (important for project!)
- page ...	-- 10.30	-- Carrier scattering in quantum wires

__Maybe Look At__
- 14	-- Multiband evelope function (kp) method
- 15	-- Empirical pseudo-potential band structure
- 16	-- Large-basis methods for quantum wires

"""

# ╔═╡ e222372e-7f80-11eb-339b-0f92d080e55c
md"""
# Impurity Effects in a 1D QPC

## Conductance of a QPC:

In a quasi-1D __ballistic__ wire, the conductance $G$ can be described using Landauer's formula:

$$G = \frac{2e^2}{h} \cdot \Sigma\ T_{mn}$$

for proapagation modes $(m,n)$.

Here, $T_{m,n}$ is the transmission probability of an electron passing from a source lead of mode $m$ -- to a drain lead of mode $n$. 
The effect of impurities in the QPC's channel will reveal themselves as changes in the  transmission coefficient.

## Impurity Categories:

There exists two general categories for models which consider impurities in the QPC channel.
The choice of which depends on the size $d$ of the impurity.

For $d ⋘ λ_F$, where $λ_F$ is the electron wavelength at the fermi energy $ε_F$, mode mixing or interchannel scattering, induced by the impurity potential is considered to affect $G$.

For the limit $d ≥ λ_F$, the impurity is treated as a large obstacle that scatters electrons into two channels. 
The transmission coefficient $T_{m,n}$ is described as coupled quantum wires (QW), hence $G$ is greaty influence by both channel interference and disorder scattering.

These models predict that the scattering of electron waves would produce phenomena, resonant in nature, which would __impose additional oscillatory features on $G$, regardless of  the impurity size__.
"""

# ╔═╡ e3b09cdc-8002-11eb-2ace-11274a5d1f8d
md"""
## Scattering


"""

# ╔═╡ a0c33c02-8121-11eb-1c03-b529b28bdf5c
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

# ╔═╡ 21e31d78-81fb-11eb-3d9e-3308ccc1e6de
md"""
## Matrix Solutions of the Discretised Schrodinger Equation

### Discretised Schrodinger Equation:

Under *effective electron mass and envelope function approximations*, the discretised Schrodinger equation is:

```math
-\frac{\hbar^2}{2m^*}\left[\frac{\psi(z+\delta z) - 2\psi(z) + \psi(z - \delta z)}{(\delta z)^2}\right] + V(z)\psi(z) = E\psi(z) \quad (1)
```

We rearrange it into the form:

```math
a\psi(z - \delta z) - b(z)\psi(z) + c\psi(z + \delta z) = E\psi(z)  \quad (2)
```
```math
a = c = -\frac{\hbar^2}{2m^* (\delta z)^2} \quad b(z) = \frac{\hbar^2}{m^* (\delta z)^2} + V(z)
```

### Matrix Formulation:

We have a discretised ``z`` dimension, of size ``N``.
Start by rewriting ``(2)`` as:

```math
a_i \psi_{i-1}- b_i \psi_i + c_i \psi_{i+1} = E\psi_i  \quad (3)
```

Here, ``i``  represents the *index* of each sample of of ``\psi`` in the discretised dimension ``z``.
We also rewrite the coefficients as:

```math
a_{i+1} = c_i = -\frac{\hbar^2}{2m^* (\delta z)^2} \quad b_i = \frac{\hbar^2}{m^* (\delta z)^2} + V(z)
```

The boundary conditions, ``\psi_0 = \psi_{N+1} = 0`` represent the first set of points outside of the system.
We can now express ``(3)`` at each sample in the system as a linear system of equations:

```math
a_1 \psi_{0} - b_1 \psi_1 + c_1 \psi_{2} = E\psi_1
```
```math
a_2 \psi_{1} - b_2 \psi_2 + c_2 \psi_{3} = E\psi_2
```
```math
\dots
```
```math
a_{N} \psi_{N-1} - b_{N} \psi_{N} + c_{N} \psi_{N+1} = E\psi_{N}
```

We can clearly see that we can turn this into a system of the form ``H\psi = E\psi``, where ``H`` is:

```math
H =
\begin{bmatrix}
	b_1 & c_1 & 0 & \dots & 0\\
	a_2 & b_2 & c_2 & \dots & 0\\
	0 & \ddots & \ddots & \ddots & 0\\
	\vdots & \dots & a_{N-1} & b_{N-1} & c_{N-1}\\
	0 & \dots & 0 & a_{N} & b_{N}\\
\end{bmatrix}
\quad (4)
```

and ``\psi`` is a vectors containing all wave functions at the sampling points:

```math
\psi = [\psi_1, \psi_2, \dots, \psi_{N}]^T
```

### Solving by Eigenvalue Decomposition:

This makes for a __matrix eigenvalue problem__ which __can__ be solved directly to locate _all_ the energies of the system simultaneously, along with the corresponding wave functions.

"""

# ╔═╡ 17bcbb60-8264-11eb-3e5e-69d957bb36fd
md"""
## Carrier Scattering in Quantum Wires

### How to approach the problem?

See P. Harrison _'QWWDTCP'_ references `[44 - 48]`, referenced on page 393 (375).

"""

# ╔═╡ Cell order:
# ╟─1ad5470c-8202-11eb-3c31-2940b4217974
# ╟─e222372e-7f80-11eb-339b-0f92d080e55c
# ╠═e3b09cdc-8002-11eb-2ace-11274a5d1f8d
# ╠═aec08616-8123-11eb-3b8e-a5db3738fdf0
# ╟─a0c33c02-8121-11eb-1c03-b529b28bdf5c
# ╠═21e31d78-81fb-11eb-3d9e-3308ccc1e6de
# ╟─17bcbb60-8264-11eb-3e5e-69d957bb36fd
