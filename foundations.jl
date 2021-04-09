### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 423e1142-5f3a-11eb-3eba-b93b5dcd725b
md"""
# Foundational theory for low-dimensional quantum systems
"""

# ╔═╡ 60669040-5f3a-11eb-1316-217b54986e6d


# ╔═╡ 023936ea-5f52-11eb-0412-75b6096848b4
md"""
## References:

**Quantum wells, wires, and dots** [(source)](https://onlinelibrary-wiley-com.libproxy.ucl.ac.uk/doi/pdf/10.1002/0470010827)
- page 93+: Numerical Solutions
- page 39+: Solutions to Schrodinger Equation
"""

# ╔═╡ 02ebaff0-5f52-11eb-1b04-a91803e5839c


# ╔═╡ 615a3006-5f3a-11eb-1b54-a50c54285aa8
md"""
## 1. Semiconductors and heterostructures

### 1.1. Wave mechanics
From De Broglie, the momentum of a particle: $p$, has an associated wavelength corresponding to:

$\lambda =  \frac{h}{p}$

Then, an electron **in a vacuum** at a position $\mathbf{r}$, away from any electromagnetic influence, can be described by a wave-form **state function**:

$\psi = e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

Here, $t$ is time, $\omega$ is the angulr frequency and the modulus of the wavevector $\mathbf{k}$ is:

$k = |\mathbf{k}| = \frac{2\pi}{\lambda}$

Quantum mechanical momentum is a linear operator acting on the *wave function* $\psi$, with momentum $\mathbf{p}$ arising as an eigenvalue:

$-i\hbar\nabla\psi = \mathbf{p}\psi$

$\nabla = \frac{\partial}{\partial x}\mathbf{\hat{i}} + \frac{\partial}{\partial y}\mathbf{\hat{j}} + \frac{\partial}{\partial z}\mathbf{\hat{k}}$

so when operating on the electron vacuum wave function $\psi$, we have:

$-i\hbar\nabla e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)} = \mathbf{p}e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

This expands to:

$-i\hbar\left(\frac{\partial}{\partial x}\mathbf{\hat{i}} + \frac{\partial}{\partial y}\mathbf{\hat{j}} + \frac{\partial}{\partial z}\mathbf{\hat{k}}\right)e^{i(k_xx + k_yy + k_zz - \omega t)} = \mathbf{p}e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

$\therefore -i\hbar\left(ik_x\mathbf{\hat{i}} + ik_y\mathbf{\hat{j}} + ik_z\mathbf{\hat{k}}\right)e^{i(k_xx + k_yy + k_zz - \omega t)} = \mathbf{p}e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

giving an eigenvalue:

$\mathbf{p} = \hbar(k_x\mathbf{\hat{i}} + k_y\mathbf{\hat{j}} + k_z\mathbf{\hat{k}}) = \hbar\mathbf{k}.$

This result is in line with what we expect from De Broglie's relationship:

$p = \hbar k = \left(\frac{h}{2\pi}\right)\left(\frac{2\pi}{\lambda}\right)$

Following on from the result above, the kinetic energy of a particle mass $m$ is:

$T = \frac{1}{2}mv^2 = \frac{(mv)^2}{2m} = \frac{p^2}{2m}$

The quantumm mechanical analogy of this is represented by an eigenvalue equation:

$\frac{1}{2m}(-i\hbar\nabla)^2\psi = -\frac{\hbar^2}{2m}\nabla^2\psi = T\psi$

Here $T$ is the kinetic energy eigenvalue.
When acting upon the electron vacuum wave function we have:

$-\frac{\hbar^2}{2m}\nabla^2e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}  = Te^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

$\to -\frac{\hbar^2}{2m}(i^2k_x^2 + i^2k_y^2 + i^2k_z^2)e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)} = Te^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

And thus the eigenvalue for kinetic energy is:

$T = \frac{\hbar^2k^2}{2m} = E_K$

This characterises the enrgy profile of an electron **in a vacuum without the influence of electromagnetic fields**.

This means that the kinetic energy of the electron is proportional to the wavevector and concequently momentum.

### 1.2. Wave mechanics summary
The equation describing the total energy of a particle in this wave description is called the **time-independent** Schrodinger equation, with only kinetic energy contributing, it can expressed as:

$-\frac{\hbar^2}{2m}\nabla^2\psi = E\psi$

We can also extend this to time-dependence explicitly as:

$i\hbar\frac{\partial}{\partial t}e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)} = i\hbar(-i\omega)e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$

i.e.

$i\hbar\frac{\partial}{\partial t}\psi = \hbar\omega\psi$

It is obvious from this that $\hbar\omega$ eigenvalue is also equivalent to the total energy but in a wave-associated-form.
This accommodates for the **particle-wave duality** of quantum mechanics.
The final, time-dependent, Schrodinger equation is given as:

$i\hbar\frac{\partial}{\partial t}\psi = E\psi$
"""

# ╔═╡ f35b3d40-5f50-11eb-00c0-73de4137cff9


# ╔═╡ f162fe92-5f52-11eb-35fd-8b42a57a0366
md"""
## 2. Solutions to Schrodinger's equation

### 2.1. Infinite well case
An infintely deep one-dimensional potential well produces the simplest confinement potential to treat in quantum mechanics.
The kinetic and potential energy summed corresponds to the total energy of the system.
In a wave mechanics setting, these values are the eigenvalues of the linear operators:

$\mathcal{T}\psi + \mathcal{V}\psi = E\psi$

here $\psi$ are the eigenfunctions which describe the system's state.
...
"""

# ╔═╡ Cell order:
# ╟─423e1142-5f3a-11eb-3eba-b93b5dcd725b
# ╟─60669040-5f3a-11eb-1316-217b54986e6d
# ╟─023936ea-5f52-11eb-0412-75b6096848b4
# ╟─02ebaff0-5f52-11eb-1b04-a91803e5839c
# ╟─615a3006-5f3a-11eb-1b54-a50c54285aa8
# ╟─f35b3d40-5f50-11eb-00c0-73de4137cff9
# ╟─f162fe92-5f52-11eb-35fd-8b42a57a0366
