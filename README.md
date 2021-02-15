# QPC_Investigation

## Running *Julia* notebooks

Contents of the `/notebooks` folder are [*Pluto*](https://juliapackages.com/p/pluto) notebooks and can be run interactively.

To achieve this:
- navigate to the location of the project, and more importantly the `manifest.toml` & `project.toml` files
- run *Julia* with: `julia --project==./`
- bring the *Pluto* package into the scope with: `julia> using Pluto`
- run the *Pluto* server with `julia> Pluto.run()`

## Development notes

### Implemented features

**Chalker-Coddington Network Model**

- constructor-like `T(μ, N) -> T::T_data` & `S(T, N) -> S::S_data` transfer and scattering matrix-assembling functions
- scattering matrix summing function: `sum_S(S1::S_data, S2::S_data) -> S::S_data`
- function to generate final scattering matrix for complete system description: `gen_S_total(V,L) -> S::S_data`
- QPC potential barrier, profile generating function: `smooth_potential(...) -> V::Array{Float64, 2}`
- error evaluation function: `error_ϵ(S::S_data, T::T_data)`
- function to pick out indices of varying wave behaviours from eigen decomp.: `pickoutᵢ(λ_values, mode) -> Array(Int, 1)`

### Buggy features

- system solving function: `system_solve(μ, V, L, i, opt) -> G::Float64`
  - output behaviour is **nominal** in: data *size* and *type*
  - output behaviour is **not nominal** in: data *range* and *scale*
  
