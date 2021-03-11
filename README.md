# QPC_Investigation

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
  
