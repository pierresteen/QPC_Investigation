## CCN model custom types
module Types

export T_data, S_data

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
end

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
end

"""
**Solution parameter type**
"""
struct Sys_sol
	G
	τ
	α
	r
	β
end

end
