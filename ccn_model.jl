# Chalker-Coddington Network Model of QPC Conductance
module ccn_network

## Data Structures:

"""
	T_data

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
	S_data

__**Scattering matrix data type**__

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



end
