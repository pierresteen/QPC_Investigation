# Package Management:
using CSV
using Plots
using Images
using PlutoUI
using DataFrames
using LinearAlgebra
using DelimitedFiles

# Constants:
const e 	= -1.602176634e-19 	# (C)
const h 	= 6.62607015e-34 	# (Js)
const ħ 	= 1.054571817e-34 	# (Js)
const h_eV 	= abs(h/e) 		 	# (eVs)
const ħ_eV 	= abs(ħ/e) 			# (eVs)
const ε_0	= 8.8541878128e-12	# (Fm^-1)
