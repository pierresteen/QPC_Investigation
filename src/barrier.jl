## CCN model potential barrier functions
## smooth potentials only - no impurity
module Barrier

export smooth_potential

"""
	meshgrid(x, y)

Generates a 2D meshgrid, same functionality as MATLAB meshgrid function.
"""
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

"""
	smooth_potential_broken(μ, N, L, xL=1.,yL=1., amp=1.)

Generates a smooth saddle-point potential profile for system ef dimensions `width = N` and `length = L`.
"""
function smooth_potential(μ, N, L; xL=1., yL=1., amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	
	return (-0.5 * amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
end

end
