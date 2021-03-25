# Chalker-Coddington Network Model System Components:
module ccnsystem

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

"""
	smooth_potential(μ, N, L, xL=1.,yL=1., amp=1.)

Creates a smooth potential profile for the model.
"""
function smooth_potential(μ, N, L, xL=1.,yL=1., amp=1.)
	px = Float64.(range(-xL, xL, length=N))
	py = Float64.(range(-yL, yL, length=L))
	
	X, Y = meshgrid(px, py)
	
	return (-0.5 * amp) * (tanh.(X.^2 - Y.^2) .+ 1) .+ μ
end

export 

end
