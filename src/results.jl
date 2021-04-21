## Results generating file for BEng Project

include("ccn_model.jl")
using .CCN_model

function conductance_gen(N, L, μ_min, μ_max, barrier_height, precision)
	G = []
	μ_range = range(μ_min, μ_max, length=precision)
	
	for μ_en in μ_range
		V_en = CCN_model.smooth_potential(μ_en, N, L; xL=1., yL=1., amp=barrier_height)
		push!(G, CCN_model.system_solve(μ_en, V_en, N, L, μ_en)[1])
	end
	
	return G, μ_range
end

N = 40
L = 100

G_t = []
μ_t = []

barrier_range = 0.3:0.1:1.0

for barrier_height in barrier_range
	G_loc, μ_loc = conductance_gen(N, L, 0.1, 0.9, barrier_height, 200)
	push!(G_t, G_loc)
	push!(μ_t, μ_loc)
end

# G_t, μ_t = conductance_gen(N, L, 0.1, 0.9, 0.5, 50)

# G_t, μ_t

gr() # set GR plotting backend
using Plots.PlotMeasures

fig = plot(
	μ_t[1],
	G_t[1],
	leg=:topleft,
	lab="Vg=" * string(barrier_range[1]),
	legendfont=(7, "times", :grey)
)

for i in 2:length(G_t)
	plot!(
		μ_t[i],
		G_t[i],
		leg=:topleft,
		lab="Vg=" * string(barrier_range[i]),
		legendfont=(8, "times", :grey2)
	)
end

plot!(
	# labelling & annotations:
	title="Quantised conductance of a QPC (scaled)",
	xlabel="μ [eV]",
	ylabel="Conductance  [G₀=(2e²/h)]",
	leg=:topleft,
	# font properties:
	titlefont=(14, "times"),
	guidefont=(13, "times"),
	# axis settings:
	yticks=0.0:1.0:10.0,
	xticks=0.2:0.1:1.0
)

savefig(fig, "G_clean_scaled.png")

display(fig)

e = -1.602176634e-19 # (C)
h = 6.62607015e-34 # (Js)

G_scaled = G_t .* (2 * e^2)/h
μ_scaled = μ_t .* (2 * e^2)/h

fig2 = plot(
	μ_scaled[1],
	G_scaled[1],
	leg=:topleft,
	lab="Vg=" * string(barrier_range[1]),
	legendfont=(7, "times", :grey)
)

for i in 2:length(G_scaled)
	plot!(
		μ_scaled[i],
		G_scaled[i],
		leg=:topleft,
		lab="Vg=" * string(barrier_range[i]),
		legendfont=(8, "times", :grey2)
	)
end

plot!(
	# labelling & annotations:
	title="Quantised conductance of a QPC (unscaled)",
	xlabel="μ [eV]",
	ylabel="Conductance [S]",
	leg=:topleft,
	# font properties:
	titlefont=(14, "times"),
	guidefont=(13, "times"),
	# axis settings:
	yticks=0.0:1.0:10.0,
	xticks=0.2:0.1:1.0
)

savefig(fig2, "G_clean_unscaled.png")

display(fig2)
