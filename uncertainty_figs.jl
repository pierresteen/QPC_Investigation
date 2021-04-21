### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ be3bf0b0-9f6e-11eb-0f09-cb392dcecec5
begin
	using Plots
	using Plots.PlotMeasures
end

# ╔═╡ 3af77434-bfd6-4ec7-89a7-b1c18877149f
domain = 0:0.1:100

# ╔═╡ c45d91ab-e70c-463c-9fdf-52f679066716
sinusoidal(ω, domain) = [sin(ω*i) for i in domain]

# ╔═╡ 7ef461c3-0800-4b9b-9aa2-282b95de7e4e
C1 = sinusoidal(0.4, domain)

# ╔═╡ 7eb9db87-b705-4549-8f7c-de9ba27dd446
M1 = sinusoidal(0.0315, domain)

# ╔═╡ 1a3faf0d-06e0-44ef-9e84-55ed8f504244
plot(domain, C1); plot!(domain,M1)

# ╔═╡ 2cf67e7d-6acb-4e98-ab5d-7b7ccafdbad3
P1 = M1 .* C1

# ╔═╡ e727bccc-ab42-4cbc-8de0-b1580008afc2
M2 = -M1

# ╔═╡ 2b31f0ff-c175-4f76-8a2e-f20a6092135c
begin
	waves = [P1, M1, -M1]
	styles = filter((s->begin
                s in Plots.supported_styles()
			end), [:solid, :dash, :dash])
	styles = reshape(styles, 1, length(styles))
	envelope = plot(
		domain,
		waves,
		# linewidth=[2, 2, 2],
		line=(5, styles),
		leg=false,
		title="ψ Packet & Envelope",
		guidefontsize=13,
		titlefontsize=15,
		xlabel="x",
		ylabel="ψ(x,t)",
		top_margin=5mm,
		left_margin=10mm,
		linecolor=[:blue :red :red],
		xticks=false,
		yticks=false,
		widen=true,
		guide=true
	)
end

# ╔═╡ 29a74fb4-ce15-4eae-80ac-56542d9e7aee
savefig(envelope, "envelope_momentum.pdf")

# ╔═╡ 7404ccc1-b762-4698-8d04-fc0c1b01c2a4
begin
	styles2 = filter((s->begin
                s in Plots.supported_styles()
			end), [:dot, :dot, :dot])
	styles2 = reshape(styles2, 1, length(styles2))
	envelope2 = plot(
		domain,
		waves,
		# linewidth=[2, 2, 2],
		line=(5, styles2),
		leg=false,
		title="ψ Packet & Envelope",
		xlabel="x",
		ylabel="ψ(x,t)",
		guidefontsize=13,
		titlefontsize=15,
		top_margin=5mm,
		left_margin=10mm,
		right_margin=20mm,
		linecolor=[:blue :red :red],
		xticks=false,
		yticks=false,
		widen=true,
		guide=true
	)
	scatter!(
		(domain[750], 0),
		marker=(10, :circle)
	)
	vline!(
		[domain[750]],
		color=:black,
		linewidth=3
	)
	annotate!(
		domain[950],
		-M1[600],
		Plots.text("x = x_measured", 12, :black)
	)
end

# ╔═╡ 89a2ea00-bc1d-4889-ae23-ed56d41954ef
savefig(envelope2, "envelope_measured.pdf")

# ╔═╡ Cell order:
# ╠═be3bf0b0-9f6e-11eb-0f09-cb392dcecec5
# ╠═3af77434-bfd6-4ec7-89a7-b1c18877149f
# ╠═c45d91ab-e70c-463c-9fdf-52f679066716
# ╠═7ef461c3-0800-4b9b-9aa2-282b95de7e4e
# ╠═7eb9db87-b705-4549-8f7c-de9ba27dd446
# ╠═1a3faf0d-06e0-44ef-9e84-55ed8f504244
# ╠═2cf67e7d-6acb-4e98-ab5d-7b7ccafdbad3
# ╠═e727bccc-ab42-4cbc-8de0-b1580008afc2
# ╠═2b31f0ff-c175-4f76-8a2e-f20a6092135c
# ╠═29a74fb4-ce15-4eae-80ac-56542d9e7aee
# ╠═7404ccc1-b762-4698-8d04-fc0c1b01c2a4
# ╠═89a2ea00-bc1d-4889-ae23-ed56d41954ef
