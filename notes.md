# BEng Project Direction

## Implemented :

1. Chalker-Coddingtion Network Model

	Discretised network form of a scattering model.
	Conductance calculated by finding eigenstates of total system's scattering matrix.

2. Plotted Experimential Results

	Conductance traces for a QPC with varying split-gate bias voltage.
	Data split into `clean` and `noisy` samples:
		- `clean` contains traces for QPC without any impurities in the channel
		- `noisy` is traces with impurities in the channel (impurity type unknown **TBC**)

3. Impurity Considerations

	The impurity is due to exited ions in the QPC lattice.
	These impart an electro-magnetic on the electrons travelling ballistically in the
	channel.
		- What impacts does this have on Q-transport?
		- Why do we see oscillations in the quatised conductance in the 'transitions' between filled energy levels?
    		- Does the bandgap periodically grow as a result of magnetic interference?
    		- Perpendicular force acts on the electron angularly!
    		- Zeeman effect is occuring?

## Where To Next? :

- is a time dependent q-conductance sim possible?
- is a model based on analytical forms (ie. considering ψ's) possible

Decided to abandon more complex analytical model implementation in order to pursue
further work on 

---

__Impurity considerations__

