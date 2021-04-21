module CCN_model

export Types, System, Barrier, NEGF

# bring model components into local scope

include("types.jl")
using .Types

include("system.jl")
using .System

include("barrier.jl")
using .Barrier

include("negf.jl")
using .NEGF

end
