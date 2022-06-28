module GeometricFittingProblems

using DelimitedFiles, LinearAlgebra

export load_problem, solve, build_problem, inverse_power_method 

import Base.show

"""
    FitProbType

It is an immutable type used by main functions of this package

"""
struct FitProbType
    name::String
    data:: Array{Float64,2}
    npts:: Int
    nout:: Int
    model::Function
    dim:: Int
    cluster::Bool
    noise::Bool
    solution::Array{Float64,1}
    description::String
end

struct FitOutputType
    status::Bool
    solution::Vector{Float64}
    niter :: Int
    minimum :: Float64
    feval :: Int
end

"""
    load_problem(filename::String)

This function is used to load a problem from a csv file and convert to FitProbType. It is an important function because FitProbType is the unique supported format in this package. 

# Examples
```
julia-repl
julia> load_problem("toy.csv")

returns a FitProbType
```
"""
function load_problem(filename::String)
    prob_matrix = readdlm(filename,':')
    return FitProbType(prob_matrix[1,2],eval(Meta.parse(prob_matrix[2,2])),prob_matrix[3,2],prob_matrix[4,2],eval(Meta.parse(prob_matrix[5,2])),prob_matrix[6,2],prob_matrix[7,2],prob_matrix[8,2],eval(Meta.parse(prob_matrix[9,2])),prob_matrix[10,2])
end


function solve(prob::FitProbType,θinit::Vector{Float64},method::String)

end

function build_problem(probtype::String,limit::Vector{Float64},params::Vector{Float64})
    if probtype == "sphere2D"
        println("params need to be setup as [center,radious,npts,nout]")
        c = [params[1],params[2]]
        r = params[3]
        npts = Int(params[4])
        x = zeros(npts)
        y = zeros(npts)
        θ = [0.0:2*π/(npts-1):2*π;]
        for k=1:npts
            x[k] = c[1]+r*cos(θ[k])
            y[k] = c[2]+r*sin(θ[k])
        end
        nout = Int(params[5])
        k = 1
        iout = []
        while k<=nout
            i = rand([1:npts;])
            if i ∉ iout
                push!(iout,i)
                k = k+1
            end
        end
        for k = 1:nout
            x[iout[k]]=x[iout[k]]+rand([0.25*r:0.1*(r);(1+0.25)*r])
            y[iout[k]]=y[iout[k]]+rand([0.25*r:0.1*(r);(1+0.25)*r])
        end
        FileMatrix = ["name :" "sphere2D";"data :" [[x y]]; "npts :" npts;"nout :" nout; "model :" "(x,t) -> (x[1]-t[1])^2 - (x[2]-t[2])^2 - t[3]^2";"dim :" 3; "cluster :" "false"; "noise :" "false"; "solution :" [push!(c,r)]; "description :" "none"]
        
        open("sphere2D_$(c[1])_$(c[2])_$(c[3])_$(nout).csv", "w") do io
           writedlm(io, FileMatrix)
        end

    end
    if probtype == "sphere3D"
    
    end
end

"""
    inverse_power_method :: function

This functions implements the inverse power method to find the smallest eigen value associated to an array A.

# Examples
```
julia-repl

julia> A = [1.0 2.0 0.0; 2.0 -5.0 3.0; 0.0 3.0 4.0]

julia> inverse_power_method(A,[1.0,1.0,1.0])

returns ???
```
"""
function inverse_power_method(A::Array{Float64};q0=ones(size(A)[1]),ε=10.0^(-4),limit=100)
    stop_criteria = 1000.0
    F = lu(A)
    B = inv(A)
    k = 1
    s = 0.0
    q = zeros(length(q0))
    while stop_criteria > ε && k<limit
        s = norm(q0,Inf)
        q = B*(q0/s)
        stop_criteria = norm(abs.(q)-abs.(q0),Inf)
        q0 = copy(q)
        k = k+1
    end
    if k==limit
        error("iteration limit of inverse power method was reached")
    else
        return q,1.0/s
    end
end

function show(io::IO, fout::FitOutputType)

    print(io,"  ▶ Output ◀ \n")
    if Bool(fout.status) ==true
        print(io,"  ↳ Status (.status) = Convergent \n")
    else
        print(io,"  ↳ Status (.status) = Divergent \n")
    end
    print(io,"  ↳ Solution (.solution) = $(fout.solution) \n")
    print(io,"  ↳ Number of iterations (.niter) = $(fout.niter) \n")
    print(io,"  ↳ Minimum (.minimum) = $(fout.minimum) \n")
    print(io,"  ↳ Number of function calls (.feval) = $(fout.feval) \n")
end


end # module

