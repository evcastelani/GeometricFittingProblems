module GeometricFittingProblems


using DelimitedFiles, LinearAlgebra, Plots

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

function CGAHypersphere(data;ε = 1.0e-4)
    (N,n) = size(data)
    D = [data';ones(1,N)]
    v = [0.5*norm(D[1:n,i] ,2)^2 for i=1:N ]
    D = [D ; v']
    DDt = D*D'
    M = zeros(n+2,n+2)
    for i=1:n
        M[i,i] = 1.0
    end
    M[n+1,n+2] = -1.0
    M[n+2,n+1] = -1.0
    p = (1.0/N)
    P = p.*(DDt*M)
    F = eigen(P)
    indmin = 1
    valmin = F.values[1]
    for i = 2:n
        if abs(valmin)>abs(F.values[i])
            if F.values[i]>-ε   
                indmin = i
                valmin = F.values[i] 
            end
        end
    end
    if valmin<-ε
        error("P does not have postive eigen value!")
    end
    xnorm = (1.0/(F.vectors[:,indmin][end-1]))*F.vectors[:,indmin]
    center = xnorm[1:end-2]
    
    return push!(center,√(norm(center,2)^2 -2.0*xnorm[end]))
end

function sort_sphere_res(P,x,nout)
    n = length(P[:,1])
    m = length(P[1,:])
    v = zeros(n)
    for i=1:n
        for j=1:m
            v[i] = v[i]+(P[i,j]-x[j])^2
        end
        v[i] = (v[i]-x[end]^2)^2
    end
    indtrust = [1:n;]
    for i=1:n-nout+1 
        for j=i+1:n
            if v[i]>v[j]
                aux = v[j]
                v[j] = v[i]
                v[i] = aux
                
                aux2 = indtrust[j]
                indtrust[j] = indtrust[i]
                indtrust[i] = aux2
            end
        end
    end
#    println(indtrust[n-nout+1:n])
    return P[indtrust[1:n-nout],:], sum(v[1:n-nout])
end

function LOVOCGAHypersphere(data,nout,θ,ε=1.0e-4)

    ordres = sort_sphere_res(data,θ,nout)
    k = 1
    antres = 0.0
    while abs(ordres[2]-antres) > ε
        display(ordres[2])
        antres = ordres[2]
        θ = CGAHypersphere(ordres[1])
        println(θ)
        ordres = sort_sphere_res(data,θ,nout)
        k = k+1
    end
 #   display(k)
    return θ

end


"""
    solve :: Function

This functions is able to solve a fitting problem previous loaded.

# Example
```
julia-repl
julia> prob =  load_problem("sphere2D_50.0_50.0_8.0_10.csv")

julia> solve(prob,"CGA-Hypersphere")

```
return the smallest positive eigen and eigen vector associated which give to us the desired parameters.

# Example
```
julia-repl
julia> prob =  load_problem("sphere2D_50.0_50.0_8.0_10.csv")

julia> solve(prob,"LOVO-CGA-Hypersphere")

```
return the desired parameters.

another way to run is using a particular initial guess.

# Example
```
julia-repl
julia> prob =  load_problem("sphere2D_50.0_50.0_8.0_10.csv")

julia> solve(prob,"LOVO-CGA-Hypersphere",rand(3))

```

"""
function solve(prob::FitProbType,method::String, initθ = CGAHypersphere(prob.data))
    if method == "CGA-Hypersphere"
        return CGAHypersphere(prob.data)
    end
    if method == "LOVO-CGA-Hypersphere"
        LOVOCGAHypersphere(prob.data,prob.nout,initθ)
    end
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
        FileMatrix = ["name :" "sphere2D";"data :" [[x y]]; "npts :" npts;"nout :" nout; "model :" "(x,t) -> (x[1]-t[1])^2 + (x[2]-t[2])^2 - t[3]^2";"dim :" 3; "cluster :" "false"; "noise :" "false"; "solution :" [push!(c,r)]; "description :" "none"]

        open("sphere2D_$(c[1])_$(c[2])_$(c[3])_$(nout).csv", "w") do io
            writedlm(io, FileMatrix)
        end

    end
    if probtype == "sphere3D"
        println("params need to be setup as [center,radious,npts,nout]")
        c = [params[1], params[2], params[3]]
        r = params[4]
        npts = Int(params[5])
        x = zeros(npts)
        y = zeros(npts)
        z = zeros(npts)
        θ = [0.0:2*π/(npts-1):2*π;]
        φ = [0.0:π/(npts-1):π;]
        for k = 1:npts #forma de espiral - ao criar outro forma, se obtem metade dos circulos máximos
            x[k] = c[1] + r * cos(θ[k]) * sin(φ[k])
            y[k] = c[2] + r * sin(θ[k]) * sin(φ[k])
            z[k] = c[3] + r * cos(φ[k])
        end
        nout = Int(params[6])
        k = 1
        iout = []
        while k <= nout
            i = rand([1:npts;])
            if i ∉ iout
                push!(iout, i)
                k = k + 1
            end
        end
        for k = 1:nout
            x[iout[k]] = x[iout[k]] + rand([0.25*r:0.1*(r); (1 + 0.25) * r])
            y[iout[k]] = y[iout[k]] + rand([0.25*r:0.1*(r); (1 + 0.25) * r])
            z[iout[k]] = z[iout[k]] + rand([0.25*r:0.1*(r); (1 + 0.25) * r])
        end
        FileMatrix = ["name :" "sphere3D"; "data :" [[x y z]]; "npts :" npts; "nout :" nout; "model :" "(x,t) -> (x[1]-t[1])^2 + (x[2]-t[2])^2 +(x[3]-t[3])^2 - t[4]^2"; "dim :" 4; "cluster :" "false"; "noise :" "false"; "solution :" [push!(c, r)]; "description :" "none"]

        open("sphere3D_$(c[1])_$(c[2])_$(c[3])_$(c[4])_$(nout).csv", "w") do io #o que essa linha faz exatamente?
            writedlm(io, FileMatrix)
        end
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

function visualize(prob,answer)
    
    plt = plot()
    if prob.name == "sphere2D" || prob.name == "\tsphere2D"
        plot!(plt,prob.data[:,1],prob.data[:,2],line=:scatter,aspect_ratio=:equal)
        θ = [0.0:2*π/360:2*π;]
        x = answer[1].+answer[3]*cos.(θ)
        y = answer[2].+answer[3]*sin.(θ)
        plot!(plt,x,y)
        display(plt)
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

