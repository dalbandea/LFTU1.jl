using Test
using LFTU1

@testset verbose = true "U1 tests" begin

    @testset verbose = true "U1 Nf=2" begin
        include("u1nf2tests.jl")
    end

    @testset verbose = true "U1 I/O" begin
        include("iotests.jl")
    end

end
