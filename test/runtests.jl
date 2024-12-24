using Test
using LFTSampling
using LFTU1

import LFTSampling: analytic_force, infinitesimal_transformation, get_field
function analytic_force(u1ws::Union{LFTU1.U1Nf2,LFTU1.U1Nf}, hmcws::AbstractHMC)
    LFTU1.force!(u1ws, hmcws)
    frc = hmcws.frc1 .+ hmcws.frc2 .+ hmcws.pfrc
    return frc
end

function infinitesimal_transformation(field_elem, epsilon, lftws2::LFTU1.U1)
    return field_elem * exp(im*epsilon)
end

get_field(u1ws::LFTU1.U1) = u1ws.U


@testset verbose = true "U1 tests" begin

        @testset verbose = true "U1 Nf=2" begin
            include("u1tests.jl")
        end

        @testset verbose = true "U1 Nf=2" begin
            include("u1nf2tests.jl")
        end

        @testset verbose = true "U1 Nf" begin
            include("u1nftests.jl")
            include("u1nfzolotarevtests.jl")
            include("u1nfcorrelatortest.jl")
        end

        @testset verbose = true "U1 I/O" begin
            include("iotests.jl")
        end
end
