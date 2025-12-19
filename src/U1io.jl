import BDIO: BDIO_write!, BDIO_read
import LFTSampling: save_cnfg_header, read_cnfg_info

function BDIO.BDIO_write!(fb::BDIO.BDIOstream, u1ws::U1)
    for mu in 1:2, iy in 1:u1ws.params.iL[2], ix in 1:u1ws.params.iL[1]
        BDIO.BDIO_write!(fb,[real(u1ws.U[ix,iy,mu])])
        BDIO.BDIO_write!(fb,[imag(u1ws.U[ix,iy,mu])])
    end
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream, u1ws::U1, reg::Array{Float64, 1})
    length(reg) != 2 && error("Register reg must be of length 2")

    for mu in 1:2, iy in 1:u1ws.params.iL[2], ix in 1:u1ws.params.iL[1]
        BDIO.BDIO_read(fb,reg)
        u1ws.U[ix,iy,mu] = complex(reg[1], reg[2])
    end
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream, u1ws::U1)
    reg = zeros(Float64,2)
    BDIO.BDIO_read(fb, u1ws, reg)
end


function save_cnfg_header(fb::BDIO.BDIOstream, u1ws::U1Quenched)
    if u1ws.params.BC == PeriodicBC
        BC = 0
    elseif u1ws.params.BC == OpenBC
        BC = 1
    end
    BDIO.BDIO_write!(fb, [u1ws.params.beta])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[2])])
    BDIO.BDIO_write!(fb, [convert(Int32, BC)])
    BDIO.BDIO_write_hash!(fb)
    return nothing
end

function save_cnfg_header(fb::BDIO.BDIOstream, u1ws::U1Nf2)
    if u1ws.params.BC == PeriodicBC
        BC = 0
    elseif u1ws.params.BC == OpenBC
        BC = 1
    end
    BDIO.BDIO_write!(fb, [u1ws.params.beta])
    BDIO.BDIO_write!(fb, [u1ws.params.am0])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[2])])
    BDIO.BDIO_write!(fb, [convert(Int32, BC)])
    BDIO.BDIO_write_hash!(fb)
    return nothing
end


"""
    read_cnfg_info(fname::String, ::Type{U1Quenched})

reads theory parameters from `fname` and returns `fb::BDIOstream` and a model
instance with the read parameters
"""
function read_cnfg_info(fname::String, ::Type{U1Quenched})

    fb = BDIO.BDIO_open(fname, "r")

    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end

    ifoo    = Vector{Float64}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    beta    = ifoo[1]
    ifoo    = Vector{Int32}(undef, 3)
    BDIO.BDIO_read(fb, ifoo)
    lsize1   = convert(Int64, ifoo[1])
    lsize2   = convert(Int64, ifoo[2])
    BC      = convert(Int64, ifoo[3])

    if BC == 0
        BCt = PeriodicBC
    elseif BC == 1
        BCt = OpenBC
    end

    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize1, lsize2),
                       BC = BCt,
                      )

    return fb, model
end


"""
Versions:
- 0.1: Square lattice only, lsize2 = lsize1
- 0.2: Rectangular lattice
"""
function read_cnfg_info(fname::String, ::Type{U1Nf2}; v = "0.2")

    fb = BDIO.BDIO_open(fname, "r")

    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end

    ifoo    = Vector{Float64}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    beta    = ifoo[1]
    BDIO.BDIO_read(fb, ifoo)
    mass    = ifoo[1]
    ifoo    = Vector{Int32}(undef, 3)
    BDIO.BDIO_read(fb, ifoo)
    lsize1   = convert(Int64, ifoo[1])
    lsize2   = convert(Int64, ifoo[2])
    if v == "0.1"
        BC = lsize2
        lsize2 = lsize1
    elseif v == "0.2"
        BC      = convert(Int64, ifoo[3])
    else
        error("Current I/O versions are 0.1 and 0.2")
    end

    if BC == 0
        BCt = PeriodicBC
    elseif BC == 1
        BCt = OpenBC
    end

    model = U1Nf2(Float64,
                       beta = beta,
                       am0 = mass,
                       iL = (lsize1, lsize2),
                       BC = BCt,
                      )

    return fb, model
end

function save_cnfg_header(fb::BDIO.BDIOstream, u1ws::U1Nf)
    if u1ws.params.BC == PeriodicBC
        BC = 0
    elseif u1ws.params.BC == OpenBC
        BC = 1
    end
    BDIO.BDIO_write!(fb, [u1ws.params.beta])
    BDIO.BDIO_write!(fb, [convert(Int32, length(u1ws.params.am0))])
    BDIO.BDIO_write!(fb, u1ws.params.am0)
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[2])])
    BDIO.BDIO_write!(fb, [convert(Int32, BC)])
    BDIO.BDIO_write_hash!(fb)
    return nothing
end

function read_cnfg_info(fname::String, ::Type{U1Nf})

    fb = BDIO.BDIO_open(fname, "r")

    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end

    ifoo    = Vector{Float64}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    beta    = ifoo[1]
    ifoo    = Vector{Int32}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    N_fermions = convert(Int64, ifoo[1])
    masses  = Vector{Float64}(undef, N_fermions)
    BDIO.BDIO_read(fb, masses)
    ifoo    = Vector{Int32}(undef, 3)
    BDIO.BDIO_read(fb, ifoo)
    lsize1   = convert(Int64, ifoo[1])
    lsize2   = convert(Int64, ifoo[2])
    BC      = convert(Int64, ifoo[3])

    if BC == 0
        BCt = PeriodicBC
    elseif BC == 1
        BCt = OpenBC
    end

    model = U1Nf(Float64,
                       beta = beta,
                       am0 = masses,
                       iL = (lsize1, lsize2),
                       BC = BCt,
                      )

    return fb, model
end
