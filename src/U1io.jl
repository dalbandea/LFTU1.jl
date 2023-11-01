import BDIO: BDIO_write!, BDIO_read

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

"""
    read_next_cnfg(fb::BDIO.BDIOstream, u1ws::U1)

reads next configuration of BDIO handle `fb` and stores it into `u1ws.U`.
"""
function read_next_cnfg(fb::BDIO.BDIOstream, u1ws::U1)
    while BDIO.BDIO_get_uinfo(fb) != 8
        BDIO.BDIO_seek!(fb)
    end
    BDIO.BDIO_read(fb, u1ws)
end

"""
    save_cnfg(fname::String, u1ws::U1)

saves model instance `u1ws` to BDIO file `fname`. If file does not exist, it creates one and stores the info in `u1ws.params`, and then saves the configuration. If it does exist, it appends the configuration to the existing file.
"""
function save_cnfg(fname::String, u1ws::U1)
    if isfile(fname)
        fb = BDIO.BDIO_open(fname, "a")
    else
        fb = save_cnfg_header(fname, u1ws)
    end

    BDIO.BDIO_start_record!(fb, BDIO.BDIO_BIN_F64LE, 8, true)
    BDIO.BDIO_write!(fb,u1ws)
    BDIO.BDIO_write_hash!(fb)
    BDIO.BDIO_close!(fb)
end

function save_cnfg_header(fname::String, u1ws::U1Quenched)
        fb = BDIO.BDIO_open(fname, "w", "U1 Configurations")

        BDIO.BDIO_start_record!(fb, BDIO.BDIO_BIN_GENERIC, 1)
        if u1ws.params.BC == PeriodicBC
            BC = 0
        elseif u1ws.params.BC == OpenBC
            BC = 1
        end
        BDIO.BDIO_write!(fb, [u1ws.params.beta])
        BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
        BDIO.BDIO_write!(fb, [convert(Int32, BC)])
        BDIO.BDIO_write_hash!(fb)
        return fb
end

function save_cnfg_header(fname::String, u1ws::U1Nf2)
        fb = BDIO.BDIO_open(fname, "w", "U1 Configurations")

        BDIO.BDIO_start_record!(fb, BDIO.BDIO_BIN_GENERIC, 1)
        if u1ws.params.BC == PeriodicBC
            BC = 0
        elseif u1ws.params.BC == OpenBC
            BC = 1
        end
        BDIO.BDIO_write!(fb, [u1ws.params.beta])
        BDIO.BDIO_write!(fb, [u1ws.params.am0])
        BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
        BDIO.BDIO_write!(fb, [convert(Int32, BC)])
        BDIO.BDIO_write_hash!(fb)
        return fb
end

"""
    read_cnfg_info(fname::String, ::Type{AbstractLFT})

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
    ifoo    = Vector{Int32}(undef, 2)
    BDIO.BDIO_read(fb, ifoo)
    lsize   = convert(Int64, ifoo[1])
    BC      = convert(Int64, ifoo[2])

    if BC == 0
        BCt = PeriodicBC
    elseif BC == 1
        BCt = OpenBC
    end

    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = BCt,
                      )

    return fb, model
end

function read_cnfg_info(fname::String, ::Type{U1Nf2})

    fb = BDIO.BDIO_open(fname, "r")

    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end

    ifoo    = Vector{Float64}(undef, 2)
    BDIO.BDIO_read(fb, ifoo)
    beta    = ifoo[1]
    mass    = ifoo[2]
    ifoo    = Vector{Int32}(undef, 2)
    BDIO.BDIO_read(fb, ifoo)
    lsize   = convert(Int64, ifoo[1])
    BC      = convert(Int64, ifoo[2])

    if BC == 0
        BCt = PeriodicBC
    elseif BC == 1
        BCt = OpenBC
    end

    model = U1Nf2(Float64,
                       beta = beta,
                       am0 = mass,
                       iL = (lsize, lsize),
                       BC = BCt,
                      )

    return fb, model
end
