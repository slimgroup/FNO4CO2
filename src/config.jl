module Config

export get_parameters

function get_parameters()
    return Dict(
        "nx" => 256,
        "nz" => 128,
        "n_offsets" => 1,
        "down_rate" => 2,
        "read_offset_start" => 10,
        "offset_start" => -500.0,
        "offset_end" => 500.0,
        "f0" => 0.015f0,
        "timeD" => 3200f0,
        "timeR" => 3200f0,
        "TD" => 3200f0,
        "dtD" => 4f0,
        "dtS" => 4f0,
        "nbl" => 120,
        "d" => (12.5f0, 12.5f0),
        "o" => (0f0, 0f0),
        # "n" => (size(m_train, 1), size(m_train, 2)),
        "nsrc" => 64,
        "n_samples" => 64,
        # "nxrec" => size(m_train, 1),
        "snr" => 12f0,
    )
end

end
