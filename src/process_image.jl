using LibRaw, Colors, Statistics

to_color(x, i, j) = Colors.RGB(x[i,j,1], x[i,j,2], x[i,j,3])
to_color(x) = [to_color(x, i, j) for i in axes(x,1), j in axes(x,2)]

sigmoid(x, σ, μ) = 1 / (1 + exp(-(x-μ)/σ))

function get_demoisaic(file)
    @info "reading file"
    img = LibRaw.RawImage(file)

    LibRaw.unpack!(img)
    LibRaw.subtract_black!(img) # this doesn't seem to do much

    @info "demoisaic"
    @time d = LibRaw.demoisaic(LibRaw.BayerAverage(), img)
    d = d ./ (LibRaw.maximum(img) - LibRaw.black_level(img))

    img, d
end

function process_raw(file, exposure, contrast)

    img, d = get_demoisaic(file)

    wb = :as_shot

    if wb == :as_shot
        # apply white balance as shot
        mult = LibRaw.camera_multipliers(img)
        LibRaw.apply_multipliers!(d, mult);
    else
        # daylight white balance
        mult = LibRaw.pre_multipliers(img)
        mult[4] = mult[2]
        LibRaw.apply_multipliers!(d, mult)
    end

    # image is RGBG, average the two green channels
    @assert LibRaw.color_description(img) == "RGBG"

    @. d[:,:,2] = d[:,:,2]/2 + d[:,:,4]/2
    d = d[:,:,1:3]

    # convert from camera color space to RGB
    # https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E
    color_matrix = LibRaw.camera_rgb(img)

    @info "White balance"
    @time LibRaw.apply_maxtrix!(d, color_matrix)

    # apply tone curve
    method = :log_sigmoid

    if method == :log_sigmoid
        out = log10.(max.(d[2:end-1, 2:end-1, 1:3], 1e-16) )
        out .-= mean(out)
        scaled = sigmoid.(out, exposure, contrast)
    elseif method == :linear
        scaled = d[2:end-1, 2:end-1, 1:3]
        scaled = scaled ./ maximum(scaled)
        scaled = clamp.(scaled, 0, 1)
    end

    @time col = [to_color(scaled,i,j) for i in axes(scaled,1), j in axes(scaled,2)]
end