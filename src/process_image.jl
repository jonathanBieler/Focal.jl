using LibRaw, Colors, Statistics
using LoopVectorization

abstract type Processor
end

function have_params_changed(p::Processor, params)
    p.params != params
end

function need_update(p::Processor, params, previous)
    previous.updated && return true
    need_update(p, params)
end

function need_update(p::Processor, params)
    !p.data.initialized && return true
    have_params_changed(p, params) && return true
    false
end

mutable struct ImageData{T}
    raw_image::LibRaw.RawImage
    img::T
    updated::Bool
    initialized::Bool
end

## Demoisaic

mutable struct DemoisaicParams
    file::String
end
Base.:(==)(x::DemoisaicParams, y::DemoisaicParams) = x.file == y.file

mutable struct Demoisaic{T} <: Processor
    params::DemoisaicParams
    data::ImageData{T}
end

function process!(p::Demoisaic, params::DemoisaicParams)
    if !need_update(p, params)
        p.data.updated = false
        return
    end
    raw_image, img = get_demoisaic(params.file)
    p.data.raw_image = raw_image
    p.data.img = img
    p.data.updated = true
    p.data.initialized = true

    p.params = deepcopy(params)
    
    p
end

function get_demoisaic(file)
    @info "reading file"
    raw_image = LibRaw.RawImage(file)

    LibRaw.unpack!(raw_image)
    LibRaw.subtract_black!(raw_image) # this doesn't seem to do much

    @info "demoisaic"
    @time img = LibRaw.demoisaic(LibRaw.BayerAverage(), raw_image)
    img = img ./ (LibRaw.maximum(raw_image) - LibRaw.black_level(raw_image))

    img = img[2:end-1, 2:end-1,:]

    raw_image, img
end

## WhiteBalance

mutable struct WhiteBalanceParams
    method::Symbol
end
Base.:(==)(x::WhiteBalanceParams, y::WhiteBalanceParams) = x.method == y.method

mutable struct WhiteBalance{T} <: Processor
    params::WhiteBalanceParams
    data::ImageData{T}
end

function process!(p::WhiteBalance, params::WhiteBalanceParams, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "White balance"
    raw_image = p.data.raw_image
    
#=     if !p.data.initialized
        p.data.img = similar(previous.img)
    end =#
    img = p.data.img
    copyto!(img, previous.img);

    if params.method == :as_shot
        # apply white balance as shot
        mult = LibRaw.camera_multipliers(raw_image)
        LibRaw.apply_multipliers!(img, mult);
    else
        # daylight white balance
        mult = LibRaw.pre_multipliers(raw_image)
        mult[4] = mult[2]
        LibRaw.apply_multipliers!(img, mult)
    end

    # image is RGBG, average the two green channels
    @assert LibRaw.color_description(raw_image) == "RGBG"
    @. img[:,:,2] = img[:,:,2]/2 + img[:,:,4]/2
    #img = img[:,:,1:3]#FIXME should be in-place
    p.data.img = img #with above

    # convert from camera color space to RGB
    # https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E
    color_matrix = LibRaw.camera_rgb(raw_image)

    LibRaw.apply_maxtrix!(img, color_matrix)

    p.data.updated = true
    p.data.initialized = true
    p.params = deepcopy(params)
    p
end

## Tone curve

sigmoid(x, μ, σ) = 1 / (1 + exp(-(x-μ)/σ))

mutable struct ToneCurveParams
    method::Symbol
    contrast::Float64
    exposure::Float64
end
Base.:(==)(x::ToneCurveParams, y::ToneCurveParams) = x.method == y.method && x.contrast == y.contrast && x.exposure == y.exposure

mutable struct ToneCurve{T} <: Processor
    params::ToneCurveParams
    previous_params::ToneCurveParams
    data::ImageData{T}
end

function process!(p::ToneCurve, params::ToneCurveParams, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "Tone Curve"
    exposure, contrast = params.exposure, params.contrast

#=     if !p.data.initialized
        p.data.img = zeros(size(previous.img,1), size(previous.img,2), size(previous.img,3))
    end =#
    img = p.data.img
    copyto!(img, @view previous.img[:,:,1:3])
    
    if params.method == :log_sigmoid
        @tturbo for i in eachindex(img)
            img[i] = log10(max(img[i], 1e-16))
        end
        mu = mean(img)
        @tturbo for i in eachindex(img)
            img[i] = sigmoid(img[i] - mu, exposure, contrast)
        end

    elseif params.method == :linear
        m = maximum(img)
        @. img = img / m 
        @. img = clamp(img, 0, 1)
    end

    p.data.updated = true
    p.data.initialized = true
    p.params = deepcopy(params)
    p
end

## Render

mutable struct RenderParams
    uv_transform::Symbol
end

mutable struct Render{T} <: Processor
    params::RenderParams
    data::ImageData{T}
end

function process!(p::Render, params::RenderParams, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "Render"
    raw_image = p.data.raw_image
    
    #=     if !p.data.initialized
        p.data.img = fill(Colors.RGB(1,1,1), size(previous.img,1), size(previous.img,2))
    end =#
    img = p.data.img

    toned = previous.img
    for j in axes(img,2), i in axes(img,1)
        img[i,j] = Colors.RGB(toned[i,j,1], toned[i,j,2], toned[i,j,3])
    end

    p.data.updated = true
    p.data.initialized = true
    p.params = deepcopy(params)
    p
end

mutable struct Pipeline
    raw_image
    demoisaic::Demoisaic
    whitebalance::WhiteBalance
    tonecurve::ToneCurve
    render::Render
end

function get_pipeline(file)

    raw_image = LibRaw.RawImage(file)
    w, h = LibRaw.raw_width(raw_image)-2, LibRaw.raw_height(raw_image)-2 # I remove 2 pixels

    demoisaic_params = DemoisaicParams(file)
    whitebalance_params = WhiteBalanceParams(:as_shot)
    tonecurve_params = ToneCurveParams(:log_sigmoid, 0.5, 0.5)
    render_params = RenderParams(:automatic)

    demoisaic = Demoisaic(
        demoisaic_params,
        ImageData(raw_image, zeros(h,w,4), true, false)
    )
    whitebalance = WhiteBalance(
        whitebalance_params,
        ImageData(raw_image, zeros(h,w,4), true, false)
    )
    tonecurve = ToneCurve(
        tonecurve_params,
        ImageData(raw_image, zeros(h,w,3), true, false)
    )
    render = Render(
        render_params,
        ImageData(raw_image, fill(Colors.RGB(1,1,1), h, w), true, false)
    )

    (;demoisaic, whitebalance, tonecurve, render, demoisaic_params, whitebalance_params, tonecurve_params, render_params)
end