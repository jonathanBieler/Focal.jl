using LibRaw, Colors, Statistics
using LoopVectorization
using ImageTransformations, ONNXRunTime, ImageCore


abstract type Processor
end

function have_params_changed(p::Processor, params)
    p.previous_params != params
end

function need_update(p::Processor, params, input)
    input.updated && return true
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
    previous_params::DemoisaicParams
    data::ImageData{T}
end

function process!(p::Demoisaic, app)
    params = p.params
    if !need_update(p, params)
        p.data.updated = false
        return
    end
    raw_image, img = get_demoisaic(params.file)

    h,w,c = size(img)
    h,w = div(h,2), div(w,2)
    img = imresize(img, (h,w,c))

    p.data.raw_image = raw_image
    p.data.img = img
    p.data.updated = true
    p.data.initialized = true

    p.previous_params = deepcopy(p.params)
    
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
    previous_params::WhiteBalanceParams
    data::ImageData{T}
end

function process!(p::WhiteBalance, input::ImageData, app)

    params = p.params
    if !need_update(p, params, input)
        p.data.updated = false
        return
    end

    @info "White balance"
    raw_image = p.data.raw_image
    
#=     if !p.data.initialized
        p.data.img = similar(input.img)
    end =#
    img = p.data.img
    copyto!(img, input.img);

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
    p.previous_params = deepcopy(p.params)
    p
end

## Depth estimation

if !@isdefined da
    @info "Loading depth_anything_v2"
    da = ONNXRunTime.load_inference("D:\\dev\\Depth-Anything-ONNX\\weights\\depth_anything_v2_vitb_17.onnx")
end

mutable struct DepthParams
end
Base.:(==)(x::DepthParams, y::DepthParams) = true

mutable struct Depth{T} <: Processor
    params::DepthParams
    previous_params::DepthParams
    data::ImageData{T}
    depth::Matrix{Float32}
end

function process!(p::Depth, input::ImageData, app)
    params = p.params
    if !need_update(p, params, input)
        p.data.updated = false
        return
    end

    @info "Depth"
    raw_image = p.data.raw_image

    img = p.data.img
    copyto!(img, @view input.img[:,:,1:3])
    p.depth = estimate_depth(img,da)

    p.data.updated = true
    p.data.initialized = true
    p.previous_params = deepcopy(p.params)
end

function estimate_depth(img, model; normalize = true)

    dh, dw = 1400,924
    
    h,w = size(img,1), size(img,2)
    img = imresize(img, (dh,dw,3))
    img = PermutedDimsArray(img, (3, 1, 2))
    img = reshape(Float32.(img), (1,3,dh,dw))
    input = Dict("image" => img)

    depth = model(input)["depth"][1,:,:]
    if normalize
        depth = depth .-  minimum(depth)
        depth = depth ./  maximum(depth)
    end
    imresize(depth, (h,w))
end

## Tone curve

sigmoid(x, μ, σ) = 1 / (1 + exp(-(x-μ)/σ))

# see plots in test.jl
sigma(σ, x, lift, highlights) = σ + highlights*sigmoid(x - 3σ, 0, √σ/1.5) + lift*sigmoid(-(x + 3σ), 0, √σ/1.5)
sigmoid2(x, μ, σ, lift, highlights) = 1 / (1 + exp(-(x-μ)/sigma(σ, x-μ, lift, highlights))) 

mutable struct ToneCurveParams
    method::Symbol
    contrast::Float64
    exposure::Float64
    lift::Float64
    highlights::Float64
    depth_slope::Float64
end
function Base.:(==)(x::ToneCurveParams, y::ToneCurveParams) 
    x.method == y.method && x.contrast == y.contrast && x.exposure == y.exposure &&
    x.lift == y.lift && x.highlights == y.highlights && x.depth_slope == y.depth_slope
end

mutable struct ToneCurve{T} <: Processor
    params::ToneCurveParams
    previous_params::ToneCurveParams
    data::ImageData{T}
end

@guarded function process!(p::ToneCurve, input::ImageData, app)

    params = p.params
    if !need_update(p, params, input)
        p.data.updated = false
        return
    end

    @info "Tone Curve"
    exposure, contrast = params.exposure, params.contrast
    lift, highlights = params.lift, params.highlights
    depth_slope = params.depth_slope
    @info depth_slope

    if contrast + lift < 0
        lift = -contrast + 0.01
    end
    if highlights + lift < 0
        highlights = -contrast + 0.01
    end

#=     if !p.data.initialized
        p.data.img = zeros(size(input.img,1), size(input.img,2), size(input.img,3))
    end =#
    img = p.data.img
    copyto!(img, @view input.img[:,:,1:3])
    
    #depth = app.pipeline.depth.depth
    #depth = zeros(size(img)) .+ depth_slope
    depth = depth_slope * Float64.(app.pipeline.depth.depth)

    if params.method == :log_sigmoid
        @tturbo for i in eachindex(img)
            img[i] = log10(max(img[i], 1e-16))
        end
        mu = mean(img)
        # @tturbo
        for i in axes(img,1), j in axes(img,2), c in axes(img,3)
            #img[i] = sigmoid(img[i] - mu, exposure, contrast)
            img[i,j,c] = sigmoid2(img[i,j,c] - mu, exposure + depth[i,j], contrast, lift, highlights)
            #img[i] = sigmoid2(img[i] - mu, exposure, contrast, lift, highlights)
        end

    elseif params.method == :linear
        m = maximum(img)
        @. img = img / m 
        @. img = clamp(img, 0, 1)
    end

    p.data.updated = true
    p.data.initialized = true
    p.previous_params = deepcopy(p.params)
    p
end

## Render

mutable struct RenderParams
    uv_transform::Symbol
end

mutable struct Render{T} <: Processor
    params::RenderParams
    previous_params::RenderParams
    data::ImageData{T}
end

function process!(p::Render, input::ImageData, app)

    params = p.params
    if !need_update(p, params, input)
        p.data.updated = false
        return
    end

    @info "Render"
    raw_image = p.data.raw_image
    
    #=     if !p.data.initialized
        p.data.img = fill(Colors.RGB(1,1,1), size(input.img,1), size(input.img,2))
    end =#
    img = p.data.img

    toned = input.img
    for j in axes(img,2), i in axes(img,1)
        img[i,j] = Colors.RGB(toned[i,j,1], toned[i,j,2], toned[i,j,3])
    end

    p.data.updated = true
    p.data.initialized = true
    p.previous_params = deepcopy(p.params)
    p
end

mutable struct Pipeline
    raw_image::LibRaw.RawImage
    demoisaic::Demoisaic
    whitebalance::WhiteBalance
    depth::Depth
    tonecurve::ToneCurve
    render::Render
end

function get_pipeline(file)

    raw_image = LibRaw.RawImage(file)
    w, h = LibRaw.raw_width(raw_image)-2, LibRaw.raw_height(raw_image)-2 # I remove 2 pixels
    h,w = div(h,2), div(w,2) 

    demoisaic_params = DemoisaicParams(file)
    whitebalance_params = WhiteBalanceParams(:as_shot)
    depth_params = DepthParams()
    tonecurve_params = ToneCurveParams(:log_sigmoid, 0.5, 0.5, 0.0, 0.0, 0.0)
    render_params = RenderParams(:automatic)

    demoisaic = Demoisaic(
        demoisaic_params,
        deepcopy(demoisaic_params),
        ImageData(raw_image, zeros(h,w,4), true, false)
    )
    whitebalance = WhiteBalance(
        whitebalance_params,
        deepcopy(whitebalance_params),
        ImageData(raw_image, zeros(h,w,4), true, false)
    )
    depth = Depth(
        depth_params,
        deepcopy(depth_params),
        ImageData(raw_image, zeros(h,w,3), true, false),
        zeros(Float32,h,w)
    )
    tonecurve = ToneCurve(
        tonecurve_params,
        deepcopy(tonecurve_params),
        ImageData(raw_image, zeros(h,w,3), true, false)
    )
    render = Render(
        render_params,
        deepcopy(render_params),
        ImageData(raw_image, fill(Colors.RGB(1,1,1), h, w), true, false)
    )

    Pipeline(
        raw_image,
        demoisaic,
        whitebalance,
        depth,
        tonecurve,
        render,
    )

end