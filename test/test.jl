include("../src/process_image.jl")

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
mutable struct Demoisaic <: Processor
    params::Dict
    data::ImageData
end

function process!(p::Demoisaic, params)
    if !need_update(p, params)
        p.data.updated = false
        return
    end
    raw_image, img = get_demoisaic(params[:file])
    p.data.raw_image = raw_image
    p.data.img = img
    p.data.updated = true
    p.data.initialized = true

    p.params = params
    
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
mutable struct WhiteBalance <: Processor
    params::Dict
    data::ImageData
end

function process!(p::WhiteBalance, params, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "White balance"
    raw_image = p.data.raw_image
    
    if !p.data.initialized
        p.data.img = similar(previous.img)
    end
    img = p.data.img
    copyto!(img, previous.img);

    if params[:method] == :as_shot
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
    img = img[:,:,1:3]#FIXME should be in-place
    p.data.img = img #with above

    # convert from camera color space to RGB
    # https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E
    color_matrix = LibRaw.camera_rgb(raw_image)

    LibRaw.apply_maxtrix!(img, color_matrix)

    p.data.updated = true
    p.data.initialized = true
    p.params = params
    p
end

## Tone curve

sigmoid(x, σ, μ) = 1 / (1 + exp(-(x-μ)/σ))

mutable struct ToneCurve <: Processor
    params::Dict
    data::ImageData
end

function process!(p::ToneCurve, params, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "Tone Curve"
    exposure, contrast = params[:exposure], params[:contrast]

    if !p.data.initialized
        p.data.img = zeros(size(previous.img,1), size(previous.img,2), size(previous.img,3))
    end
    img = p.data.img
    copyto!(img, previous.img[:,:,1:3])
    
    if params[:method] == :log_sigmoid
        @. img = log10(max(img, 1e-16))
        mu = mean(img)
        @. img = sigmoid(img - mu, exposure, contrast)
    elseif params[:method] == :linear
        img = img ./ maximum(scaled)
        @. img = img(scaled, 0, 1)
    end

    p.data.updated = true
    p.data.initialized = true
    p.params = params
    p

end

## Render

mutable struct Render <: Processor
    params::Dict
    data::ImageData
end

function process!(p::Render, params, previous::ImageData)

    if !need_update(p, params, previous)
        p.data.updated = false
        return
    end

    @info "Render"
    raw_image = p.data.raw_image
    
    if !p.data.initialized
        p.data.img = fill(Colors.RGB(1,1,1), size(previous.img,1), size(previous.img,2))
    end
    img = p.data.img

    toned = previous.img
    for i in axes(img,1), j in axes(img,2)
        img[i,j] = Colors.RGB(toned[i,j,1], toned[i,j,2], toned[i,j,3])
    end

    p.data.updated = true
    p.data.initialized = true
    p.params = params
    p

end

##


file = Dict(:file => "J:\\photo2\\2024\\Juin\\Manif contre le FN\\0V2A9704.CR3")
wb = Dict(:method => :as_shot)
tone_curve_params = Dict(:method => :log_sigmoid, :contrast => 0.5, :exposure => 0.5)

raw_image = LibRaw.RawImage(file[:file])
w, h = LibRaw.raw_width(raw_image), LibRaw.raw_height(raw_image)

demoisaic = Demoisaic(
    Dict(:file => ""),
    ImageData(raw_image, zeros(w,h,4), true, false)
)
white_balance = WhiteBalance(
    Dict(:method => :as_shot),
    ImageData(raw_image, zeros(1,1,4), true, false)
)
tone_curve = ToneCurve(
    tone_curve_params,
    ImageData(raw_image, zeros(w,h,3), true, false)
)
render = Render(
    Dict(),
    ImageData(raw_image, fill(Colors.RGB(1,1,1), 1, 1), true, false)
)

process!(demoisaic, file)
process!(white_balance, wb, demoisaic.data)
process!(tone_curve, tone_curve_params, white_balance.data)
process!(render, Dict(), tone_curve.data)

image(render.data.img)
image(white_balance.data.img[:,:,3])
image(tone_curve.data.img[:,:,3])

#@time raw_image, d = get_demoisaic(file)parama