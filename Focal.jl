
cd("D:\\dev\\Focal.jl")
using Pkg; Pkg.activate(".")

##

using Gtk4, Gtk4Makie, LibRaw, Memoize, Colors, Graphics
using StatsBase
import Colors.FixedPointNumbers

@eval Gtk4 begin
    function _canvas_on_realize(::Ptr, canvas)
        canvas.is_sized && _canvas_on_resize(canvas.handle, width(canvas), height(canvas), canvas)
        nothing
    end
end

using CairoMakie
config = CairoMakie.ScreenConfig(1.0, 1.0, :good, true, false, nothing)
CairoMakie.activate!()

include("src/process_image.jl")

mutable struct App2
    busy::Bool
    need_update::Bool
    last_update_time::Float64
    pipeline::@NamedTuple{demoisaic::Demoisaic{Array{Float64, 3}}, whitebalance::WhiteBalance{Array{Float64, 3}}, tonecurve::ToneCurve{Array{Float64, 3}}, render::Render{Matrix{RGB{FixedPointNumbers.N0f8}}}, demoisaic_params::DemoisaicParams, whitebalance_params::WhiteBalanceParams, tonecurve_params::ToneCurveParams, render_params::RenderParams}
    
    App2() = new(false, false, time())
end

function update!(app)
    app.busy = true
    demoisaic, whitebalance, tonecurve, render, demoisaic_params, whitebalance_params, tonecurve_params, render_params = app.pipeline
    process!(demoisaic, demoisaic_params)
    process!(whitebalance, whitebalance_params, demoisaic.data)
    process!(tonecurve, tonecurve_params, whitebalance.data)
    process!(render, render_params, tonecurve.data)
    app.busy = false
    render
end

if !@isdefined app
    global const app = App2()
end

Gtk4.GLib.g_timeout_add(500) do  # create a function that will be called every 50 milliseconds
    if app.need_update && !app.busy && (time() - app.last_update_time) > 2
        @info "running update via main loop"
        update_display!(app)
        app.need_update = false
        app.last_update_time = time()
    end
    true
end

##

screen = Gtk4Makie.GTKScreen(size=(800, 800), title="Focal.jl", aspect = 1)
display(screen, image(rand(1000,1000)))
ax = current_axis()
win = window(screen)

g = grid(screen)

toolbox = GtkBox(:v)
g[2,1] = toolbox

# Tone Curve
tonecurve_expander = GtkExpander("Tone curve")
tonecurve_expander.expanded = true
Gtk4.size_request(tonecurve_expander, 250, 200)

tonecurve_vbox = GtkBox(:v)
tonecurve_expander[] = tonecurve_vbox
push!(toolbox, tonecurve_expander)

# contrast scale
contrast_scale = GtkScale(:h, 0.1, 2, 0.01)
Gtk4.size_request(contrast_scale, 250, 10)

contrast_box = GtkBox(:h)
push!(contrast_box, contrast_scale)

contrast_box_gesture = GtkGestureClick(contrast_box, 0)
Gtk4.G_.set_propagation_phase(contrast_box_gesture, Gtk4.PropagationPhase_CAPTURE)

Gtk4.value(contrast_scale, 0.25)
push!(tonecurve_vbox, GtkLabel("Contrast"))
push!(tonecurve_vbox, contrast_box)

global exposure = 0
global contrast = 0

signal_connect(contrast_scale, "value-changed") do contrast_scale
    app.pipeline.tonecurve_params.contrast = Gtk4.value(contrast_scale)
    global contrast = Gtk4.value(contrast_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(contrast_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# exposure scale
exposure_scale = GtkScale(:h, -1, 1, 0.01)
Gtk4.size_request(exposure_scale, 250, 10)

exposure_box = GtkBox(:h)
push!(exposure_box, exposure_scale)

exposure_box_gesture = GtkGestureClick(exposure_box, 0)
Gtk4.G_.set_propagation_phase(exposure_box_gesture, Gtk4.PropagationPhase_CAPTURE)

Gtk4.value(exposure_scale, 0)
push!(tonecurve_vbox, GtkLabel("Exposure"))
push!(tonecurve_vbox, exposure_box)

signal_connect(exposure_scale, "value-changed") do exposure_scale
    app.pipeline.tonecurve_params.exposure = Gtk4.value(exposure_scale)
    global exposure = Gtk4.value(exposure_scale)

    c.draw(c)
    reveal(c)
end

signal_connect(exposure_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

c = GtkCanvas()
Gtk4.size_request(c,(250,200))
push!(tonecurve_vbox, c)

if isdefined(app, :pipeline)
    data = mean(app.pipeline.demoisaic.data.img, dims=(3)) |> vec
    @. data = log10(max(data, 1e-16))
    mu = mean(data)
    @. data -= mu

    bins = -2:0.1:2
    hist = fit(Histogram, data, bins)
    hist = hist.weights ./ maximum(hist.weights)
else
    bins = -2:0.1:2
    hist = bins[1:end-1]
end

@guarded draw(c) do widget
    ctx = getgc(c)
    h = Gtk4.height(c)
    w = Gtk4.width(c)
    
    #if isdefined(app, :pipeline)

#=         # clear background
        rectangle(ctx, 0, 0, w, h)
        set_source_rgb(ctx, 1, 1, 1)
        fill(ctx)

        set_source_rgb(ctx, .1, .1, .1)
        for i in 1:length(bins)-1
            rectangle(ctx, (i-1)*w/length(bins), h - hist[i]*h, w/length(bins), hist[i]*h)
            fill(ctx)
        end =#

        f, ax, p = CairoMakie.barplot(bins[1:end-1], hist)
        CairoMakie.autolimits!(ax) 	
        yi = sigmoid.(bins, exposure, contrast)
        lines!(ax, bins, yi)

        screen = CairoMakie.Screen(f.scene, config, Gtk4.cairo_surface(c))
        CairoMakie.resize!(f.scene, Gtk4.width(c), Gtk4.height(c))
        CairoMakie.cairo_draw(screen, f.scene)
    #end
end

# Render

render_expander = GtkExpander("Render")
render_expander.expanded = true
render_vbox = GtkBox(:v)
render_expander[] = render_vbox

push!(toolbox, render_expander)

choices = [:automatic,:rotr90, :rotl90, :rot180, :swap_xy, :transpose, :flip_x, :flip_y, :flip_xy]
uv_transform_dd = GtkDropDown(choices)
# Let's set the active element to be "two", keeping in mind that the "selected" property uses 0 based indexing
uv_transform_dd.selected = 0

signal_connect(uv_transform_dd, "notify::selected") do widget, others...
    idx = widget.selected
    str = Gtk4.selected_string(widget)
    app.pipeline.render_params.uv_transform = Symbol(str)
    update_display!(app)
end

push!(render_vbox, uv_transform_dd)

#

open_button = GtkButton(:icon_name, "document-open")
render_button = GtkButton("Render")
box = GtkBox(:h)
push!(box, open_button, render_button)
g[1,2] = box

function update_display!(app)
    app.busy && return
    render = update!(app)

    empty!(ax)
    img = render.data.img
    ax.aspect = size(img,1)/size(img,2)

    uv_transform = render.params.uv_transform 
    if uv_transform == :automatic
        image!(ax, img)
    else
        image!(ax, img; uv_transform)
    end
end

id = signal_connect(open_button, "clicked") do widget
    open_dialog("Pick a file to open", win) do filename
        
        # initialize a new pipeline
        app.pipeline = get_pipeline(filename)
        update_display!(app)
    end
end

id = signal_connect(render_button, "clicked") do widget
    update_display!(app)
end

show(win)


##

#xi = LinRange(-3,3,100)

#lines(xi, sigmoid.(xi, 0.2,0))