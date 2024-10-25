
cd("D:\\dev\\Focal.jl")
using Pkg; Pkg.activate(".")

##

using Revise
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
    pipeline::Pipeline
    App2() = new(false, false, time())
end

function update!(app)
    app.busy = true
    p = app.pipeline
    process!(p.demoisaic, app)
    process!(p.whitebalance, p.demoisaic.data, app)
    process!(p.depth, p.whitebalance.data, app)
    process!(p.tonecurve, p.whitebalance.data, app)
    process!(p.render, p.tonecurve.data, app)
    app.busy = false
    p.render
end

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

## start window

screen = Gtk4Makie.GTKScreen(size=(800, 800), title="Focal.jl", aspect = 1)
display(screen, image(rand(1000,1000)))
ax = current_axis()
win = window(screen)

global const toolbox_width = 300

g = grid(screen)
toolbox = GtkBox(:v)
g[2,1] = toolbox

# start app

if !@isdefined app

    global const app = App2()
    app.pipeline = get_pipeline("I:\\photos\\2021\\avril\\crissier dimanche\\0V2A0073.CR3")
    update_display!(app)

end

Gtk4.GLib.g_timeout_add(50) do  # create a function that will be called every 50 milliseconds
    if app.need_update && !app.busy && (time() - app.last_update_time) > 2
        @info "running update via main loop"
        update_display!(app)
        app.need_update = false
        app.last_update_time = time()
    end
    true
end 

# Tone Curve
tonecurve_expander = GtkExpander("Tone curve")
tonecurve_expander.expanded = true
Gtk4.size_request(tonecurve_expander, toolbox_width, 200)

tonecurve_vbox = GtkBox(:v)
tonecurve_expander[] = tonecurve_vbox
push!(toolbox, tonecurve_expander)

function get_scale(parent, min, max, default, label)

    scale = GtkScale(:h, min, max, 0.01)
    Gtk4.size_request(scale, toolbox_width, 10)

    box = GtkBox(:h)
    push!(box, scale)

    box_gesture = GtkGestureClick(box, 0)
    Gtk4.G_.set_propagation_phase(box_gesture, Gtk4.PropagationPhase_CAPTURE)

    Gtk4.value(scale, default)
    push!(parent, GtkLabel(label))
    push!(parent, box)

    scale, box_gesture
end

# contrast scale
#= contrast_scale = GtkScale(:h, 0.5, 5.5, 0.01)
Gtk4.size_request(contrast_scale, toolbox_width, 10)

contrast_box = GtkBox(:h)
push!(contrast_box, contrast_scale)

contrast_box_gesture = GtkGestureClick(contrast_box, 0)
Gtk4.G_.set_propagation_phase(contrast_box_gesture, Gtk4.PropagationPhase_CAPTURE)

Gtk4.value(contrast_scale, 3)
push!(tonecurve_vbox, GtkLabel("Contrast"))
push!(tonecurve_vbox, contrast_box) =#

contrast_scale, contrast_box_gesture = get_scale(tonecurve_vbox, 0.2, 5.8, 3, "Contrast")

signal_connect(contrast_scale, "value-changed") do contrast_scale
    app.pipeline.tonecurve.params.contrast = 1 / Gtk4.value(contrast_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(contrast_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# exposure scale
exposure_scale, exposure_box_gesture = get_scale(tonecurve_vbox, -1.5, 1.5, 0, "Offset")

signal_connect(exposure_scale, "value-changed") do exposure_scale
    app.pipeline.tonecurve.params.exposure = Gtk4.value(exposure_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(exposure_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# lift scale
lift_scale, lift_box_gesture = get_scale(tonecurve_vbox, -0.5, 0.5, 0, "Lift")

signal_connect(lift_scale, "value-changed") do lift_scale
    app.pipeline.tonecurve.params.lift = Gtk4.value(lift_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(lift_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# highlights scale
highlights_scale, highlights_box_gesture = get_scale(tonecurve_vbox, -0.5, 0.5, 0, "Highlights")

signal_connect(highlights_scale, "value-changed") do highlights_scale
    app.pipeline.tonecurve.params.highlights = Gtk4.value(highlights_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(highlights_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# depth scale
depth_scale, depth_box_gesture = get_scale(tonecurve_vbox, -1.5, 1.5, 0, "Depth slope")

signal_connect(depth_scale, "value-changed") do depth_scale
    app.pipeline.tonecurve.params.depth_slope = Gtk4.value(depth_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(depth_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

c = GtkCanvas()
Gtk4.size_request(c,(toolbox_width,200))
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

        params = app.pipeline.tonecurve.params
        exposure, contrast = params.exposure, params.contrast
        lift, highlights = params.lift, params.highlights

        f, ax, p = CairoMakie.barplot(bins[1:end-1], hist)
        CairoMakie.autolimits!(ax) 	
        #yi = sigmoid.(bins, app.pipeline.tonecurve.params.exposure, app.pipeline.tonecurve.params.contrast)
        yi = sigmoid2.(bins, exposure, contrast, lift, highlights)
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
# keep in mind that the "selected" property uses 0 based indexing
uv_transform_dd.selected = 0

signal_connect(uv_transform_dd, "notify::selected") do widget, others...
    idx = widget.selected
    str = Gtk4.selected_string(widget)
    app.pipeline.render.params.uv_transform = Symbol(str)
    update_display!(app)
end

push!(render_vbox, uv_transform_dd)

#

open_button = GtkButton(:icon_name, "document-open")
render_button = GtkButton("Render")
box = GtkBox(:h)
push!(box, open_button, render_button)
g[1,2] = box


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