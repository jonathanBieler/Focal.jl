using Revise
using Gtk4, Gtk4Makie, LibRaw, Colors, Graphics
using StatsBase, Statistics
import Colors.FixedPointNumbers

using ONNXRunTime, FileIO, ImageCore
using CUDA
import cuDNN

using LoopVectorization
using ImageTransformations, ImageCore

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
include("src/depth_estimation.jl")
include("src/ui.jl")

mutable struct App
    busy::Bool
    need_update::Bool
    last_update_time::Float64
    pipeline::Pipeline
    histogram::@NamedTuple{bins::Vector{Float64}, hist::Vector{Float64}}
    App() = new(false, false, time())
end

function update!(app)
    app.busy = true
    p = app.pipeline
    process!(p.demoisaic, app)
    process!(p.whitebalance, p.demoisaic.data, app)
    process!(p.depth, p.whitebalance.data, app)
    process!(p.fog, p.whitebalance.data, app)
    process!(p.depth_of_field, p.depth.data, app)
    process!(p.bokeh, p.fog.data, app)
    process!(p.tonecurve, p.bokeh.data, app)
    process!(p.render, p.tonecurve.data, app)
    app.busy = false
    p.render
end

function update_display!(app)
    app.busy && return
    render = update!(app)

    empty!(ax)
    if show_depth_button.active
        img = app.pipeline.depth.depth
    elseif show_dof_button.active
        img = app.pipeline.depth_of_field.mask
    else
        img = render.data.img
    end

    uv_transform = render.params.uv_transform
    ax.aspect = get_aspect_ratio(uv_transform, img)

    if uv_transform == :automatic
        image!(ax, img)
    else
        image!(ax, img; uv_transform)
    end
end

function get_aspect_ratio(uv_transform, img)
    if uv_transform ∈ (:automatic, :rot180, :flip_x, :flip_y, :flip_xy)
        return size(img,1)/size(img,2)
    else
        size(img,2)/size(img,1)
    end
end

function update_histogram!(app)
    data = mean(app.pipeline.demoisaic.data.img, dims=(3)) |> vec
    @. data = log10(max(data, 1e-16))
    mu = mean(data)
    @. data -= mu

    bins = -2:0.1:2 |> collect
    hist = fit(Histogram, data, bins)
    hist = hist.weights ./ maximum(hist.weights)
    app.histogram = (;bins, hist)
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

    global const app = App()
    app.pipeline = get_pipeline(joinpath("data", "0V2A0073.CR3"))
    update_histogram!(app)
end

# depth

depth_expander = GtkExpander("Depth estimation")
depth_expander.expanded = true
#Gtk4.size_request(depth_expander, toolbox_width, 200)

depth_vbox = GtkBox(:v)
depth_expander[] = depth_vbox
push!(toolbox, depth_expander)

refine_depth_button = GtkCheckButton("Refine depth")

signal_connect(refine_depth_button, "toggled") do widget
    app.pipeline.depth.params.refine = refine_depth_button.active
    app.need_update = true
end
push!(depth_vbox, refine_depth_button)

choices = [:linear, :extrema]
depth_method_dd = GtkDropDown(choices)
# keep in mind that the "selected" property uses 0 based indexing
depth_method_dd.selected = 0

signal_connect(depth_method_dd, "notify::selected") do widget, others...
    idx = widget.selected
    str = Gtk4.selected_string(widget)
    app.pipeline.depth.params.method = Symbol(str)
    app.need_update = true
end

push!(depth_vbox, depth_method_dd)

## fog

fog_expander = GtkExpander("Fog")
fog_expander.expanded = true

fog_vbox = GtkBox(:v)
fog_expander[] = fog_vbox
push!(toolbox, fog_expander)

fog_radius_scale, fog_radius_box_gesture = get_scale(fog_vbox, 0.0, 1, 0, "Strength")
signal_connect(fog_radius_scale, "value-changed") do fog_radius_scale
    app.pipeline.fog.params.strength = Gtk4.value(fog_radius_scale)
end
signal_connect(fog_radius_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

fog_power_scale, fog_power_box_gesture = get_scale(fog_vbox, 0.5, 4, 1, "Power")
signal_connect(fog_power_scale, "value-changed") do fog_power_scale
    app.pipeline.fog.params.power = Gtk4.value(fog_power_scale)
end
signal_connect(fog_power_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# DoF

dof_expander = GtkExpander("dof estimation")
dof_expander.expanded = true

dof_vbox = GtkBox(:v)
dof_expander[] = dof_vbox
push!(toolbox, dof_expander)

distance_scale, distance_box_gesture   = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, 0.01, 1.2, 0.5, "Distance"; mapping_func = x -> 1 - x)
width_scale, width_box_gesture         = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, 0.05, 3, 2, "Width")
pinch_scale, pinch_box_gesture         = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, 0.5, 2, 1, "Pinch")
contrast_scale, contrast_box_gesture   = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, 0.1, 5.9, 3, "Contrast")
threshold_scale, threshold_box_gesture = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, 0.01, 0.99, 0.5, "Threshold")
tilt_x_scale, tilt_x_box_gesture       = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, -1, 1, 0, "Tilt x"; field = :tilt_x)
tilt_y_scale, tilt_y_box_gesture       = setup_scale_and_gesture(app, app.pipeline.depth_of_field.params, dof_vbox, -1, 1, 0, "Tilt y"; field = :tilt_y)

# bokeh

bokeh_expander = GtkExpander("Bokeh")
bokeh_expander.expanded = true

bokeh_vbox = GtkBox(:v)
bokeh_expander[] = bokeh_vbox
push!(toolbox, bokeh_expander)

bokeh_radius_scale, bokeh_radius_box_gesture = get_scale(bokeh_vbox, 1, 40, 15, "Radius")

signal_connect(bokeh_radius_scale, "value-changed") do bokeh_radius_scale
    app.pipeline.bokeh.params.radius = Gtk4.value(bokeh_radius_scale)
end
signal_connect(bokeh_radius_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

# Tone Curve
tonecurve_expander = GtkExpander("Tone curve")
tonecurve_expander.expanded = true
Gtk4.size_request(tonecurve_expander, toolbox_width, 200)

tonecurve_vbox = GtkBox(:v)
tonecurve_expander[] = tonecurve_vbox
push!(toolbox, tonecurve_expander)

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
exposure_scale, exposure_box_gesture = get_scale(tonecurve_vbox, -2, 2, 0, "Exposure")

signal_connect(exposure_scale, "value-changed") do exposure_scale
    app.pipeline.tonecurve.params.exposure = -Gtk4.value(exposure_scale)
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

# depth scale
dof_slope_scale, dof_slope_box_gesture = get_scale(tonecurve_vbox, -1.5, 1.5, 0, "DoF slope")

signal_connect(dof_slope_scale, "value-changed") do dof_slope_scale
    app.pipeline.tonecurve.params.dof_slope = Gtk4.value(dof_slope_scale)
    c.draw(c)
    reveal(c)
end
signal_connect(dof_slope_box_gesture, "released") do controller, n_press, x, y
    app.need_update = true
end

c = GtkCanvas()
Gtk4.size_request(c,(toolbox_width,200))
push!(tonecurve_vbox, c)

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

        bins = app.histogram.bins
        hist = app.histogram.hist

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

show_depth_button = GtkCheckButton("Show Depth")
signal_connect(show_depth_button, "toggled") do widget
    
    if !show_depth_button.active 
        app.pipeline.tonecurve.force_update = true
    end
    update_display!(app)
end
push!(render_vbox, show_depth_button)

show_dof_button = GtkCheckButton("Show DoF")
signal_connect(show_dof_button, "toggled") do widget

    # update bokeh when deactivate
    if !show_dof_button.active && app.pipeline.depth_of_field.data.updated
        app.pipeline.bokeh.force_update = true
    end
    update_display!(app)
end
push!(render_vbox, show_dof_button)

#

open_button = GtkButton(:icon_name, "document-open")
render_button = GtkButton("Render")
box = GtkBox(:h)
push!(box, open_button, render_button)
g[1,2] = box

id = signal_connect(open_button, "clicked") do widget
    open_dialog("Pick a file to open", win) do filename
        
        # initialize a new pipeline
        #app.pipeline = get_pipeline(filename)
        reset!(app.pipeline, filename)
        update_display!(app)
        c.draw(c)
        reveal(c)
    end
end

id = signal_connect(render_button, "clicked") do widget
    update_display!(app)
end

## start and show

update_display!(app)

Gtk4.GLib.g_timeout_add(25) do  # create a function that will be called every x milliseconds
    if app.need_update && !app.busy && (time() - app.last_update_time) > 0.1
        update_display!(app)
        app.need_update = false
        app.last_update_time = time()
    end
    true
end 

update_histogram!(app)
show(win)

##
