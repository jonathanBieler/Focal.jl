using Gtk4, Graphics
c = GtkCanvas()
win = GtkWindow(c, "Canvas")
@guarded draw(c) do widget
    @info "redrawing"
    ctx = getgc(c)
    h = height(c)
    w = width(c)
    # Paint red rectangle
    rectangle(ctx, 0, 0, w, h/2)
    set_source_rgb(ctx, 1, 0, 0)
    fill(ctx)
    # Paint blue rectangle
    rectangle(ctx, 0, 3h/4, w, h/4)
    set_source_rgb(ctx, 0, 0, 1)
    fill(ctx)
end

##

using Gtk4, CairoMakie

config = CairoMakie.ScreenConfig(1.0, 1.0, :good, true, false, nothing)
CairoMakie.activate!()

t = 0.0:0.1:10.0

canvas = GtkCanvas(400, 200; vexpand=true, hexpand=true)
b = push!(GtkBox(:v),canvas)
w = GtkWindow(b,"CairoMakie example")
s = GtkScale(:h,1,10,0.01, draw_value = true)
push!(b,s)

mutable struct MyApp
    freq::Float64
end

global const myapp = MyApp(1)

@guarded draw(canvas) do widget
    f, ax, p = lines(t, sin.(myapp.freq*t))
    CairoMakie.autolimits!(ax)
    screen = CairoMakie.Screen(f.scene, config, Gtk4.cairo_surface(canvas))
    CairoMakie.resize!(f.scene, Gtk4.width(widget), Gtk4.height(widget))
    CairoMakie.cairo_draw(screen, f.scene)
end

signal_connect(s, "value-changed") do widget
    myapp.freq = Gtk4.value(s)
    canvas.draw(canvas)
    reveal(canvas)
end

##