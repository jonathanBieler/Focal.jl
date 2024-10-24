using Gtk4, Graphics
c = GtkCanvas()
win = GtkWindow(c, "Canvas")

@guarded draw(c) do widget
    ctx = getgc(c)
    h = Gtk4.height(c)
    w = Gtk4.width(c)
    # Paint red rectangle
    rectangle(ctx, 0, 0, w, h/2)
    set_source_rgb(ctx, 1, 0, 0)
    fill(ctx)
    # Paint blue rectangle
    rectangle(ctx, 0, 3h/4, w, h/4)
    set_source_rgb(ctx, 0, 0, 1)
    fill(ctx)
end

g = GtkGestureClick(c, 0)

signal_connect(g, "released") do controller, n_press, x, y
    @info "released"
    w=widget(controller)
    ctx = getgc(w)
    set_source_rgb(ctx, 0, 1, 0)
    Gtk4.arc(ctx, x, y, 5, 0, 2pi)
    stroke(ctx)
    reveal(w)
end

##

using Gtk4
c = GtkScale(:h,0,1, 0.01)
box = GtkBox(:v)
push!(box,c)

win = GtkWindow(box, "Scale")

g = GtkGestureClick(box, 0)
Gtk4.G_.set_propagation_phase(g, Gtk4.PropagationPhase_CAPTURE)

signal_connect(g, "released") do controller, n_press, x, y
    @info "released"
end
signal_connect(g, "pressed") do controller, n_press, x, y
    @info "pressed"
end

##

using Gtk4

c = GtkScale(:h, 0, 1, 0.01)
win = GtkWindow(c, "Scale")

g = GtkGestureClick(c, 0)

signal_connect(g, "released") do controller, n_press, x, y
    @info "released"
end
signal_connect(g, "pressed") do controller, n_press, x, y
    @info "pressed"
end

##