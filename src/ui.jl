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

function setup_scale_and_gesture(app, params, vbox, min_val, max_val, default_val, name; mapping_func = identity, field = Symbol(lowercase(name)))
    scale, box_gesture = get_scale(vbox, min_val, max_val, default_val, name)

    signal_connect(scale, "value-changed") do scale_widget
        setfield!(params, field, mapping_func(Gtk4.value(scale_widget)))
    end

    signal_connect(box_gesture, "released") do controller, n_press, x, y
        app.need_update = true
    end

    return scale, box_gesture
end