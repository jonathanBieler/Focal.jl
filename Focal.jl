
cd("D:\\dev\\Focal.jl")
using Pkg; Pkg.activate(".")

##

using Gtk4, Gtk4Makie, LibRaw, Memoize

include("src/process_image.jl")

screen = Gtk4Makie.GTKScreen(size=(800, 800),title="10 random numbers")
display(screen, image(rand(1000,1000)))
ax=current_axis()
win = window(screen)

g = grid(screen)

open_button = GtkButton(:icon_name, "document-open")
g[1,2] = open_button


id = signal_connect(open_button, "clicked") do widget
    open_dialog("Pick a file to open", win) do filename
        img = process_raw(filename, 0.5, 0.5)
        empty!(ax)
        image!(ax, img')
    end
end

show(win)