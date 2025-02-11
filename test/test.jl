include("../src/process_image.jl")


##

file = "J:\\photo2\\2024\\Juin\\Manif contre le FN\\0V2A9704.CR3"

demoisaic, whitebalance, tonecurve, render, demoisaic_params, whitebalance_params, tonecurve_params, render_params = get_pipeline(file)

##

tonecurve_params.contrast = 0.01n
tonecurve_params.exposure = 0.3

@time process!(demoisaic, demoisaic_params)
@time process!(whitebalance, whitebalance_params, demoisaic.data)
@time process!(tonecurve, tonecurve_params, whitebalance.data)
@time process!(render, render_params, tonecurve.data)

image(render.data.img)

##

@code_warntype process!(tonecurve, tonecurve_params, whitebalance.data)

@profview process!(tonecurve, tonecurve_params, whitebalance.data)

## test curves

sigma(σ, x, lift, highlights) = σ + highlights*sigmoid(x - 2σ, 0, √σ/2) + lift*sigmoid(-(x + 2σ), 0, √σ/2)

sigmoid2(x, μ, σ, lift, highlights) = 1 / (1 + exp(-(x-μ)/sigma(σ, x-μ, lift, highlights))) 

xi = -2:0.01:2
σ = 0.2
μ = 0.1
lift = 0.2
highlights = 0.0

p = lines(xi, sigmoid2.(xi, μ, σ, lift, highlights))
lines!(xi, sigma.(σ, xi .-μ, lift, highlights))
lines!(xi, σ .+ 0*xi)
p

##
