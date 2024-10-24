include("../src/process_image.jl")


##

file = "J:\\photo2\\2024\\Juin\\Manif contre le FN\\0V2A9704.CR3"

demoisaic, whitebalance, tonecurve, render, demoisaic_params, whitebalance_params, tonecurve_params, render_params = get_pipeline(file)

##

tonecurve_params.contrast = 0.01
tonecurve_params.exposure = 0.3

@time process!(demoisaic, demoisaic_params)
@time process!(whitebalance, whitebalance_params, demoisaic.data)
@time process!(tonecurve, tonecurve_params, whitebalance.data)
@time process!(render, render_params, tonecurve.data)

image(render.data.img)

##

@code_warntype process!(tonecurve, tonecurve_params, whitebalance.data)

@profview process!(tonecurve, tonecurve_params, whitebalance.data)

##

