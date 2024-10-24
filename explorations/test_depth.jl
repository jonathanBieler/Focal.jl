
using ONNXRunTime

#d = ONNXRunTime.load_inference("data\\depth_anything_v2_vitb.onnx")
da = ONNXRunTime.load_inference("data\\depth_anything_v2_vitb_dynamic.onnx")

img = rand(Float32,1,3,518,518)

input = Dict("image" => img)

out = da(input)

##

