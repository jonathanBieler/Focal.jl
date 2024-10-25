
using ImageTransformations, ONNXRunTime, ImageCore

#d = ONNXRunTime.load_inference("data\\depth_anything_v2_vitb.onnx")
da = ONNXRunTime.load_inference("D:\\dev\\Depth-Anything-ONNX\\weights\\depth_anything_v2_vitb_17.onnx")

#python dynamo.py export --encoder vitb -b 1 -h 14*100 -w 980

##
img = rand(Float32,1,3,980,980)

input = Dict("image" => img)

@time out = da(input)


##
img = rand(Float32, 1, 3, 2*518, 2*518)

input = Dict("image" => img)

# Run inference - output will be (B,H,W)
@time out = da(input)

##

function estimate_depth(img, model; normalize = true)

    h,w = size(img,1), size(img,2)
    img = imresize(img, (518,518,3))
    img = PermutedDimsArray(img, (3, 1, 2))
    img = reshape(Float32.(img), (1,3,518,518))
    input = Dict("image" => img)

    depth = model(input)["depth"][1,:,:]
    if normalize
        depth = depth .-  minimum(depth)
        depth = depth ./  maximum(depth)
    end
    imresize(depth, (h,w))
end

img = rand(1024,600,3)
estimate_depth(img, da; normalize = true)

##

