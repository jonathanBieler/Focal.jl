
using ImageTransformations, ONNXRunTime, ImageCore

##
using ONNXRunTime
using CUDA
import cuDNN

#CUDA.set_runtime_version!(v"11.8")

##


#da = ONNXRunTime.load_inference("data\\depth_anything_v2_vitb_dynamic.onnx", execution_provider=:cuda)
@time depth_model = ONNXRunTime.load_inference("D:\\dev\\Depth-Anything-ONNX\\weights\\depth_anything_v2_vitb_17.onnx", execution_provider=:cuda)

# depth_anything\Scripts\activate
# python dynamo.py export --encoder vitb -b 1 -h 1400 -w 924

##
img = rand(Float32, 1, 3, 1400, 924)
img = rand(Float32, 1, 3, 1200, 800)

input = Dict("image" => img)

@time out = depth_model(input)

##

img = rand(Float32, 1, 3, 2*518, 2*518)

input = Dict("image" => img)

# Run inference - output will be (B,H,W)
@time out = da(input)

##

function estimate_depth(img, depth_model; normalize = true)

    h,w = size(img,1), size(img,2)
    img = imresize(img, (518,518,3))
    img = PermutedDimsArray(img, (3, 1, 2))
    img = reshape(Float32.(img), (1,3,518,518))
    input = Dict("image" => img)

    depth = depth_model(input)["depth"][1,:,:]
    if normalize
        depth = depth .-  minimum(depth)
        depth = depth ./  maximum(depth)
    end
    imresize(depth, (h,w))
end

img = rand(1024,600,3)
estimate_depth(img, da; normalize = true)

## patching code

#ImageFiltering
# Interpolations
using ONNXRunTime, FileIO
using Statistics
using CUDA
import cuDNN

using Makie, GLMakie
depth_model = ONNXRunTime.load_inference("D:\\dev\\Depth-Anything-ONNX\\weights\\depth_anything_v2_vitb_17.onnx", execution_provider=:cuda)

##

function estimate_depth(img, depth_model; normalize = true)

    img = imresize(img, (1190,910))
    img = Float32.(channelview(img)[1:3,:,:])
    img = reshape(img, (1,3,1190,910))
    input = Dict("image" => img)

    depth = depth_model(input)["depth"][1,:,:]
    if normalize
        depth = depth .-  minimum(depth)
        depth = depth ./  maximum(depth)
    end
    depth
end

## load image

file = "0V2A8984.jpg"
#file = "0V2A8281.JPG"
file = "i:\\photos\\win7\\london\\jpg\\IMG_0996.JPG"
#file = "i:\\photos\\win7\\2013_09_25\\IMG_0244.JPG"
file = "i:\\photos\\web\\converted\\DSC00148-1.jpg"
img =  FileIO.load(file)

img = convert.(RGB{Float64}, img)
img = imresize(img, ratio=1)
#img = reverse(img',dims=1)

## estimate depth

depth = estimate_depth(img, depth_model)
depth = imresize(depth, size(img))
#heatmap(reverse(depth, dims=1), size=(400,400), legend = false)

RGB.(depth)

FileIO.save("depth.png", depth)
FileIO.save("img.png", img)

##

struct ImageSplit{T}
    part::T
    idx::UnitRange{Int64}
    coefs::Vector{Float64}
end

part1_coefs(i, N, margin) = i <= N ? 1.0 : 1-(i-N)/margin 
part2_coefs(i, N, margin) = begin
    if i <= margin
        return (i-1)/margin
    elseif i <= N
        return 1.0
    else
        k = i - (N+1)
        return 1 -k/margin 
    end
end
part3_coefs(i, N, margin) = i <= margin ? (i-1)/margin : 1.0

function get_splits(img; margin = 20)

    dim = argmax(size(img))
    Ntot = size(img)[dim]
    N = div(Ntot,3)

    idx1 = 1:N+margin
    idx2 = N:2N+margin
    idx3 = (Ntot -N):Ntot

    if dim == 1
        part1 = img[idx1,:]
        part2 = img[idx2,:]
        part3 = img[idx3,:]
    else
        part1 = img[:,idx1]
        part2 = img[:,idx2]
        part3 = img[:,idx3]
    end
    
    c1 = part1_coefs.(1:length(idx1), N , margin)
    c2 = part2_coefs.(1:length(idx2), N , margin)
    c3 = part3_coefs.(1:length(idx3), N , margin)

    split1 = ImageSplit(part1, idx1, c1)
    split2 = ImageSplit(part2, idx2, c2)
    split3 = ImageSplit(part3, idx3, c3)

    split1, split2, split3
end

function blend_splits(img, splits; margin = 20)

    dim = argmax(size(img))
    Ntot = size(img)[dim]
    N = div(Ntot,3)
    out = similar(img)
    
    if dim == 1
        for s in splits
            for (i,k) in enumerate(s.idx)
                out[k,:] += s.coefs[i] * s.part[i,:]
            end
        end
    else
        for s in splits
            for (i,k) in enumerate(s.idx)
                out[:,k] += s.coefs[i] * s.part[:,i]
            end
        end
    end
    out
end

margin = 300
splits = get_splits(img; margin = margin)

#hcat(splits[1].part, splits[2].part, splits[3].part)
#vcat(splits[1].part, splits[2].part, splits[3].part)

f = lines(splits[1].idx, splits[1].coefs)
lines!(splits[2].idx, splits[2].coefs)
lines!(splits[3].idx, splits[3].coefs)
f
blended = blend_splits(img, splits; margin = margin)

## compute depth and blend

function get_linear_regression(depth, s, dim)
    if dim == 1
        x = depth[s.idx,:] |> vec
    else
        x = depth[:,s.idx] |> vec
    end
    y = s.part |> vec

    β, α = hcat(ones(length(x)), x) \ y 
    (;β, α)
end

splits = get_splits(img; margin = margin)
depths = [estimate_depth(s.part, depth_model; normalize = true) .|> Float64 for s in splits]

for i in eachindex(depths)
    #depths[i] = depths[i] .- imfilter(depths[i], Kernel.gaussian(20))
end

splits = [ 
    ImageSplit(imresize(depths[i], size(splits[i].part)), splits[i].idx, splits[i].coefs)
    for i in eachindex(splits)
]

dim = argmax(size(img))
if dim == 1
    #vals = [mean(depth[s.idx,:]) for s in splits]
    vals = [extrema(depth[s.idx,:]) for s in splits]
else
    #vals = [mean(depth[:, s.idx]) for s in splits]
    vals = [extrema(depth[:,s.idx]) for s in splits]
end
reg = [get_linear_regression(depth,s,dim) for s in splits]

##

for i in eachindex(splits)
    #depths[i] = @. depths[i] * vals[i][2] + vals[i][1]
    #depths[i] = @. depths[i] - mean(depths[i] )
    #splits[i].part .= @. clamp( splits[i].part * vals[i][2] + vals[i][1], 0, 1)
    splits[i].part .= @. clamp( (splits[i].part - reg[i].β) / reg[i].α, 0, 1)
end

blended = blend_splits(depth, splits; margin = margin)

RGB.(depth)
RGB.(blended)

FileIO.save("depth.png", depth)
FileIO.save("blended_lin.png", blended)
#FileIO.save("blended_ext.png", blended)

##

function get_linear_regression2(depth, s, dim)
    if dim == 1
        x = depth[s.idx,:] |> vec
    else
        x = depth[:,s.idx] |> vec
    end
    y = s.part |> vec

    β, α = hcat(ones(length(x)), x) \ y 
    β, α, x, y
end

s = splits[1]

β, α, x, y = get_linear_regression2(depth, s, dim)

yt = @. (y - β)/α

p = scatter(x, y , alpha=0.01)
ablines!(0, 1)
ablines!(β, α)
p

##
