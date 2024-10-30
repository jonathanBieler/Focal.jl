
function estimate_depth(img, model; normalize = true)

    # must be multiple of 14, took ratio close to 3:2
    dh, dw = 1190,910
    
    h,w = size(img,1), size(img,2)
    img = imresize(img, (dh,dw,3))
    img = PermutedDimsArray(img, (3, 1, 2))
    img = reshape(Float32.(img), (1,3,dh,dw))
    input = Dict("image" => img)

    depth = model(input)["depth"][1,:,:]
    if normalize
        depth = depth .-  minimum(depth)
        depth = depth ./  maximum(depth)
    end
    depth
end


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
        part1 = img[idx1,:,:]
        part2 = img[idx2,:,:]
        part3 = img[idx3,:,:]
    else
        part1 = img[:,idx1,:]
        part2 = img[:,idx2,:]
        part3 = img[:,idx3,:]
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

##

function estimate_depth(img, file::String, method::Symbol, do_refine::Bool, margin)


    outfile = splitext(file)[1] * "_depth.png"
    if isfile(outfile)
        #@info "Loading depth mask: $outfile"
        #depth = FileIO.load(outfile)
        #return Float32.(depth)
    end

    @info "Loading depth_anything_v2"
    @time depth_model = ONNXRunTime.load_inference("D:\\dev\\Depth-Anything-ONNX\\weights\\depth_anything_v2_vitb_17.onnx", execution_provider=:cuda)

    # depth on whole image
    depth = estimate_depth(img, depth_model)
    depth = imresize(depth, size(img)[1:2])

    if do_refine
        splits = get_splits(img; margin = margin)
        depths = [estimate_depth(s.part, depth_model; normalize = true) .|> Float64 for s in splits]

        splits = [ 
            ImageSplit(imresize(depths[i], size(splits[i].part)[1:2]), splits[i].idx, splits[i].coefs)
            for i in eachindex(splits)
        ]

        dim = argmax(size(img))
        if dim == 1
            vals = [extrema(depth[s.idx,:]) for s in splits]
        else
            vals = [extrema(depth[:,s.idx]) for s in splits]
        end
        reg = [get_linear_regression(depth,s,dim) for s in splits]

        for i in eachindex(splits)
            if method == :extrema
                splits[i].part .= @. clamp( splits[i].part * vals[i][2] + vals[i][1], 0, 1)
            elseif method == :linear
                splits[i].part .= @. clamp( (splits[i].part - reg[i].β) / reg[i].α, 0, 1)
            else
                error("Unknown method $method")
            end
        end

        depth = blend_splits(depth, splits; margin = margin)
    end

    outfile = splitext(file)[1] * "_depth.png"
    #@info "Saved depth mask: $outfile"
    #FileIO.save(outfile, depth)

    # make sure memory is free'd
    release(depth_model)
    depth
end

##

incircle(i,j,radius) = begin
    i^2 + j^2 < radius^2
end

function apply_blur!(img, out, depth, mask)

    N = zeros(size(img,1), size(img,2))
    fill!(out, 0.0)

    @inbounds for x in axes(img,1), y in axes(img,2)

        d = mask[x,y]
        radius = max(round(Int, d*25), 1)

        for i in -radius:radius, j in -radius:radius

            xi, yj = x + i, y + j

            #(i != 0 || j !=0 ) &&
            !incircle(i, j, radius) && continue

            xi < 1 && continue
            yj < 1 && continue
            xi > size(img,1) && continue
            yj > size(img,2) && continue

            # low value = far
            # point far away shouldn't blur closer points
            # Far away = lower value
            depth_source = depth[x, y]
            depth_destination = depth[xi, yj]
            #if  (depth_source - depth_destination)/(depth_source .+ 1e-16) < 1.1
            if  (depth_source > depth_destination)
                for c in axes(img,3)
                    out[xi, yj, c] += img[x, y, c]
                end
                N[xi, yj] += 1
            end
        end
        
    end
    @tturbo for x in axes(img,1), y in axes(img,2), c in axes(img,3)
        out[x,y,c] = out[x,y,c] / N[x,y]
    end
    
    out
end

## 


function apply_blur_kernel!(input, output, depth, mask, N, radius, h, w)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y 

    d = mask[i,j]
    scaled_radius = max(ceil(Int, d*radius), 1)

    for oi in -scaled_radius:scaled_radius
        for oj in -scaled_radius:scaled_radius
            
            !incircle(oi,oj,scaled_radius) && continue

            ii = i + oi
            jj = j + oj

            ii < 1 && continue
            jj < 1 && continue
            ii > h && continue
            jj > w && continue

            # low value = far
            # further point shouldn't blur closer points
            depth_source = depth[i, j]
            depth_destination = depth[ii, jj]

            if  (depth_source - depth_destination)/(depth_source .+ 1e-16) < 1.1
                for c in 1:3
                    output[i,j,c] += input[ii,jj,c]
                end
                N[i,j] += 1
            end
        end
    end

    return
end

function normalize_bokeh!(output, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    for c in 1:3
        output[i,j,c] = output[i,j,c] / N[i,j]
    end
    return
end

#= 
img = app.pipeline.bokeh.data.img

depth = app.pipeline.depth.depth
mask = app.pipeline.depth_of_field.mask
input_img = @view app.pipeline.whitebalance.data.img[:,:,1:3]
 =#
function apply_blur_gpu!(input_img, img, depth, mask, radius)

    h,w,_ = size(input_img)

    input = CuArray{Float64}(input_img)
    output = CuArray{Float64}(zeros(h,w,3))
    depth_cu = depth |> cu
    mask_cu = mask |> cu
    N = CuArray{Int64}(zeros(h,w))

    tile = 16 
    blocks1 = size(input,1) ÷ tile
    blocks2 = size(input,2) ÷ tile 
    
    #for c in 1:3
        @cuda threads=(tile, tile) blocks=(blocks1, blocks2) apply_blur_kernel!(input, output, depth_cu, mask_cu, N, radius, h, w)
        @cuda threads=(tile, tile) blocks=(blocks1, blocks2) normalize_bokeh!(output, N)
    #end
    output |> Array
end



##


#= img = app.pipeline.bokeh.data.img

depth = app.pipeline.depth.depth
mask = app.pipeline.depth_of_field.mask
input_img = @view app.pipeline.whitebalance.data.img[:,:,1:3]

@time out, N = apply_blur!(input_img, img, depth, mask)
 =#


## split image into 3



##