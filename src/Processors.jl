mutable struct ImageProcessor
    output::Array
    params::Dict

    function ImageProcessor(input::Array, params::Dict)
        new(input, similar(input), params)
    end
end

function process_image!(processor::ImageProcessor)
    # Your image processing code here
    # For example, let's assume we're just copying the input to the output
    processor.output .= processor.input
    return processor.output
end

function update!(processor::ImageProcessor, new_params::Dict)
    if processor.params != new_params
        processor.params = new_params
        process_image!(processor)
    end
    return processor.output
end