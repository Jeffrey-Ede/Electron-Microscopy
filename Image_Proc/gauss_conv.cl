__kernel
void gauss_conv_kernel(
    __read_only image2d_t inputImage,
    __write_only image2d_t outputImage,
    __constant float* filter,
    int filterWidth,
    sampler_t sampler)
{
    //Get location of work item
    int column = get_global_id(0);
    int row = get_global_id(1);

    int halfWidth = (int)(filterWidth/2); //Needed when indexing memory later

    //Accesses to images return data as 4-element vectors, although only the x component will contain meaningful data here
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

    //Iterator for filter
    int filterIdx = 0;

    //Work items iterate their local area dictated by the filter size
    int2 coords; //Coordinates for accessing the image

    //Iterate through filter rows
    for(int i = -halfWidth; i <= halfWidth; i++){

        coords.y = row + i;

        //Iterate through filter columns
        for(int j = -halfWidth; j <= halfWidth; j++){
            
            coords.x = column + j;

            //Read pixel from image. Single chanel image stores the pixel in the x component of the returned vector
            float4 pixel;

            pixel = read_imagef(inputImage, sampler, coords);
            sum.x += pixel.x*filter[filterIdx++];
        }
    }

    //Copy data to the output image
    coords.x = column;
    coords.y = row;
    write_imagef(outputImage, coords, sum);
}