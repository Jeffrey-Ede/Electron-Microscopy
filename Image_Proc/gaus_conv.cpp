#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include "opencv2/highgui/highgui.hpp" //Loading images
#include <opencv2/imgproc/imgproc.hpp> //Convert RGB to greyscale
#include "opencv2/core/ocl.hpp"
 
static const char* inputImagePath = "D:/data/baracktocat.jpg";

static float gaussianBlurFilter[25] = {
	1.0f/273.0f, 4.0f/273.0f, 7.0f/273.0f, 4.0f/273.0f, 1.0f/273.0f, 
	4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f, 
	7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
	4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f, 
	1.0f/273.0f, 4.0f/273.0f, 7.0f/273.0f, 4.0f/273.0f, 1.0f/273.0f
};

static const int gaussianBlurFilterWidth = 5;

using namespace cv;

int main()
{
	float *hInputImage;
	float *hOutputImage;

	//Set filter
	int filterWidth = gaussianBlurFilterWidth;
	float *filter = gaussianBlurFilter;

	//Read in the image
	Mat imageGrey; Mat image32F;
	Mat image = imread(inputImagePath, CV_LOAD_IMAGE_COLOR);
	cvtColor(image, imageGrey, CV_BGR2GRAY);
	imageGrey.convertTo(image32F, CV_32F);

	//float dummy_query_data[25] = { 1.0, 1.0, 1.0, 1.0, 2.0,
	//	1.0, 1.0, 1.0, 1.0, 2.0,
	//	1.0, 1.0, 1.0, 1.0, 5.0,
	//	1.0, 1.0, 1.0, 1.0, 2.0,
	//	1.0, 1.0, 1.0, 1.0, 2.0,
	//};
	//cv::Mat image32F = cv::Mat(5, 5, CV_32F, dummy_query_data);

	int imageRows = image32F.rows;
	int imageCols = image32F.cols;

	//Reference to input image

	hInputImage = (float*)(image32F.data);

	/*namedWindow("average", CV_WINDOW_AUTOSIZE);
	imshow("average", imageGrey);

	waitKey(0);*/

	//std::cout << hInputImage[image32F.rows*image32F.c] << " ";

	//std::cout << "dsfdsfsdds" << std::endl;

	//Creat space for the output image
	hOutputImage = new float [imageRows*imageCols];

	try{
		//Get platforms
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Get devices
		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		//Create context
		cl::Context context(devices);

		//Create command queue
		cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

		//Create images
		cl::ImageFormat imageFormat = cl::ImageFormat(CL_R, CL_FLOAT);
		cl::Image2D inputImage = cl::Image2D(context, CL_MEM_READ_ONLY, imageFormat, imageCols, imageRows);
		cl::Image2D outputImage = cl::Image2D(context, CL_MEM_WRITE_ONLY, imageFormat, imageCols, imageRows);

		//Create a buffer for the filter
		cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, filterWidth*filterWidth*sizeof(float));

		//Copy input data to the input image
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		cl::size_t<3> region;
		region[0] = imageCols;
		region[1] = imageRows;
		region[2] = 1;
		queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0, hInputImage);

		//Copy the filter to the buffer
		queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, filterWidth*filterWidth*sizeof(float), filter);

		//Create the sampler
		cl::Sampler sampler = cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

		//Read the program source
		std::ifstream sourceFile("D:/C_C++_C#/gauss_conv.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		//Make the program from the source code
		cl::Program program = cl::Program(context, source);

		//Build the program for the devices
		try
		{
			cl::Error err = program.build( devices, "" );
		}
		//Catch any build errors and pring buildlog for debugging
		catch( cl::Error err )
		{
			cl::STRING_CLASS buildlog;
			program.getBuildInfo( devices[0], (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &buildlog );
			std::cout << buildlog << std::endl;
		}

		//Create the kernel
		cl::Kernel kernel(program, "gauss_conv_kernel");

		//Set the kernel arguments
		kernel.setArg(0, inputImage);
		kernel.setArg(1, outputImage);
		kernel.setArg(2, filterBuffer);
		kernel.setArg(3, filterWidth);
		kernel.setArg(4, sampler);

		//Execute the kernel
		cl::NDRange global(imageCols, imageRows);
		cl::NDRange local(8,8);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		
		//Copy output data back to the host
		queue.enqueueReadImage(outputImage, CL_TRUE, origin, region, 0, 0, hOutputImage);

		////Convert output to Mat to display
		/*UMat blurred_img;
		ocl::convertFromImage(hOutputImage, blurred_img);
*/
		//std::cout << blurred_img;
		/*for (int i = 0; i < 25; i++) {
			std::cout << hOutputImage[i] << std::endl;
		}*/

		cv::Mat blurred_img = cv::Mat(imageCols, imageRows, CV_32F, hOutputImage);
		//blurred_img32F.convertTo(blurred_img, CV_16U);

		cv::Mat dst;
		cv::normalize(blurred_img, dst, 0, 1, cv::NORM_MINMAX);
		cv::imshow("test", dst);

		//std::cout << blurred_img << std::endl;

		/*namedWindow("average", CV_WINDOW_AUTOSIZE);
		imshow("average", blurred_img);
*/
		waitKey(0);

		//Save the output BMP image
		//writeBMPFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols, inputImagePath);
	}
	catch(cl::Error error)
	{
		std::cout << error.what() <<  "(" << error.err() << ")" << std::endl;
	}

	std::free(hInputImage);
	delete hOutputImage;

	return 0;
}
