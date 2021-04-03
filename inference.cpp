// inference.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<torch/script.h>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/parallel/data_parallel.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstddef>
#include "CamVid_Load.hpp"
#include "args.hxx"
#include "option.hpp"
#include "fcn.hpp"

using namespace torch;
using namespace cv;

torch::DeviceType device_type;

struct FloatReader
{
	void operator()(const std::string &name, const std::string &value, std::tuple<double, double> &destination)
	{
		size_t commapos = 0;
		std::get<0>(destination) = std::stod(value, &commapos);
		std::get<1>(destination) = std::stod(std::string(value, commapos + 1));
	}
};

COptionMap<int> COptInt;
COptionMap<bool> COptBool;
COptionMap<std::string> COptString;
COptionMap<double> COptDouble;
COptionMap<std::tuple<double, double>> COptTuple;

//rgb
int label_colors[32][3] = { {64, 128, 64},{192, 0, 128},{0, 128, 192},{0, 128, 64},{128, 0, 0},{64, 0, 128},{64, 0, 192},{192, 128, 64},{192, 192, 128},
							{64, 64, 128},{128, 0, 192},{192, 0, 64},{128, 128, 64},{192, 0, 192},{128, 64, 64},{64, 192, 128},{64, 64, 0},{128, 64, 128},
							{128, 128, 192},{0, 0, 192},{192, 128, 128},{128, 128, 128},{64, 128, 192},{0, 0, 64},{0, 64, 64},{192, 64, 128},{128, 128, 0},
							{192, 128, 192},{64, 0, 64},{192, 192, 0},{0, 0, 0}, {64, 192, 0} };


class Train
{
public:
	Train() {	}
	~Train() {	}

public:

	template <typename DataLoader>
	void infer(FCN8s& fcns, torch::Device device, DataLoader& data_loader, int n_class);
};

//语义分割的验证过程是比较特殊的
template <typename DataLoader>
void Train::infer(FCN8s& fcn8s, torch::Device device, DataLoader& val_loader, int n_class)
{
	fcn8s.to(device);
	fcn8s.train(false);
	fcn8s.eval();
	
	for (auto batch : *val_loader)
	{
		auto data = batch.data();
		if (!data->data.numel())
		{
			std::cout << "tensor is empty!" << std::endl;
			continue;
		}
		torch::Tensor input = data->data.unsqueeze(0);
		input = input.to(device);
		torch::Tensor target = data->target.to(device);

		auto tm_start = std::chrono::system_clock::now();
		auto fcns_output = fcn8s.forward(input);
		auto tm_end = std::chrono::system_clock::now();

		//accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
		int N = fcns_output.sizes()[0];
		int h = fcns_output.sizes()[2];
		int w = fcns_output.sizes()[3];

		//output结果为{1, 32, h, w} => {-1, num_class} 即拉成每个像素一行,这个像素点的32类概率
		//然后argmax(1)求每行最大值下标,即是求出当前像素点属于哪一类,注意这里将像素点的值变成了分类值
		//最后重新调整成{h,w}类型
		torch::Tensor pred = fcns_output.permute({ 0, 2, 3, 1 }).reshape({ -1, num_class }).argmax(1).reshape({ N, h, w });
		pred = pred.to(torch::kCPU);
		pred = pred.squeeze(0);
		pred = pred.toType(torch::kInt32);
		int * n_pred = (int*)pred.data_ptr();
		Mat image(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
		//set pixels to create colour pattern
		for (int y = 0; y < image.rows; y++) //go through all rows (or scanlines)
			for (int x = 0; x < image.cols; x++) {
				image.at<Vec3b>(y, x)[0] = label_colors[*n_pred][2]; //set blue component
				image.at<Vec3b>(y, x)[1] = label_colors[*n_pred][1];//set green component
				image.at<Vec3b>(y, x)[2] = label_colors[*n_pred ++][0]; //set red component
			}
		//construct a window for image display
		namedWindow("Display window", CV_WINDOW_AUTOSIZE);
		//visualise the loaded image in the window
		imshow("Display window", image);
		//wait for a key press until returning from the program
		waitKey(0);
		//free memory occupied by image
		image.release();
	}
}

int main()
{
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	args::ArgumentParser parser("This is a Semantic segmentation inference using fcn.", "This goes after the options.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> val_csv_dir(parser, "val_csv_dir", "dataset directory", { "val_csv_dir" }, "..\\CamVid\\val.csv");
	args::ValueFlag<int> n_class(parser, "n_class", "Pixel classification", { "n_class" }, 32);
	args::ValueFlag<int> batch_size(parser, "batch_size", "train patch size", { "batch_size" }, 1);

	std::string	val_csv_file = (std::string)args::get(val_csv_dir);
	int c = (int)args::get(n_class);
	int batch = (int)args::get(batch_size);

	auto val_dataset = CamVidDataset(val_csv_file, "val", c, true, 0.);
	auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(val_dataset), batch);

	Train T;
	FCN8s fcn8s(c);
	torch::serialize::InputArchive archive;
	archive.load_from("..\\retModel\\fcn8s.pt");

	fcn8s.load(archive);
	T.infer(fcn8s, device, val_loader, c);

	printf("Finish infer!\n");
}
