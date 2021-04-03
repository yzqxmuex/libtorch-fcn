// fcn.cpp : 此文件222包含 "main" 函数。程序执行将在此处开始并结束。
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

class Train
{
public:
	Train() {	}
	~Train() {	}

public:
	template <typename DataLoader>
	void train(int nEpoch, FCN8s& fcns, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, torch::nn::BCEWithLogitsLoss& criterion);

	template <typename DataLoader>
	void val(int nEpoch, FCN8s& fcns, torch::Device device, DataLoader& data_loader);
};

template <typename DataLoader>
void Train::train(int nEpoch, FCN8s& fcn8s, torch::Device device, DataLoader& train_loader, torch::optim::Optimizer& optimizer, torch::nn::BCEWithLogitsLoss& criterion)
{
	fcn8s.to(device);
	fcn8s.train(true);

	int iter = 0;
	long long accumulationCost = 0;
	for (auto batch : *train_loader)
	{	
		iter++;
		auto data = batch.data();
		if (!data->data.numel())
		{
			std::cout << "tensor is empty!" << std::endl;
			continue;
		}
		torch::Tensor input = data->data.unsqueeze(0);
		torch::Tensor target = data->target.unsqueeze(0);
		input = input.to(device);
		target = target.to(device);
	
		optimizer.zero_grad();
		auto tm_start = std::chrono::system_clock::now();
		auto fcns_output = fcn8s.forward(input);
		auto tm_end = std::chrono::system_clock::now();

		auto loss = criterion(fcns_output, target);
		loss.backward();
		optimizer.step();
		accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
		
		if (iter % 10 == 0)
		{
			printf("epoch {%d}, iter{%d}, loss: {%.3f} cost:{%lld msec}\n",
				nEpoch,
				iter,
				loss.item().toFloat(),
				accumulationCost);
			accumulationCost = 0;
		}
	}
	
}

std::vector<float> iou(torch::Tensor pred, torch::Tensor target)
{
	pred = pred.to(torch::kCPU);
	target = target.to(torch::kCPU);
	pred = pred.toType(torch::kInt32);
	target = target.toType(torch::kInt32);
	int h = target.sizes()[0];
	int w = target.sizes()[1];
	torch::Tensor pred_inds = torch::zeros({ h, w }).toType(torch::kInt32);
	torch::Tensor target_inds = torch::zeros({ h, w }).toType(torch::kInt32);

	//将tensor转换成int array
	pred = pred.squeeze(0);
	
	//将pred和target全部转换成int 数组,因为tensor的迭代计算非常缓慢
	int* pred_array = (int*)pred.data_ptr();
	int* target_array = (int*)target.data_ptr();
	int intersection = 0;	//并集
	int union_ = 0;			//交集
	
	std::vector<float> ious;	//单张图片的n_class的ious
	//auto tm_start = std::chrono::system_clock::now();
	for (int cls = 0; cls < num_class; cls++)
	{
		intersection = 0;
		//以下两行为重新调整数组的开头,否则会因为++而导致指针往后走
		pred_array = (int*)pred.data_ptr();
		target_array = (int*)target.data_ptr();
		//以下两行为清0,每次循环计算一个分类计数
		pred_inds.zero_();
		target_inds.zero_();
		//同样的转换pred_inds为int类型数组,每次循环需要重新设置指针起始
		int* pred_inds_array = (int*)pred_inds.data_ptr();
		int* target_inds_array = (int*)target_inds.data_ptr();
		//开始计算pred_array也就是推理结果的tensor中等于cls计数,设置为1
		//同样的target_array也就是标签,未经过one-hot的标签,在CamVidUtils中已经保存在Labeled文件夹中
		for (int j = 0; j < h; j ++)
		{
			for (int k = 0; k < w; k ++, pred_inds_array++, target_inds_array++)
			{
				if (*pred_array++ == cls)
					*pred_inds_array = 1;
				if (*target_array++ == cls)
					*target_inds_array = 1;
			}
		}
		//重新把指针起始设置回来
		pred_inds_array = (int*)pred_inds.data_ptr();
		target_inds_array = (int*)target_inds.data_ptr();
		//交集的计算,即标签中等于1的对应像素值和推理结果中的对应像素值是多少,对这个值进行累加,即为并集
		//也就是推理出来的图片在当前分类的比对中多少像素点的分类和标签当前类中的分类是一样的
		for (int k = 0; k < h * w; k++, target_inds_array++, pred_inds_array++)
		{
			if (*target_inds_array == 1)
			{
				intersection += *pred_inds_array;
			}
		}
		//printf("交集 intersection = %d\n", intersection);
		//求并集
		union_ = (pred_inds.sum().item<int>() + target_inds.sum().item<int>() - intersection);
		
		//如果并集为0,当前类并没有ground truth
		if (union_ == 0)
			ious.push_back(std::nanf("nan"));
		else
			ious.push_back(float(intersection) / max(union_, 1));	//求iou,将每个类的iou推入到ious中作为函数返回
	}
	//auto tm_end = std::chrono::system_clock::now();
	//printf("cost:{%lld msec}\n", std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	return ious;
}

float pixel_acc(torch::Tensor pred, torch::Tensor target)
{
	pred = pred.to(torch::kCPU);
	target = target.to(torch::kCPU);
	pred = pred.toType(torch::kInt32);
	target = target.toType(torch::kInt32);

	int correct = 0;
	int total = 0;

	pred = pred.squeeze(0);

	int h = pred.sizes()[0];
	int w = pred.sizes()[1];

	int* pred_array = (int*)pred.data_ptr();
	int* target_array = (int*)target.data_ptr();

	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i ++)
		{
			//printf("pred class : %d target class : %d\n", *pred_array, *target_array);
			if (*pred_array++ == *target_array++)
				correct++;
			total++;
		}
	}
	//printf("correct : %d total : %d\n", correct, total);
	return (float)correct / (float)total;
}

//语义分割的验证过程是比较特殊的
template <typename DataLoader>
void Train::val(int nEpoch, FCN8s& fcn8s, torch::Device device, DataLoader& val_loader)
{
	fcn8s.to(device);
	fcn8s.train(false);
	fcn8s.eval();

	std::vector<float> ious;
	long long accumulationCost = 0;

	float totalMeanIoU = .0;
	float totalPixel_accs = .0;
	int N = 0;

	for (auto batch : *val_loader)
	{
		N++;
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

		accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
		int N = fcns_output.sizes()[0];
		int h = fcns_output.sizes()[2];
		int w = fcns_output.sizes()[3];

		//output结果为{1, 32, h, w} => {-1, num_class} 即拉成每个像素一行,这个像素点的32类概率
		//然后argmax(1)求每行最大值下标,即是求出当前像素点属于哪一类,注意这里将像素点的值变成了分类值
		//最后重新调整成{h,w}类型
		torch::Tensor pred = fcns_output.permute({ 0, 2, 3, 1 }).reshape({ -1, num_class }).argmax(1).reshape({ N, h, w });
		//iou函数将返回每张图片(如果是有batch的)在n_class中的分类iou
		//例如,[第一张图片[0类的iou, 1类的iou, .... n_class-1类iou],第二张图片[0类iou, 1类iou, ....n_class-1类iou]....batch张]
		ious = iou(pred, target);
		//因为在论文中的建议以及gpu内存的限制,在进行语义分割的时候经常使用batch = 1,因此在这里就直接累加ious中的值(vector<float>)进行一次mean即可
		//至此,求完像素类型分类的iou
		
		//注意像素点的精确accs和IoU是不一样的衡量尺度,accs很大的情况下,IoU并不一定大,IoU衡量的是图像的重合程度,accs是像素点的相等程度
		//换句话讲,图片中如果存在车辆和道路,车辆像素点都一样而道路仍旧错误的情况下,就会造成accs很大,然而整张图片重合程度仍旧很低
		float meanIoU = .0;
		float pixel_accs = .0;
		std::vector<float>::iterator it;
		for (it = ious.begin(); it != ious.end(); ++it)
		{
			if (std::isnan(*it))
				continue;
			else
				meanIoU += (*it);
		}
		meanIoU /= num_class;
		totalMeanIoU += meanIoU;
		pixel_accs = pixel_acc(pred, target);
		totalPixel_accs += pixel_accs;
		//cout << "meanIoU: " << meanIoU << " pixel_accs: " << pixel_accs << endl;
	}
	totalMeanIoU /= N;
	totalPixel_accs /= N;
	printf("epoch{%d}, pix_acc: {%0.6f}, meanIoU: {%0.6f}\n", nEpoch, totalPixel_accs, totalMeanIoU);
	//printf("epoch {%d}, meanIoU:{%0.5f} cost:{%lld msec}\n", nEpoch, meanIoU, accumulationCost);
}

int main()
{
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	srand(time(NULL));
	args::ArgumentParser parser("This is a Semantic segmentation using fcn.", "This goes after the options.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });
	//# Hardware specifications
	args::ValueFlag<int> n_threads(parser, "n_threads", "number of threads for data loading", { "n_threads" }, 1);
	args::ValueFlag<bool> cpu(parser, "cpu", "use cpu only", { "cpu" }, 0);
	args::ValueFlag<int> n_GPUs(parser, "n_GPUs", "number of GPUs", { "n_GPUs" }, 1);

	//# Data specifications
	args::ValueFlag<std::string> root_dir(parser, "root_dir", "dataset directory", { "root_dir" }, "..\\CamVid");
	args::ValueFlag<std::string> model_dir(parser, "model_dir", "create dir for model", { "model_dir" }, "..\\save_models");
	args::ValueFlag<std::string> train_csv_dir(parser, "train_csv_dir", "dataset directory", { "train_csv_dir" }, "..\\CamVid\\train.csv");
	args::ValueFlag<std::string> val_csv_dir(parser, "val_csv_dir", "dataset directory", { "val_csv_dir" }, "..\\CamVid\\val.csv");

	//# Training specifications
	args::ValueFlag<int> n_class(parser, "n_class", "Pixel classification", { "n_class" }, 32);
	args::ValueFlag<int> batch_size(parser, "batch_size", "train patch size", { "batch_size" }, 1);
	args::ValueFlag<int> epochs(parser, "epochs", "number of epochs to train", { "epochs" }, 500);

	//# Optimization specifications
	args::ValueFlag<double> lr(parser, "lr", "learning rate", { "lr" }, 1e-4);
	args::ValueFlag<double> momentum(parser, "momentum", "RMSprop", { "momentum" }, 0);
	args::ValueFlag<double> weight_decay(parser, "weight_decay", "weight decay", { "weight_decay" }, 1e-5);
	args::ValueFlag<int> step_size(parser, "step_size", "lr scheduler step", { "step_size" }, 50);
	args::ValueFlag<double> gamma(parser, "gamma", "decay LR by a factor of gamma", { "gamma" }, 0.5);

	//get parameters
	std::string train_csv_file = (std::string)args::get(train_csv_dir);
	std::string	val_csv_file = (std::string)args::get(val_csv_dir);
	double learning_rate = (double)args::get(lr);
	double momentum_ = (double)args::get(momentum);
	double weight_decay_ = (double)args::get(weight_decay);
	int c = (int)args::get(n_class);
	int batch = (int)args::get(batch_size);

	//process
	std::vector<double> norm_mean = { 0.406, 0.456, 0.485 };
	std::vector<double> norm_std = { 0.225, 0.224, 0.229 };
	std::tuple<float, float, float> mean = std::tuple<float, float, float>(103.939 / 255, 116.779 / 255, 123.68 / 255); //这个值求出来和{0.406，0.456, 0.485}一模一样,主要小心BGR和RGB顺序
	auto train_dataset = CamVidDataset(train_csv_file, "train", 32, true, 0.5);
		/*.map(torch::data::transforms::Normalize<>(norm_mean, norm_std))*/
	
	// Train_Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batch);

	auto val_dataset = CamVidDataset(val_csv_file, "val", c, true, 0.);

	// Val_Data loader
	auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(val_dataset), batch);
	
	Train T;
	FCN8s fcn8s(c);

	auto criterion = torch::nn::BCEWithLogitsLoss();
	auto optimizer = torch::optim::RMSprop(fcn8s.parameters(), torch::optim::RMSpropOptions(learning_rate).momentum(momentum_).weight_decay(weight_decay_));
	criterion->to(device);

	int Epochs = (int)args::get(epochs);
	for (int e = 0; e < Epochs; e++)
	{
		if (((e + 1) % 30) == 0)
		{
			learning_rate *= gamma;
			static_cast<torch::optim::RMSpropOptions &>(optimizer.param_groups()[0].options()).lr(learning_rate);
		}
		for (auto param_group : optimizer.param_groups()) {

			printf("lr = %.9f ", static_cast<torch::optim::RMSpropOptions &>(param_group.options()).lr());
		}
		T.train(e, fcn8s, device, train_loader, optimizer, criterion);
		T.val(e, fcn8s, device, val_loader);
	}

	printf("Finish training!\n");
	torch::serialize::OutputArchive archive;
	fcn8s.save(archive);
	archive.save_to("..\\retModel\\fcn8s.pt");
	printf("Save the training result to ..\\fcn8s.pt.\n");
}
