#include<torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>        // for strcpy(), strcat()
#include <io.h>
#include <stdio.h>
#include <random>
#include <math.h>

const double EPS = 1e-6;

//对于libtorch transform是不能够进行input target同时进行图片增强处理
//如果场景是类似分类网络,推理输出的output对应的是one-hot结果,仅仅只是增强训练图片
//则可以使用libtorch图片变换函数以及继承
//如果对于图片重建这些数据准备要求相对复杂的则需要进行自己定制 get() 中的内容

using namespace std;
using namespace torch;
using namespace cv;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;

int num_class = 32;		//如果是CamVid则分为32类像素
int h = 720, w = 960;
//int batch_size = 1;
//这里对input image的尺寸进行了裁剪才能适合网络结构中间层的输入和输出
int train_h = int(h * 2 / 3);  //480
int train_w = int(w * 2 / 3);  //640

int val_h = int(h / 32) * 32;	//704
int val_w = w;					//960

typedef struct data
{
	std::string input_data;
	std::string target_data;
}DATA;

class CamVidDataset : public torch::data::Dataset<CamVidDataset>
{
public:
	explicit CamVidDataset(std::string csv_file, std::string phase, int n_class = num_class, bool crop = true, double flip_rate = 0.5);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

private:
	void read_csv(std::string csv_file);
	vector<DATA> data;
	std::tuple<float, float, float> mean;
	int n_class;
	double flip_rate;
	bool crop;
	int new_h;
	int new_w;
	std::string	phase;

private:
	int64_t rand_int(int64_t max) {
		return torch::randint(max, 1)[0].item<int64_t>();
	}

	double rand_double() {
		return torch::rand(1)[0].item<double>();
	}
};


CamVidDataset::CamVidDataset(std::string csv_file, std::string phase, int num_class, bool crop, double flip_rate) :
		mean(103.939/255, 116.779/255, 123.68/255), n_class(num_class), flip_rate(flip_rate), crop(crop), phase(phase)
{
	read_csv(csv_file);
	if (0 == phase.compare("train"))
	{
		new_h = train_h;
		new_w = train_w;
	}
	else if (0 == phase.compare("val"))
	{
		flip_rate = 0.;
		new_h = val_h;
		new_w = val_w;
	}
}

void CamVidDataset::read_csv(std::string csv_file)
{
	std::ifstream flabel_colors_file(csv_file);

	if (!flabel_colors_file.is_open())
		throw std::runtime_error("Could not open csv file");
	std::string input_data;
	std::string target_data;
	while (flabel_colors_file.good())
	{
		while (getline(flabel_colors_file, input_data, ','))
		{
			DATA csv_data;
			csv_data.input_data = input_data;
			getline(flabel_colors_file, target_data, '\n');
			//printf("target_data dir : %s\n", target_data.c_str());
			csv_data.target_data = target_data;
			data.push_back(csv_data);
		}
	}
}

//如果使用opencv imread 返回的颜色空间为: B G R order.
//注意 image、label、target之间的区别
//image 就是原始图像
//label 是标签shape为{height, weight}，(x,y)的值为32-classvalue
//target 为one-hot类型shape为{channel = 32, height, weight} (height, weight)值为0 or 1
torch::data::Example<> CamVidDataset::get(size_t index)
{
	std::string	img_name;
	std::string	label_name;

	DATA d = data.at(index);
	img_name = d.input_data;
	//printf("img_name dir : %s\n", img_name.c_str());
	cv::Mat img = cv::imread(img_name);
	label_name = d.target_data;
	//cout << "img_name : " << img_name << "  " << "label_name : " << label_name << endl;

	// Convert image to tensor
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous(); // Channels x Height x Width
	img_tensor = img_tensor.toType(torch::kFloat32);

	torch::Tensor target_tensor;
	FILE* fp = NULL;
	if(0 == phase.compare("train"))
		target_tensor = torch::zeros({ n_class, h, w }).toType(torch::kByte);
	else if(0 == phase.compare("val"))
		target_tensor = torch::zeros({ h, w }).toType(torch::kByte);
	if (0 == phase.compare("train"))
	{
		if (0 == label_name.compare("img"))
		{
			torch::Tensor a, b;
			return { a, b };
		}
		fopen_s(&fp, label_name.c_str(), "rb");
		if (fp == NULL)
		{
			printf("open label failed. %s\n", label_name.c_str());
			torch::Tensor a, b;
			return { a, b };
		}
		else
		{
			fread(target_tensor.data_ptr(), sizeof(unsigned char), n_class * h * w, fp);
		}
		fclose(fp);
		fp = NULL;
		target_tensor = target_tensor.toType(torch::kFloat32);
	}
	else if (0 == phase.compare("val"))
	{
		if (0 == label_name.compare("label"))
		{
			torch::Tensor a, b;
			return { a, b };
		}
		label_name.replace(/*label_name.begin() + */10, /*label_name.begin() + */11, "Labeled");
		//printf("label_name : %s\n", label_name.c_str());
		fopen_s(&fp, label_name.c_str(), "rb");
		if (fp == NULL)
		{
			printf("open label failed. %s\n", label_name.c_str());
			torch::Tensor a, b;
			return { a, b };
		}
		else
		{
			fread(target_tensor.data_ptr(), sizeof(torch::kByte), h * w * sizeof(torch::kByte), fp);
		}
		fclose(fp);
		fp = NULL;
		//target_tensor = target_tensor.toType(torch::kInt32);
		//cout << target_tensor << endl;
		target_tensor = target_tensor.toType(torch::kFloat32);
	}
	if (crop)
	{
		auto height_offset_length = h - new_h;
		auto width_offset_length = w - new_w;
		int nwidth_offset_length = 0;
		//printf("height_offset_length %d width_offset_length %d\n", height_offset_length, width_offset_length);
		auto height_offset = rand_int(height_offset_length);
		if (0 != width_offset_length)
			/*auto width_offset*/nwidth_offset_length = rand_int(width_offset_length);
		else if (0 == width_offset_length)
			nwidth_offset_length = nwidth_offset_length;
		auto width_offset = nwidth_offset_length;
		//printf("height_offset %d width_offset %d\n", height_offset, width_offset);
		img_tensor = img_tensor.index({ Ellipsis,Slice(height_offset, height_offset + new_h), Slice(width_offset, width_offset + new_w) }).contiguous();
		target_tensor = target_tensor.index({ Ellipsis,Slice(height_offset, height_offset + new_h), Slice(width_offset, width_offset + new_w) }).contiguous();
	}
	double dd = rand_double();
	if (dd - flip_rate < 0)
	{
		img_tensor = img_tensor.flip(-1);
		target_tensor = target_tensor.flip(-1);
	}
	
	float bMean = std::get<0>(mean);
	float gMean = std::get<1>(mean);
	float rMean = std::get<2>(mean);
	img_tensor = img_tensor.div_(255).sub((bMean, gMean, rMean))/*.div_((0.225, 0.224, 0.229))*/;	//这个一般是减去mean之后除以sdv

	//cout << img_tensor.sizes() << target_tensor.sizes() << endl;
	return { img_tensor.clone(), target_tensor.clone() };
}

torch::optional<size_t> CamVidDataset::size() const
{
	return data.size();
}