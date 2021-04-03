// CamVidUtils.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <string>
#include <fstream>
#include <vector>
#include <utility>			// std::pair
#include <stdexcept>		// std::runtime_error
#include <sstream>			// std::stringstream
#include <iostream>
#include"commonUtils.hpp"

//*******************************************************************************************************************************//
//CamVid:http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
//本工程参考来源是github上的
//CamVid在不同的工程下有各种不同的目录结构,在本工程下结构为:
//CamVid(701_StillsRaw_full(701张静态原始照片),LabeledApproved_full(701张对应的语义分割),label_colors.txt(语义标签的RGB一共32行))
//CamVidUtils将对原始图片进行预先处理
//1:区分训练集和验证集
//******************************************************************************************************************************//

//#############################
//# global variables #
//#############################
std::string	root_dir = "..\\CamVid\\";					
std::string	data_dir = root_dir + "701_StillsRaw_full\\";			//训练集图片
std::string	label_dir = root_dir + "LabeledApproved_full\\";		//训练集标签
std::string label_colors_file = root_dir + "label_colors.txt";	//标签对应的颜色RGB
std::string val_label_file = root_dir + "val.csv";				//验证集图片路径和对应的npy文件路径	
std::string train_label_file = root_dir + "train.csv";			//训练集图片路径和对应的npy文件路径

//create dir for one-hot label index
std::string label_idx_dir = root_dir + "Labeled_idx";

//create dir for label
std::string label_dir_no_one_hot = root_dir + "Labeled";

bool divide_train_val(float val_rate = 0.1, bool shuffle = true, int random_seed = 1)
{
	bool bret = false;
	int data_len = 0, val_len = 0;
	vector<int> data_idx;
	vector<string> train_idx, val_idx;
	
	//枚举出dataset
	fileNameList_t	data_list;

	listFiles(data_dir.c_str(), data_list);
	data_len = data_list.size();
	val_len = data_len * val_rate;
	printf("data_dir : %s data_len = %d val_len = %d\n", data_dir.c_str(), data_list.size(), val_len);

	//打乱图片序号或者顺序存放进数据idx
	srand(time(NULL));
	if(shuffle)
		randperm(data_len, data_idx);
	else
	{
		for (int i = 0; i < data_len; ++i)
		{
			data_idx.push_back(i);
		}
	}

	//拆分出训练集和验证集
	int train_len = (data_len - val_len);
	printf("train_len : %d, val_len : %d data_len : %d\n", train_len, val_len, data_len);
	for (int i = 0; i < train_len; i ++)
	{
		train_idx.push_back(queryImgNamelist(data_list, i));
	}
	for (int i = train_len; i < data_len; i++)
	{
		val_idx.push_back(queryImgNamelist(data_list, i));
	}

	//create val.csv
	FILE* v = NULL;
	fopen_s(&v, val_label_file.c_str(), "wb");
	if (NULL == v)
	{
		printf("open %s failed.\n", val_label_file.c_str());
		return bret;
	}
	const char* p = "img,label\r\n";
	fwrite(p, sizeof(char), strlen(p), v);

	vector<string>::iterator itVal;
	for (itVal = val_idx.begin(); itVal != val_idx.end(); itVal++)
	{
		std::string img_name = data_dir + *itVal;
		std::string lab_name = label_idx_dir + "\\" + *itVal;
		int dot = lab_name.rfind(".");
		lab_name = lab_name.replace(dot, 4, "_L.png.bin");
		char sz[512] = {0};
		sprintf_s(sz, 512, "%s,%s\r\n", img_name.c_str(), lab_name.c_str());
		fwrite(sz, sizeof(char), strlen(sz), v);
	}
	fclose(v);
	v = NULL;

	//create train.csv
	FILE* t = NULL;
	fopen_s(&t, train_label_file.c_str(), "wb");
	if (NULL == t)
	{
		printf("open %s failed.\n", train_label_file.c_str());
		return bret;
	}
	const char* q = "img,label\r\n";
	fwrite(q, sizeof(char), strlen(q), t);

	vector<string>::iterator itTrain;
	for (itTrain = train_idx.begin(); itTrain != train_idx.end(); itTrain++)
	{
		std::string img_name = data_dir + *itTrain;
		std::string lab_name = label_idx_dir + "\\" + *itTrain;
		int dot = lab_name.rfind(".");
		lab_name = lab_name.replace(dot, 4, "_L.png.bin");
		char zs[512] = { 0 };
		sprintf_s(zs, 512, "%s,%s\r\n", img_name.c_str(), lab_name.c_str());
		fwrite(zs, sizeof(char), strlen(zs), t);
	}
	fclose(t);
	t = NULL;

	bret = true;
	return bret;
}

typedef std::tuple<int, int, int> COLOR;
std::map<std::string, COLOR> label2color;
std::map<COLOR, std::string> color2label;
std::map<std::string, int> label2index;
std::map<int, std::string> index2label;
COLOR color;

bool parse_label()
{
	bool bret = true;

	std::ifstream flabel_colors_file(label_colors_file);
	
	if (!flabel_colors_file.is_open())
		throw std::runtime_error("Could not open label colors file");

	std::string r, g, b, label;
	int ir = 0, ig = 0, ib = 0;
	int accumulation = 0;
	while (flabel_colors_file.good())
	{
		while (getline(flabel_colors_file, r, ' '))
		{
			ir = atoi(r.c_str());
			getline(flabel_colors_file, g, ' ');
			ig = atoi(g.c_str());
			getline(flabel_colors_file, b, '	');
			ib = atoi(b.c_str());
			getline(flabel_colors_file, label, '\n');
			color = std::make_tuple(ib, ig, ir);
			//cout << "color: " << std::get<0>(color) << " " << std::get<1>(color) << " " << std::get<2>(color) << endl;
			label2color.insert(std::map<std::string, COLOR>::value_type(label, color));
			color2label.insert(std::map<COLOR, std::string>::value_type(color, label));
			label2index.insert(std::map<std::string, int>::value_type(label, accumulation));		//每种颜色对应1个数值,一共0 - 31
			index2label.insert(std::map<int, std::string>::value_type(accumulation++, label));
		}
	}

	//枚举出label图片的名字
	fileNameList_t	label_list;
	int label_len;

	listFiles(label_dir.c_str(), label_list);
	label_len = label_list.size();

	for (int i = 0; i < label_len; i ++)
	{
		std::string label_name = queryImgNamelist(label_list, i);
		std::string fullpath_label_name = label_idx_dir + "\\" + label_name + ".bin";
		std::string fullpath_on_one_hot_label_name = label_dir_no_one_hot + "\\" + label_name + ".bin";
		bool bf = fileExists(fullpath_label_name);
		if (bf)
		{
			printf("Skip %s\n", fullpath_label_name.c_str());
			continue;
		}
		else
		{
			printf("Parse %s\n", label_name.c_str());
			std::string imgPath = label_dir + label_name;
			cv::Mat3b img3b = cv::imread(imgPath);
			int height = img3b.rows;
			int weight = img3b.cols;
			//从这里开始,原pytorch_unet版本为np.array类型直接修改成librotch的tensor类型
			torch::Tensor idx_mat = torch::zeros({ height, weight }).toType(torch::kInt32);
			//cout << "idx_mat size " << idx_mat.sizes() << endl;
			for (int h = 0; h < height; h ++)
			{
				for (int w = 0; w < weight; w ++)
				{
					cv::Point point(w, h);
					const cv::Vec3b& bgr = img3b(point);
					int r = bgr[2];
					int g = bgr[1];
					int b = bgr[0];
					color = std::make_tuple(b,g,r);

					std::map<COLOR, std::string>::iterator itc2l;
					itc2l = color2label.find(color);
					if (itc2l == color2label.end())
					{
						bret = false;
						break;
					}
					std::string label = itc2l->second;
					std::map<std::string, int>::iterator itl2i;
					itl2i = label2index.find(label);
					if (itl2i == label2index.end())
					{
						bret = false;
						break;
					}
					int index = itl2i->second;
					idx_mat[h][w] = index;
				}
				if (!bret)
					break;
			}
			//create create one-hot encoding target
			torch::Tensor target_tensor = torch::zeros({ 32, idx_mat.sizes()[0], idx_mat.sizes()[1] }).toType(torch::kByte);
			for (int ih = 0; ih < idx_mat.sizes()[0]; ih++)
			{
				for (int iw = 0; iw < idx_mat.sizes()[1]; iw++)
				{
					target_tensor[idx_mat[ih][iw].item<int>()][ih][iw] = 1;
				}
			}
			
			FILE* fo = NULL;
			fopen_s(&fo, fullpath_on_one_hot_label_name.c_str(), "wb");
			if (fo == NULL)
			{
				bret = false;
				break;
			}
			idx_mat = idx_mat.toType(torch::kByte);
			fwrite(idx_mat.data_ptr(), sizeof(torch::kByte), height * weight * sizeof(torch::kByte), fo);
			printf("Finish write class_mat %s \n", label_name.c_str());
			fclose(fo);
			fo = NULL;

			FILE* f = NULL;
			fopen_s(&f, fullpath_label_name.c_str(), "wb");
			if (f == NULL)
			{
				bret = false;
				break;
			}
			fwrite(target_tensor.data_ptr(), sizeof(torch::kByte), 32 * height * weight * sizeof(torch::kByte), f);
			fclose(f);
			f = NULL;
			printf("Finish %s \n", label_name.c_str());
		}
		if (!bret)
			break;
	}
	if(!bret)
		printf("parse error.\n");
	return bret;
}

int main()
{
	bool b = false;
	b = directoryExists(root_dir.c_str());
	if (!b)
	{
		printf("%s no exists.\n", root_dir.c_str());
		return 0;
	}
	b = directoryExists(label_idx_dir.c_str());
	if (!b)
		b = makeDirectory(label_idx_dir.c_str());
	else
		printf("%s is exists.\n", label_idx_dir.c_str());

	b = directoryExists(label_dir_no_one_hot.c_str());
	if (!b)
		b = makeDirectory(label_dir_no_one_hot.c_str());
	else
		printf("%s is exists.\n", label_dir_no_one_hot.c_str());

	divide_train_val();
	parse_label();

	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
