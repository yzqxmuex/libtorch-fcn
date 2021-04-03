#include<torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstring>
#include <io.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <direct.h>
#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

using namespace std;

bool fileExists(const std::string& name)
{
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

//判断目录是否存在
bool directoryExists(const char* pathname)
{
	struct stat info;

	if (stat(pathname, &info) != 0)
		return false;
	else if (info.st_mode & S_IFDIR)  // S_ISDIR() doesn't exist on my windows 
		return true;
	else
		return false;
}

//创建目录
bool makeDirectory(const char* dir)
{
	bool b = false;
	if (_mkdir(dir) == 0)
	{
		printf("Directory '%s' was successfully created\n", dir);
		b = true;
		//system("dir \\testtmp");
	}
	else
	{
		b = false;
		printf("Problem creating directory '%s'\n", dir);
	}

	return b;
}


void randperm(int Num, vector<int>& data_idx)
{
	for (int i = 0; i < Num; ++i)
	{
		data_idx.push_back(i);
	}
	random_shuffle(data_idx.begin(), data_idx.end());
}
typedef struct stuFileInfoList
{
	std::string		strFilePath;
	std::string		strFileName;
}STUFILEINFOLIST, *PSTUFILEINFOLIST;

typedef std::list< STUFILEINFOLIST > fileNameList_t;

void append(fileNameList_t &List, STUFILEINFOLIST &info);
void for_each(fileNameList_t &List);

void append(fileNameList_t &List, STUFILEINFOLIST &info)
{
	List.push_back(info);
}

//返回图片集中每份图片的名字
std::string queryImgNamelist(fileNameList_t &List, int check)
{
	int i = 0;
	fileNameList_t::iterator iter;
	for (iter = List.begin(); iter != List.end(); iter++,i++)
	{
		if (i == check)
			return iter->strFileName;
	}
	return "";
}

void listFiles(const char * dir, fileNameList_t& _t)
{
	std::string		strDir = dir;
	char dirNew[200];
	strcpy_s(dirNew, 200, dir);
	strcat_s(dirNew, 200, "*.png");    // 在目录后面加上"\\*.*"进行第一次搜索
	//cout <<"xxxx"<< dirNew << endl;

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // 检查是否成功
		return;

	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			//cout << findData.name << "\t<dir>\n";

			// 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
			strcpy_s(dirNew, 200, dir);
			strcat_s(dirNew, 200, "\\");
			strcat_s(dirNew, 200, findData.name);

			listFiles(dirNew, _t);
		}
		else
		{
			STUFILEINFOLIST stuFileInfo;
			stuFileInfo.strFilePath = strDir + "\\" + findData.name;
			stuFileInfo.strFileName = findData.name;
			//cout << findData.name << endl;
			append(/*listHrFileName_*/_t, stuFileInfo);
		}
		//cout << findData.name << "\t" << findData.size << " bytes.\n";
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // 关闭搜索句柄
}