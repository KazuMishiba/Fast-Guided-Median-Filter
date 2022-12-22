//###########################################
//#  Ver 1.0
//###########################################
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <any>
#include "json.hpp"

//パラメータ
class ExperimentManager_Parameter
{
public:
	ExperimentManager_Parameter() {}
	ExperimentManager_Parameter(std::map<std::string, std::any> p) {
		this->param = p;
	}
	std::map<std::string, std::any> param;

	template <typename T>
	T get(std::string name) {
		return std::any_cast<T>(param.at(name));
	}

};

//手法
class ExperimentManager_Method
{
public:
	//手法名
	std::string name;
	//手法パラメータ
	std::list< ExperimentManager_Parameter> parameters;
};

//データ
struct ExperimentManager_Data
{
	//input list
	std::vector<std::string> dataNames;
	//gt list
	std::vector<std::string> gtNames;
	//result list
	std::vector<std::string> resultNames;
};

//全体
class ExperimentManager
{
public:
	ExperimentManager(std::string jsonFileName);
	~ExperimentManager();
	//手法
	std::map<std::string, ExperimentManager_Method> methods;
	//データ
	ExperimentManager_Data data;

	//書き出し用
	// 保存先のフォルダを生成し、保存ファイル名を返す
	std::string getResultPath(std::string resultBaseDir, std::string methodName, std::string id, std::string saveFileName);
	//mapデータをJSON形式で保存
	void saveResultInJSON(std::string resultBaseDir, std::string methodName, std::string id, std::string saveFileName, std::map<std::string, float> mapForJson, std::string fileModeStr);
private:
	//ExperimentManagerのバージョン
	std::string ver;
	//任意のコメント
	std::string comment;

	//ファイルから読み込み処理
	std::vector<std::string> readFromFile(std::string fileName);

};

/*
ExperimentManager em = ExperimentManager(settingFileName);
for (int i = 0; i < em.data.dataNames.size(); i++)
{
	for (auto &param : em.methods.at(methodName).parameters)
	{
		int val_int = param.get<int>("パラメータ名");
		float val_float = param.get<float>("パラメータ名");
		std::string val_str = param.get<std::string>("パラメータ名");
	}
}
*/

inline ExperimentManager::ExperimentManager(std::string jsonFileName)
{
	std::ifstream fin(jsonFileName, std::ios::in);
	if (!fin) {
		std::cerr << jsonFileName << " does not exist." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	//読み込み
	nlohmann::json json;
	fin >> json;

	try
	{
		//パラメータ展開版かどうか
		if (json["parameterDev"].get<int>() != 1)
		{
			std::cerr << jsonFileName << " はパラメータ展開版ではない。" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		//バージョン
		this->ver = json["ver"];

		//リスト読み込み
		std::string dataListName = json["data"]["dataListName"];
		std::string gtListName = json["data"]["gtListName"];
		std::string resultListName = json["data"]["resultListName"];

		//ファイルから読み込み
		if (filesystem::exists(dataListName))
		{
			this->data.dataNames = this->readFromFile(dataListName);
		}
		else
		{
			std::cerr << dataListName << " does not exist." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		if (json["data"]["gtExist"].get<int>())
		{
			//真値読み込み
			if (filesystem::exists(gtListName))
			{
				this->data.gtNames = this->readFromFile(gtListName);
			}
			else
			{
				std::cerr << gtListName << " does not exist." << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}
		if (filesystem::exists(resultListName))
		{
			this->data.resultNames = this->readFromFile(resultListName);
		}
		else
		{
			std::cerr << resultListName << " does not exist." << std::endl;
			std::exit(EXIT_FAILURE);
		}


		//手法
		auto methods_json = json["methods"];
		for (auto& m : methods_json)
		{
			auto parameters_json = m["parameters"];

			std::list<ExperimentManager_Parameter> params;
			std::map<std::string, std::string> types = m["parameters"]["types"].get<std::map<std::string, std::string>>();


			for (auto& p : m["parameters"]["params"]) {
				ExperimentManager_Parameter ep;
				for (const auto& [key, value] : p.items()) {
					for (const auto& [keyType, valueType] : types) {
						if (key == keyType)
						{
							//valueの型がintだったとしても、valueTypeの指定がfloatだったら
							//(float)value.get<int>(); とする必要があるので
							if (valueType == "int")
							{
								ep.param.emplace(key, value.get<int>());
							}
							else if (valueType == "float")
							{
								ep.param.emplace(key, value.get<float>());
							}
							else if (valueType == "str")
							{
								ep.param.emplace(key, value.get<std::string>());
							}
							else
							{
								std::cerr << "Unknown type." << std::endl;
								std::exit(EXIT_FAILURE);
							}
							break;
						}
					}

				}
				params.push_back(ep);
			}
			ExperimentManager_Method emm;

			emm.name = m["name"];
			emm.parameters = params;
			this->methods[m["name"]] = emm;
		}
	}
	catch (nlohmann::json::parse_error& ex)
	{
		std::cerr << "parse error at byte " << ex.byte << std::endl;
	}
}

inline ExperimentManager::~ExperimentManager()
{
}

//ファイルから1行ずつ読み取る
inline std::vector<std::string> ExperimentManager::readFromFile(std::string fileName)
{
	std::vector<std::string> list;
	std::ifstream fs(fileName);

	std::string tmp;
	while (std::getline(fs, tmp))
	{
		if (tmp != "")
		{
			list.push_back(tmp);
		}
	}
	return list;
}

inline std::string ExperimentManager::getResultPath(std::string resultBaseDir, std::string methodName, std::string id, std::string saveFileName = "")
{
	std::string resultDir = resultBaseDir + "\\" + methodName + "\\" + id + "\\";
	if (!std::filesystem::exists(resultDir))
	{
		//ディレクトリの作成
		try {
			std::filesystem::create_directories(resultDir);
		}
		catch (std::exception& e) {
			std::cerr << "結果保存フォルダの作成失敗：" << resultDir << std::endl;
		}
	}
	std::string resultPath = resultDir + saveFileName;
	return resultPath;
}

inline void ExperimentManager::saveResultInJSON(std::string resultBaseDir, std::string methodName, std::string id, std::string saveFileName, std::map<std::string, float> mapForJson, std::string fileModeStr = "w")
{
	std::ios_base::openmode fileMode;
	if (fileModeStr == "w")
	{
		fileMode = std::ios::trunc;//上書き保存
	}
	else if (fileModeStr == "a")
	{
		fileMode = std::ios::app;
	}
	else
	{
		std::cerr << "ファイル書き込みモード指定エラー: " << fileModeStr << std::endl;
		std::exit(EXIT_FAILURE);
	}
	nlohmann::json json(mapForJson);
	std::string resultPath = this->getResultPath(resultBaseDir, methodName, id, saveFileName);
	//std::cout << resultPath << std::endl;
	std::ofstream of;

	try {
		of.open(resultPath, fileMode);
		of << std::setw(4) << json << std::endl;
		of.close();
	}
	catch (std::exception& e) {
		std::cerr << "Exception" << std::endl;
	}
}

