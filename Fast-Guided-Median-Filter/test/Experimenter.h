#pragma once
#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <direct.h>
#include<opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <windows.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/optional.hpp>
#include <boost/tokenizer.hpp>
#include <boost/shared_array.hpp>
#include <direct.h>
#include <sys/stat.h>
#include <chrono>
#include <fstream>
//#include <opencv2/quality/qualitypsnr.hpp>
#include "DataContainer.h"
#include "FGMF.h"
#include "FGMF_type1.h"
#include "FGMF_type2.h"
#include "FGMF_type3.h"
//#include "GuidedFilter.h"
//#include "GMF_GPU.h"
#include "l1solver.h"
#include "ConstantTimeWMF.h"
#include "common.h"
#include "ExperimentManager.h"


#define MAX_PATH          260

class Experimenter
{
public:
	///////////////////////
	//動作テスト
	void test(std::string filePathSrc, std::string filePathGuide = "");


	///////////////////////
	//速度計測
	void performSpeedTest(std::string filePathSrc);


	///////////////////////
	//2D
	//ノイズ除去2D
	void performNoiseReduction2D();
	void postProcessForNoiseReduction2D(const cv::Mat& raw, const cv::Mat& result, double calculationTime, std::string fileName);
	static double calculatePSNR(const cv::Mat& I1, const cv::Mat& I2, int range);
	static double calculateSSIM(const cv::Mat& I1, const cv::Mat& I2);
	static double calculateEKI(const cv::Mat& I1, const cv::Mat& I2);

	///////////////////////
	//3D
	//動画に対する単なるフィルタリング
	void performFilteringForVideo(std::string filePathSrc, std::string filePathGuide = "");


	///////////////////////
	//論文用実験
	//2D 8bit画像速度テスト
	void speedTest2D8bitForPaper();
	//2D higher bit画像速度テスト
	void speedTest2DHigherBitForPaper();
	//2D マルチスペクトラル画像ノイズ除去テスト
	void noiseRemovalForMultispectralImageForPaper();
	void noiseRemovalForMultispectralImageForPaperNew();
	//3D 動画オプティカルフローリファインメントテスト（しない）
	void opticalFlowRefinementTtestForPaper();
	//4D ライトフィールドデノイジングテスト
	void noiseRemovalForLightFieldForPaper();
	//4D ライトフィールドディスパリティリファインメント
	void disparityRefinementForLightFieldForPaper();
	void disparityRefinementForLightFieldForPaperNew();
	void disparityRefinementForLightFieldForPaperNewColor();
	//フラッシュ/ノンフラッシュによるreversal artifact確認用
	void flashNoFlashForPaper();

	//追加実験
	//カラー画像ゴマ塩ノイズ除去評価テスト
	void noiseRemovalEvaluationForColorImage(std::string settingFileName);
	//視差推定リファインメント1枚ガイド評価テスト
	void noiseRemovalEvaluationForDepthImageGuideColor();
	void noiseRemovalEvaluationForDepthImageGuideGray();
	//Constant time のデバッグ用(もういらない)
	void testForConstantTimeWMF();
	//GPU4D color guideの確認用
	void testForProp4DColor();


	//論文用
	//100+とかでガイド色を圧縮したときの影響が出るか見る
	void colorQuantizationDifference();
	//fminとmedian trackingで差が生じる画素％を計算する
	void trackingDifference1();
	//fminとmedian trackingで差が生じる場所を見つける、データを出力する
	void trackingDifference2();
	//GFとSGFの結果 比較用
	void gfDifference();

	//Ablation study用
	void testForAblationStudy();


	//他の論文用　joint upsamplingのテスト
	void jointUpsamplingTest();


	//読み込みデータ
	//2D
	Container_Image* image;
	//3D
	Container_Video* video;




	//以下未使用


	//表示
	void showResult(const cv::Mat& I, std::string windowName);
	//保存
	void saveResult(const cv::Mat& I);


	//読み込むデータ名(フルパス)
	std::vector<std::string> fileNames;
	std::string fileName;


	//設定読み込み
	std::string getConfigFilePath(std::string fileNameWithoutExtension);
	void loadSettings(std::string settingFileName);

	std::string configFileName;

	//設定
	bool useGuideImage;

	//パラメータ
	//提案法
	struct Param_FGMF
	{
		int radius;
		float eps2;
		int Imax;
	};
	Param_FGMF param_fgmf;
};

/*
実験内容
速度
2次元グレー・グレー、カラー・カラー　画像サイズ、半径変化：大量データセット

アプリケーション
2次元：ノイズ除去、平滑化、detail enhancement, hdr compression, guided feathering
3次元：動画奥行き推定リファインメント：動画をpng化したデータをフォルダに格納　推定デプス画像、カラー画像(ガイド)
4次元：ライトフィールドノイズ除去：視点ごとにばらして000～番号付けしたデータをフォルダに格納　セルフガイド
マルチチャンネル：ハイパースペクトラルイメージに対するデノイズ：スペクトルごとにpng化したデータをフォルダに格納

データセット構成規則

*/