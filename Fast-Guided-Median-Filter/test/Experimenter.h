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
	//����e�X�g
	void test(std::string filePathSrc, std::string filePathGuide = "");


	///////////////////////
	//���x�v��
	void performSpeedTest(std::string filePathSrc);


	///////////////////////
	//2D
	//�m�C�Y����2D
	void performNoiseReduction2D();
	void postProcessForNoiseReduction2D(const cv::Mat& raw, const cv::Mat& result, double calculationTime, std::string fileName);
	static double calculatePSNR(const cv::Mat& I1, const cv::Mat& I2, int range);
	static double calculateSSIM(const cv::Mat& I1, const cv::Mat& I2);
	static double calculateEKI(const cv::Mat& I1, const cv::Mat& I2);

	///////////////////////
	//3D
	//����ɑ΂���P�Ȃ�t�B���^�����O
	void performFilteringForVideo(std::string filePathSrc, std::string filePathGuide = "");


	///////////////////////
	//�_���p����
	//2D 8bit�摜���x�e�X�g
	void speedTest2D8bitForPaper();
	//2D higher bit�摜���x�e�X�g
	void speedTest2DHigherBitForPaper();
	//2D �}���`�X�y�N�g�����摜�m�C�Y�����e�X�g
	void noiseRemovalForMultispectralImageForPaper();
	void noiseRemovalForMultispectralImageForPaperNew();
	//3D ����I�v�e�B�J���t���[���t�@�C�������g�e�X�g�i���Ȃ��j
	void opticalFlowRefinementTtestForPaper();
	//4D ���C�g�t�B�[���h�f�m�C�W���O�e�X�g
	void noiseRemovalForLightFieldForPaper();
	//4D ���C�g�t�B�[���h�f�B�X�p���e�B���t�@�C�������g
	void disparityRefinementForLightFieldForPaper();
	void disparityRefinementForLightFieldForPaperNew();
	void disparityRefinementForLightFieldForPaperNewColor();
	//�t���b�V��/�m���t���b�V���ɂ��reversal artifact�m�F�p
	void flashNoFlashForPaper();

	//�ǉ�����
	//�J���[�摜�S�}���m�C�Y�����]���e�X�g
	void noiseRemovalEvaluationForColorImage(std::string settingFileName);
	//�������胊�t�@�C�������g1���K�C�h�]���e�X�g
	void noiseRemovalEvaluationForDepthImageGuideColor();
	void noiseRemovalEvaluationForDepthImageGuideGray();
	//Constant time �̃f�o�b�O�p(��������Ȃ�)
	void testForConstantTimeWMF();
	//GPU4D color guide�̊m�F�p
	void testForProp4DColor();


	//�_���p
	//100+�Ƃ��ŃK�C�h�F�����k�����Ƃ��̉e�����o�邩����
	void colorQuantizationDifference();
	//fmin��median tracking�ō����������f�����v�Z����
	void trackingDifference1();
	//fmin��median tracking�ō���������ꏊ��������A�f�[�^���o�͂���
	void trackingDifference2();
	//GF��SGF�̌��� ��r�p
	void gfDifference();

	//Ablation study�p
	void testForAblationStudy();


	//���̘_���p�@joint upsampling�̃e�X�g
	void jointUpsamplingTest();


	//�ǂݍ��݃f�[�^
	//2D
	Container_Image* image;
	//3D
	Container_Video* video;




	//�ȉ����g�p


	//�\��
	void showResult(const cv::Mat& I, std::string windowName);
	//�ۑ�
	void saveResult(const cv::Mat& I);


	//�ǂݍ��ރf�[�^��(�t���p�X)
	std::vector<std::string> fileNames;
	std::string fileName;


	//�ݒ�ǂݍ���
	std::string getConfigFilePath(std::string fileNameWithoutExtension);
	void loadSettings(std::string settingFileName);

	std::string configFileName;

	//�ݒ�
	bool useGuideImage;

	//�p�����[�^
	//��Ė@
	struct Param_FGMF
	{
		int radius;
		float eps2;
		int Imax;
	};
	Param_FGMF param_fgmf;
};

/*
�������e
���x
2�����O���[�E�O���[�A�J���[�E�J���[�@�摜�T�C�Y�A���a�ω��F��ʃf�[�^�Z�b�g

�A�v���P�[�V����
2�����F�m�C�Y�����A�������Adetail enhancement, hdr compression, guided feathering
3�����F���扜�s�����胊�t�@�C�������g�F�����png�������f�[�^���t�H���_�Ɋi�[�@����f�v�X�摜�A�J���[�摜(�K�C�h)
4�����F���C�g�t�B�[���h�m�C�Y�����F���_���Ƃɂ΂炵��000�`�ԍ��t�������f�[�^���t�H���_�Ɋi�[�@�Z���t�K�C�h
�}���`�`�����l���F�n�C�p�[�X�y�N�g�����C���[�W�ɑ΂���f�m�C�Y�F�X�y�N�g�����Ƃ�png�������f�[�^���t�H���_�Ɋi�[

�f�[�^�Z�b�g�\���K��

*/