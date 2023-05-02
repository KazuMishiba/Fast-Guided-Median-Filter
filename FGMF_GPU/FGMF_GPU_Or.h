#pragma once
#include "CalculateDC.cuh"
#include "FGMF_GPU_Or.cuh"


namespace FGMF_GPU_Or
{
    // 
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int blockSize);
	cv::Mat filter_2d_multichannel(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int blockSize);
	std::vector<cv::Mat> filter_2d_multichannel_channelRadius(std::vector<cv::Mat>& f_img_channels, std::vector<cv::Mat>& g_img_channels, int radius2D, int radiusChannel, float epsilon, int fRange, int blockSize, bool exclude_same_guide_channel);
	std::vector<cv::Mat> filter_Nd(std::vector<cv::Mat>& f_imgs, std::vector<cv::Mat>& g_imgs, int radius2D, std::vector<int> radii3DAndUp, std::vector<int> size3DAndUp, float eepsilonps2, int fRange, int blockSize);


	//For management of sums in a window of guide image
	template<typename DC_TYPE>
	class SumG_Manager
	{
	public:
		SumG_Manager(Helper::SizeInfo& sizeInfo);
		~SumG_Manager();
		void newSumGs(std::vector<int> oldIndices, std::vector<int> newIndices);
		void addSumGs(std::vector<int> indices);
		void remSumGs(std::vector<int> indices);
		void calculateAndSaveSumG(std::vector<cv::Mat> gs, int radius2D);
	};

	template<>
	class SumG_Manager<float2>
	{
	public:
		SumG_Manager(Helper::SizeInfo& sizeInfo)
			: sizeInfo_(sizeInfo) {
			Helper::UtilityForCUDA::allocateDeviceMemoryWithZero(sumG_total_, sizeInfo_, NULL);
		}
		~SumG_Manager() {
			cudaFree(sumG_total_);
		};

		//Calculate the sum in the window of all 2D guide images and save them as std::vector<cv::Mat>
		void calculateAndSaveSumG(std::vector<cv::Mat> gs, int radius2D) {
			sumGs_device_.resize(gs.size());
			sumGs_.resize(gs.size());

			int2* sumG;
			int2* temp_;
			Helper::UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo_);
			Helper::UtilityForCUDA::allocateDeviceMemory(temp_, sizeInfo_);
			int* g;
			Helper::UtilityForCUDA::allocateDeviceMemory(g, sizeInfo_);
			for (size_t i = 0; i < gs.size(); i++)
			{
				Helper::UtilityForCUDA::uploadMatToDevice(gs[i], g, sizeInfo_);
				Helper::UtilityForCUDA::initializeDeviceMemoryWithZero(sumG, sizeInfo_);
				cu_calculateSumG(sizeInfo_, NULL, g, radius2D, sumG, temp_);
				sumGs_[i] = Helper::UtilityForCUDA::downloadLinearArrayAsMat(sumG, sizeInfo_);

			}

			cudaFree(temp_);
			cudaFree(sumG);
			cudaFree(g);
		}
		//Calculate sumG_total_ from scratch
		void newSumGs(std::vector<int> oldIndices, std::vector<int> newIndices) {
			for (int index : oldIndices) {
				cudaFree(sumGs_device_[index]);
			}
			// Initialize with 0
			Helper::UtilityForCUDA::initializeDeviceMemoryWithZero(sumG_total_, sizeInfo_, NULL);
			addSumGs(newIndices);
		}
		//Add to sumG_total_
		void addSumGs(std::vector<int> indices) {
			for (int index : indices) {
				Helper::UtilityForCUDA::allocateDeviceMemory(sumGs_device_[index], sumGs_[index], sizeInfo_, NULL);
				cu_addSumG(sizeInfo_, NULL, sumGs_device_[index], sumG_total_);
			}
		}
		//Remove from sumG_total_
		void remSumGs(std::vector<int> indices) {
			for (int index : indices) {
				cu_remSumG(sizeInfo_, NULL, sumGs_device_[index], sumG_total_);
				cudaFree(sumGs_device_[index]);
			}
		}

		int2* sumG_total_;
	private:
		Helper::SizeInfo& sizeInfo_;
		std::vector<int2*> sumGs_device_;
		std::vector<cv::Mat> sumGs_;
	};

	template<>
	class SumG_Manager<float4>
	{
	public:
		SumG_Manager(Helper::SizeInfo& sizeInfo)
			: sizeInfo_(sizeInfo) {
			sumG_total_ = new Helper::DeviceArray<int>(3, sizeInfo, true);
			sumGG_total_ = new Helper::DeviceArray<int>(6, sizeInfo, true);
		}
		~SumG_Manager() {
			delete sumG_total_;
			delete sumGG_total_;
			
			for (auto it = sumGs_device_.begin(); it != sumGs_device_.end(); ++it) {
				if (*it != nullptr) {
					delete* it;
					*it = nullptr;
				}
			}
			for (auto it = sumGGs_device_.begin(); it != sumGGs_device_.end(); ++it) {
				if (*it != nullptr) {
					delete* it;
					*it = nullptr;
				}
			}
		};
		//Calculate the sum in the window of all 2D guide images and save them as std::vector<cv::Mat>
		void calculateAndSaveSumG(std::vector<cv::Mat> gs, int radius2D) {
			sumGs_device_.resize(gs.size());
			sumGGs_device_.resize(gs.size());
			sumGs_.resize(gs.size());
			sumGGs_.resize(gs.size());

			Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(3, sizeInfo_);
			Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>(6, sizeInfo_);
			Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(3, sizeInfo_);
			Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>(6, sizeInfo_);

			for (size_t i = 0; i < gs.size(); i++)
			{
				Helper::DeviceArray<int>* g = new Helper::DeviceArray<int>(gs[i], sizeInfo_);

				sumG->initializeWithZero(sizeInfo_);
				sumGG->initializeWithZero(sizeInfo_);
				cu_calculateSumG3(sizeInfo_, NULL, g, radius2D, sumG, sumGG, tempG, tempGG);
				sumGs_[i] = Helper::UtilityForCUDA::downloadLinearArrayAsMat(sumG, sizeInfo_);
				sumGGs_[i] = Helper::UtilityForCUDA::downloadLinearArrayAsMat(sumGG, sizeInfo_);

				delete g;
			}


			delete sumG;
			delete sumGG;
			delete tempG;
			delete tempGG;
		}
		//Calculate sumG_total_ from scratch
		void newSumGs(std::vector<int> oldIndices, std::vector<int> newIndices) {
			for (int index : oldIndices) {
				delete sumGs_device_[index];
				delete sumGGs_device_[index];
				sumGs_device_[index] = nullptr;
				sumGGs_device_[index] = nullptr;
			}
			// Initialize with 0
			sumG_total_->initializeWithZero(sizeInfo_);
			sumGG_total_->initializeWithZero(sizeInfo_);
			addSumGs(newIndices);
		}
		//Add to sumG_total_
		void addSumGs(std::vector<int> indices) {
			for (int index : indices) {
				sumGs_device_[index] = new Helper::DeviceArray<int>(sumGs_[index], sizeInfo_);
				sumGGs_device_[index] = new Helper::DeviceArray<int>(sumGGs_[index], sizeInfo_);
				cu_addSumG3(sizeInfo_, NULL, sumGs_device_[index], sumGGs_device_[index], sumG_total_, sumGG_total_);
			}
		}
		//Remove from sumG_total_
		void remSumGs(std::vector<int> indices) {
			for (int index : indices) {
				cu_remSumG3(sizeInfo_, NULL, sumGs_device_[index], sumGGs_device_[index], sumG_total_, sumGG_total_);
				delete sumGs_device_[index];
				delete sumGGs_device_[index];
				sumGs_device_[index] = nullptr;
				sumGGs_device_[index] = nullptr;
			}
		}


		Helper::DeviceArray<int>* sumG_total_;
		Helper::DeviceArray<int>* sumGG_total_;
	private:
		Helper::SizeInfo& sizeInfo_;
		std::vector<Helper::DeviceArray<int>*> sumGs_device_;
		std::vector<Helper::DeviceArray<int>*> sumGGs_device_;
		std::vector<cv::Mat> sumGs_;
		std::vector<cv::Mat> sumGGs_;
	};


	//For 2D
	class WMF_2D
	{
	public:
		WMF_2D(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int blockSize)
			: f_img_(f_img), g_img_(g_img), radius_(radius), epsilon_(epsilon), fRange_(fRange), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels()), sizeInfo_(f_img.cols, f_img.rows, fRange, dim3(blockSize,blockSize,1)), originalDepth_(f_img_.depth())
		{
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);

			f_device_ = new Helper::DeviceArray<int>(f_img_, sizeInfo_);
			g_device_ = new Helper::DeviceArray<int>(g_img_, sizeInfo_);
			result_device_ = new Helper::DeviceArray<int>(channelNum_f_, sizeInfo_);
		}

		// Applies the 2D GPU-O(r) filter to grayscane/color image.
		void apply_2d_filter();
		// Applies the 2D GPU-O(r) filter to multispectral image.
		void apply_2d_filter_multichannel();
		// Downloads image data from CUDA device memory to host memory and returns it in cv::Mat format.
		cv::Mat downloadResultAsMat();

		~WMF_2D()
		{
			delete f_device_;
			delete g_device_;
			delete result_device_;
		}


	private:
		cv::Mat f_img_;                           // Input image
		cv::Mat g_img_;                           // Guide image
		int radius_;                              // Filter radius
		float epsilon_;                           // Epsilon parameter for the filter
		int fRange_;                              // Range of intensity values of input data
		int originalDepth_;                       // Original depth of input image
		int channelNum_f_;                        // Number of channels in the input image
		int channelNum_g_;                        // Number of channels in the guide image

		Helper::SizeInfo sizeInfo_;               // Size information of the input image for CUDA
		Helper::DeviceArray<int>* f_device_;      // Input image on the device
		Helper::DeviceArray<int>* g_device_;      // Guide image on the device
		Helper::DeviceArray<int>* result_device_; // Result image on the device


		template<typename FG_TYPE, typename DC_TYPE, int CHANNNEL_NUM_F, int CHANNNEL_NUM_G>
		void filtering();
	};


	template<typename DC_TYPE>
	class WMF_ND
	{
	public:
		WMF_ND(std::vector<cv::Mat>& f_imgs, std::vector<cv::Mat>& g_imgs, int radius2D, std::vector<int>& radii3DAndUp, std::vector<int>& size3DAndUp, float epsilon, int fRange, int blockSize)
			: f_imgs_(f_imgs), g_imgs_(g_imgs), radius2D_(radius2D), radii3DAndUp_(radii3DAndUp), size3DAndUp_(size3DAndUp), epsilon_(epsilon), fRange_(fRange), channelNum_f_(f_imgs[0].channels()), channelNum_g_(g_imgs[0].channels()), sizeInfo_(f_imgs[0].cols, f_imgs[0].rows, fRange, dim3(blockSize, blockSize, 1)), originalDepth_(f_imgs_[0].depth())
		{
			elementNumOf2D_ = 1;
			for (int dimension : size3DAndUp_)
				elementNumOf2D_ *= dimension;

			if (f_imgs_[0].depth() != CV_32S)
				for (int i = 0; i < f_imgs_.size(); i++)
					f_imgs_[i].convertTo(f_imgs_[i], CV_32S);
			if (g_imgs_[0].depth() != CV_32S)
				for (int i = 0; i < g_imgs_.size(); i++)
					g_imgs_[i].convertTo(g_imgs_[i], CV_32S);


			f_device_.resize(elementNumOf2D_);
			g_device_.resize(elementNumOf2D_);

			result_device_ = new Helper::DeviceArray<int>(channelNum_f_, sizeInfo_);
			results_.resize(elementNumOf2D_);
			sumG_Manager_ = new SumG_Manager<DC_TYPE>(sizeInfo_);
		}

		// Applies the GPU-O(r) filter to grayscane/color multidimensional data.
		std::vector<cv::Mat> apply_nd_filter();


		~WMF_ND()
		{
			for (Helper::DeviceArray<int>*& dev : f_device_)
				delete dev;
			for (Helper::DeviceArray<int>*& dev : g_device_)
				delete dev;
			delete result_device_;
			delete sumG_Manager_;
		}

	private:
		std::vector<cv::Mat> f_imgs_;                     // Input data array
		std::vector<cv::Mat> g_imgs_;                     // Guide data array
		int radius2D_;                                    // Filter radius in the first two dimensions (image data)
		std::vector<int> radii3DAndUp_;                   // Filter radii in the third and subsequent dimensions
		std::vector<int> size3DAndUp_;                    // Size of data in the third and subsequent dimensions
		int elementNumOf2D_;                              // Number of 2D data (= product of size3DAndUp_ )
		float epsilon_;                                   // Epsilon parameter for the filter
		int fRange_;                                      // Range of intensity values of input data
		int originalDepth_;                               // Original depth of input data
		int channelNum_f_;                                // Number of channels in the input data
		int channelNum_g_;                                // Number of channels in the guide data

		Helper::SizeInfo sizeInfo_;                       // Size information of the input image for CUDA
		std::vector<Helper::DeviceArray<int>*> f_device_; // Input data array on the device
		std::vector<Helper::DeviceArray<int>*> g_device_; // Guide data array on the device
		Helper::DeviceArray<int>* result_device_;         // Result image on the device
		std::vector<cv::Mat> results_;                    // Result image array on the host

		SumG_Manager<DC_TYPE>* sumG_Manager_;

		void filtering();

		//Free Helper::DeviceArray<int>* at specified index
		void release_device_arrays_at_indices(std::vector<int>& indices);
		//Upload cv::Mat to Helper::DeviceArray<int>* at specified index
		void initialize_device_arrays_at_indices(const std::vector<int>& indices);


		void calculateDC(int index, int pixelNumInWindow, DC_TYPE* dc);

	};




}