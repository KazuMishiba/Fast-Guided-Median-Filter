#include "FGMF_GPU_Or.h"

namespace FGMF_GPU_Or
{
	//For 2D grayscale and color images
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int blockSize)
	{
		WMF_2D wmf = WMF_2D(f_img, g_img, radius, epsilon, fRange, blockSize);
		wmf.apply_2d_filter();
		return wmf.downloadResultAsMat();
	}

	void WMF_2D::apply_2d_filter()
	{
		if (channelNum_f_ == 1 && channelNum_g_ == 1)
			filtering<int2, float2, 1, 1>();
		else if (channelNum_f_ == 1 && channelNum_g_ == 3)
			filtering<int4, float4, 1, 3>();
		else if (channelNum_f_ == 3 && channelNum_g_ == 1)
			filtering<int2, float2, 3, 1>();
		else if (channelNum_f_ == 3 && channelNum_g_ == 3)
			filtering<int4, float4, 3, 3>();
	}

	template<typename FG_TYPE, typename DC_TYPE, int CHANNNEL_NUM_F, int CHANNNEL_NUM_G>
	void WMF_2D::filtering()
	{
		int pixelNumInWindow = (radius_ * 2 + 1) * (radius_ * 2 + 1);
		DC_TYPE* dc;
		Helper::UtilityForCUDA::allocateDeviceMemory(dc, sizeInfo_);

		cu_calculateDC(sizeInfo_, NULL, g_device_, radius_, pixelNumInWindow, epsilon_, dc);
		FGMF_GPU_Or::cu_filter2D<FG_TYPE, DC_TYPE, CHANNNEL_NUM_F, CHANNNEL_NUM_G>(sizeInfo_, NULL, radius_, fRange_, result_device_, f_device_, g_device_, dc);

		cudaFree(dc);
	}


	//For 2D multichannel image
	cv::Mat filter_2d_multichannel(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int blockSize)
	{
		WMF_2D wmf = WMF_2D(f_img, g_img, radius, epsilon, fRange, blockSize);
		wmf.apply_2d_filter_multichannel();
		return wmf.downloadResultAsMat();
	}

	void WMF_2D::apply_2d_filter_multichannel()
	{
		int n = channelNum_g_;
		Helper::DeviceArray<float>* dc = new Helper::DeviceArray<float>(n + 1, sizeInfo_);
		int pixelNumInWindow = (radius_ * 2 + 1) * (radius_ * 2 + 1);

		cu_calculateDCx(sizeInfo_, NULL, g_device_, radius_, pixelNumInWindow, epsilon_, dc);
		FGMF_GPU_Or::cu_filter2D_Multichannel(sizeInfo_, NULL, radius_, fRange_, result_device_, f_device_, g_device_, dc);

		delete dc;
	}

	//For 2D multichannel image with channel radius
	std::vector<cv::Mat> filter_2d_multichannel_channelRadius(std::vector<cv::Mat>& f_img_channels, std::vector<cv::Mat>& g_img_channels, int radius2D, int radiusChannel, float epsilon, int fRange, int blockSize, bool avoidSameChannelAsGuide)
	{
		std::vector<cv::Mat> results;
		for (int i = 0; i < f_img_channels.size(); i++)
		{
			//Set guide image for i-th channel
			std::vector<cv::Mat> g_channels;
			for (int j = std::max(0, i - radiusChannel); j <= std::min(i + radiusChannel, (int)g_img_channels.size() - 1); j++)
				if (!avoidSameChannelAsGuide || (j != i && avoidSameChannelAsGuide))
					g_channels.push_back(g_img_channels[j]);
			cv::Mat g;
			cv::merge(g_channels, g);

			cv::Mat result = filter_2d_multichannel(f_img_channels[i], g, radius2D, epsilon, fRange, blockSize);
			results.push_back(result);
		}
		return results;
	}


	cv::Mat WMF_2D::downloadResultAsMat()
	{
		cv::Mat result = Helper::UtilityForCUDA::downloadLinearArrayAsMat(result_device_, sizeInfo_);
		// Restore the data type
		if (result.depth() != originalDepth_)
			result.convertTo(result, originalDepth_);

		return result;
	}



	//////////////////////////////////////////////////////////////////////
	//For Multidimensional data

	//Index calculation

	// Function to convert the index to N-dimensional coordinates
	std::vector<int> indexToCoordinates(int index, const std::vector<int>& dimensions) {
		std::vector<int> coords(dimensions.size(), 0);
		for (size_t i = 0; i < dimensions.size(); ++i) {
			coords[i] = index % dimensions[i];
			index /= dimensions[i];
		}
		return coords;
	}

	// Function to convert N-dimensional coordinates to an index
	int coordinatesToIndex(const std::vector<int>& coords, const std::vector<int>& dimensions) {
		int index = 0;
		int factor = 1;
		for (size_t i = 0; i < dimensions.size(); ++i) {
			index += coords[i] * factor;
			factor *= dimensions[i];
		}
		return index;
	}

	// Recursive helper function for getNeighbors
	void getNeighborsRecursive(int dim, std::vector<int>& current_coords, std::vector<int>& neighbors, const std::vector<int>& coords, const std::vector<int>& dimensions, const std::vector<int>& radii) {
		if (dim == static_cast<int>(dimensions.size())) {
			neighbors.push_back(coordinatesToIndex(current_coords, dimensions));
		}
		else {
			for (int i = -radii[dim]; i <= radii[dim]; ++i) {
				int new_coord = coords[dim] + i;
				if (new_coord >= 0 && new_coord < dimensions[dim]) {
					current_coords[dim] = new_coord;
					getNeighborsRecursive(dim + 1, current_coords, neighbors, coords, dimensions, radii);
				}
			}
		}
	}

	// Function to get the neighbors of a point in N-dimensional space
	std::vector<int> getNeighbors(const std::vector<int>& coords, const std::vector<int>& dimensions, const std::vector<int>& radii) {
		std::vector<int> neighbors;
		std::vector<int> current_coords(coords);
		getNeighborsRecursive(0, current_coords, neighbors, coords, dimensions, radii);
		return neighbors;
	}


	std::pair<std::vector<int>, std::vector<int>> getEnteringAndLeaving(const std::vector<int>& old_coords, const std::vector<int>& new_coords, const std::vector<int>& dimensions, const std::vector<int>& radii) {
		std::vector<int> old_neighbors = getNeighbors(old_coords, dimensions, radii);
		std::vector<int> new_neighbors = getNeighbors(new_coords, dimensions, radii);

		std::set<int> old_neighbors_set(old_neighbors.begin(), old_neighbors.end());
		std::set<int> new_neighbors_set(new_neighbors.begin(), new_neighbors.end());

		std::vector<int> entering_neighbors, leaving_neighbors;
		for (const auto& neighbor : new_neighbors) {
			if (old_neighbors_set.find(neighbor) == old_neighbors_set.end()) {
				entering_neighbors.push_back(neighbor);
			}
		}
		for (const auto& neighbor : old_neighbors) {
			if (new_neighbors_set.find(neighbor) == new_neighbors_set.end()) {
				leaving_neighbors.push_back(neighbor);
			}
		}
		return { entering_neighbors, leaving_neighbors };
	}

	bool shouldRecomputeNeighbors(const std::vector<int>& old_coords, const std::vector<int>& new_coords, const std::vector<int>& radii) {
		for (size_t i = 0; i < old_coords.size(); ++i) {
			if (std::abs(old_coords[i] - new_coords[i]) > radii[i]) {
				return true;
			}
		}
		return false;
	}



	//Filtering for N-dimension data
	std::vector<cv::Mat> filter_Nd(std::vector<cv::Mat>& f_imgs, std::vector<cv::Mat>& g_imgs, int radius2D, std::vector<int> radii3DAndUp, std::vector<int> size3DAndUp, float epsilon, int fRange, int blockSize)
	{
		if (g_imgs[0].channels() == 1)
		{
			WMF_ND<float2> wmf = WMF_ND<float2>(f_imgs, g_imgs, radius2D, radii3DAndUp, size3DAndUp, epsilon, fRange, blockSize);
			return wmf.apply_nd_filter();
		}
		else if (g_imgs[0].channels() == 3)
		{
			WMF_ND<float4> wmf = WMF_ND<float4>(f_imgs, g_imgs, radius2D, radii3DAndUp, size3DAndUp, epsilon, fRange, blockSize);
			return wmf.apply_nd_filter();
		}
		else
		{
			return std::vector<cv::Mat>();
		}
	}


	template<typename DC_TYPE>
	std::vector<cv::Mat> WMF_ND<DC_TYPE>::apply_nd_filter()
	{
		filtering();
		return results_;
	}

	//N-dimension
	template<typename DC_TYPE>
	void WMF_ND<DC_TYPE>::filtering()
	{
		DC_TYPE* dc;
		Helper::UtilityForCUDA::allocateDeviceMemory(dc, sizeInfo_);

		//Calculate the sum in the window of all 2D guide images first.
		sumG_Manager_->calculateAndSaveSumG(g_imgs_, radius2D_);

		std::vector<int> coords;
		std::vector<int> oldNeighborIndices;
		std::vector<int> neighborIndices;
		bool recomputeNeighborsFlag;
		std::vector<int> entering_neighbors, leaving_neighbors;

		for (int index = 0; index < elementNumOf2D_; ++index) {
			//For efficient computation, sumG is updated using a multidimensional sliding window.
			if (index != 0)
			{
				std::vector<int> new_coords = indexToCoordinates(index, size3DAndUp_);
				recomputeNeighborsFlag = shouldRecomputeNeighbors(coords, new_coords, radii3DAndUp_);
				if (recomputeNeighborsFlag) {
					oldNeighborIndices = neighborIndices;
					coords = new_coords;
					neighborIndices = getNeighbors(coords, size3DAndUp_, radii3DAndUp_);
				}
				else {
					std::tie(entering_neighbors, leaving_neighbors) = getEnteringAndLeaving(coords, new_coords, size3DAndUp_, radii3DAndUp_);
					for (const auto& leaving_neighbor : leaving_neighbors) {
						neighborIndices.erase(std::remove(neighborIndices.begin(), neighborIndices.end(), leaving_neighbor), neighborIndices.end());
					}
					neighborIndices.insert(neighborIndices.end(), entering_neighbors.begin(), entering_neighbors.end());
					coords = new_coords;
				}
			}
			else {
				oldNeighborIndices = {};
				coords = indexToCoordinates(0, size3DAndUp_);
				neighborIndices = getNeighbors(coords, size3DAndUp_, radii3DAndUp_);
				recomputeNeighborsFlag = true;
			}

			if (recomputeNeighborsFlag)
			{
				//Recompute;
				release_device_arrays_at_indices(oldNeighborIndices);
				initialize_device_arrays_at_indices(neighborIndices);
				sumG_Manager_->newSumGs(oldNeighborIndices, neighborIndices);
			}
			else {
				if (entering_neighbors.size())
				{
					//Entering;
					initialize_device_arrays_at_indices(entering_neighbors);
					sumG_Manager_->addSumGs(entering_neighbors);
				}
				if (leaving_neighbors.size())
				{
					//Leaving;
					release_device_arrays_at_indices(leaving_neighbors);
					sumG_Manager_->remSumGs(leaving_neighbors);
				}
			}


			int pixelNumInWindow = (radius2D_ * 2 + 1) * (radius2D_ * 2 + 1) * neighborIndices.size();
			//Calculate d and c
			calculateDC(index, pixelNumInWindow, dc);


			std::vector<Helper::DeviceArray<int>*> f, g;
			f.reserve(neighborIndices.size());
			g.reserve(neighborIndices.size());
			for (int neighborIndex : neighborIndices)
			{
				f.push_back(f_device_[neighborIndex]);
				g.push_back(g_device_[neighborIndex]);
			}


			
			//Filtering for data at index
			if (channelNum_f_ == 1 && channelNum_g_ == 1)
				FGMF_GPU_Or::cu_filterND<int2, float2, 1, 1>(sizeInfo_, NULL, radius2D_, fRange_, result_device_, f, g, (float2*)(dc));
			else if (channelNum_f_ == 1 && channelNum_g_ == 3)
				FGMF_GPU_Or::cu_filterND<int4, float4, 1, 3>(sizeInfo_, NULL, radius2D_, fRange_, result_device_, f, g, (float4*)(dc));
			else if (channelNum_f_ == 3 && channelNum_g_ == 1)
				FGMF_GPU_Or::cu_filterND<int2, float2, 3, 1>(sizeInfo_, NULL, radius2D_, fRange_, result_device_, f, g, (float2*)(dc));
			else if (channelNum_f_ == 3 && channelNum_g_ == 3)
				FGMF_GPU_Or::cu_filterND<int4, float4, 3, 3>(sizeInfo_, NULL, radius2D_, fRange_, result_device_, f, g, (float4*)(dc));

			results_[index] = Helper::UtilityForCUDA::downloadLinearArrayAsMat(result_device_, sizeInfo_);
			
			// Restore the data type
			if (results_[index].depth() != originalDepth_)
				results_[index].convertTo(results_[index], originalDepth_);
		}
		
		cudaFree(dc);
	}

	template<typename DC_TYPE>
	void WMF_ND<DC_TYPE>::calculateDC(int index, int pixelNumInWindow, DC_TYPE* dc)
	{
	}

	template<>
	void WMF_ND<float2>::calculateDC(int index, int pixelNumInWindow, float2* dc)
	{
		cu_calculateDC(sizeInfo_, NULL, g_device_[index]->host[0], pixelNumInWindow, epsilon_, dc, sumG_Manager_->sumG_total_);
	}

	template<>
	void WMF_ND<float4>::calculateDC(int index, int pixelNumInWindow, float4* dc)
	{
		cu_calculateDC3(sizeInfo_, NULL, g_device_[index], pixelNumInWindow, epsilon_, dc, sumG_Manager_->sumG_total_, sumG_Manager_->sumGG_total_);
	}




	template<typename DC_TYPE>
	void WMF_ND<DC_TYPE>::release_device_arrays_at_indices(std::vector<int>& indices)
	{
		for (const int index : indices) {
			if (index >= 0 && index < static_cast<int>(f_device_.size())) {
				delete f_device_[index];
				f_device_[index] = nullptr;
			}

			if (index >= 0 && index < static_cast<int>(g_device_.size())) {
				delete g_device_[index];
				g_device_[index] = nullptr;
			}
		}
	}


	template<typename DC_TYPE>
	void WMF_ND<DC_TYPE>::initialize_device_arrays_at_indices(const std::vector<int>& indices) {
		for (const int index : indices) {
			f_device_[index] = new Helper::DeviceArray<int>(f_imgs_[index], sizeInfo_);
			g_device_[index] = new Helper::DeviceArray<int>(g_imgs_[index], sizeInfo_);
		}
	}

}