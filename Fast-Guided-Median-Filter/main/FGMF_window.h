#pragma once
#include "FGMF_base.h"

/*
Windowベースの定義
*/


////////////////////////////////
//多次元用
struct Pos
{
	int center;//処理対象中心位置
	int add;//追加される位置
	int rem;//削除される位置
};
inline void setPos(Pos& x, const int& radius) {
	x.add = 0;
	x.rem = -2 * radius - 1;
	x.center = -radius;
}
inline void setPosAtDim0(Pos& x, const int& radius, const int& dim0Start) {
	x.add = dim0Start + radius;
	x.rem = dim0Start - radius - 1;
	x.center = dim0Start;
}
struct DimStatus
{
	int hasAdd;
	int hasRem;
	int hasAddOnly;
	int hasRemOnly;
	int isInside_image;
};
inline void setStatusAtDim0(const Pos& x, const int& size, const int& remStartPos, const int& insidePos, const int& isInsideAtUpperDim, DimStatus& status) {
	status.hasAdd = x.add < size;
	status.hasRem = x.rem >= remStartPos;
	status.hasAddOnly = !status.hasRem && status.hasAdd;
	status.hasRemOnly = status.hasRem && !status.hasAdd;
	status.isInside_image = isInsideAtUpperDim && x.center >= insidePos;
}
inline void setStatus(const Pos& x, const int& size, const int& isInsideAtUpperDim, DimStatus& status) {
	status.hasAdd = x.add < size;
	status.hasRem = x.rem >= 0;
	status.hasAddOnly = !status.hasRem && status.hasAdd;
	status.hasRemOnly = status.hasRem && !status.hasAdd;
	status.isInside_image = isInsideAtUpperDim && x.center >= 0;
}
inline void setStatusAtOutermostLoop(const Pos& x, const int& size, DimStatus& status) {
	setStatus(x, size, 1, status);
}
//GPU用 sumのみ追加・削除・両方、またはcxdxも計算
struct DimStatusForGPU
{
	int hasAdd;
	int hasRem;
	int hasAddOnly;
	int hasRemOnly;
	int isInside_image;//cx,dx計算
};
//GPU用次のステータス計算
inline void setNextStatus(const Pos& x, const int& size, const int& isInsideAtUpperDim, DimStatus& status) {
	status.hasAdd = x.add + 1 < size;
	status.hasRem = x.rem + 1 >= 0;
	status.hasAddOnly = !status.hasRem && status.hasAdd;
	status.hasRemOnly = status.hasRem && !status.hasAdd;
	status.isInside_image = isInsideAtUpperDim && (x.center + 1) >= 0 && (x.center + 1) < size;
}
inline int calculatePixelNumAtDim(const Pos& x, const int& size, const int& r, const DimStatus& status, const int& pixelNumAtUpperDim) {
	if (status.hasAddOnly)
		return (x.center + r + 1) * pixelNumAtUpperDim;
	else if (status.hasRemOnly)
		return (size - x.center + r) * pixelNumAtUpperDim;
	else
		return (2 * r + 1) * pixelNumAtUpperDim;
}
inline int calculatePixelNumAtOutermostLoop(const Pos& x, const int& size, const int& r, const DimStatus& status) {
	return calculatePixelNumAtDim(x, size, r, status, 1);
}
inline void calculatePixelNumAtIntermostLoop(int& pixel_sum_window, float& pixel_sum_window_inv, const int& pixelNumAtUpperDim, const DimStatus& status) {
	if (status.hasAddOnly)//追加のみ
		addPixelSum(pixelNumAtUpperDim, pixel_sum_window, pixel_sum_window_inv);
	else if (status.hasRemOnly)//削除のみ
		subtractPixelSum(pixelNumAtUpperDim, pixel_sum_window, pixel_sum_window_inv);
}

/////////////////////////
//Window_single
/////////////////////////
template<typename GSum, typename FGSumUpToIndex, typename FG>
class Window_single
{
public:
	Window_single();
	Window_single(const int& Imax);
	~Window_single();
	void initialize();

	GSum gsum;
	FGSumUpToIndex sumUpToIndex;
	FG* histo;

	int Imax;

private:

};

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline Window_single<GSum, FGSumUpToIndex, FG>::Window_single()
{
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_single<GSum, FGSumUpToIndex, FG>::Window_single(const int& Imax)
{
	this->gsum;
	this->sumUpToIndex;
	this->histo = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
	this->Imax = Imax;
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_single<GSum, FGSumUpToIndex, FG>::~Window_single()
{
	_aligned_free(this->histo);
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
void Window_single<GSum, FGSumUpToIndex, FG>::initialize()
{
	//初期化
	memset(&this->gsum, 0, sizeof(GSum));
	memset(&this->sumUpToIndex, 0, sizeof(FGSumUpToIndex));
	memset(this->histo, 0, sizeof(FG) * this->Imax);
}

//doubly-linked list 用
template<typename GSum, typename FGSumUpToIndex, typename FG>
class Window_single_list
{
public:
	Window_single_list();
	Window_single_list(const int& Imax);
	~Window_single_list();
	void initialize();

	GSum gsum;
	FGSumUpToIndex sumUpToIndex;
	std::list<FG> histo;

	int Imax;

private:

};

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline Window_single_list<GSum, FGSumUpToIndex, FG>::Window_single_list()
{
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_single_list<GSum, FGSumUpToIndex, FG>::Window_single_list(const int& Imax)
{
	this->gsum;
	this->sumUpToIndex;
	this->Imax = Imax;
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_single_list<GSum, FGSumUpToIndex, FG>::~Window_single_list()
{
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
void Window_single_list<GSum, FGSumUpToIndex, FG>::initialize()
{
	//初期化
	memset(&this->gsum, 0, sizeof(GSum));
	memset(&this->sumUpToIndex, 0, sizeof(FGSumUpToIndex));
	this->histo.clear();
}


/////////////////////////
//Window_vector
/////////////////////////
template<typename GSum, typename FGSumUpToIndex, typename FG>
class Window_vector
{
public:
	Window_vector(const int& Imax, const std::vector<int>& size, int level, bool storeGSum);
	Window_vector();
	~Window_vector();
	void resetPos();
	void setZero();

	GSum* gsum;
	FGSumUpToIndex* sumUpToIndex;
	FG** histo;

	int length;
	bool storeGSum;
	int histoMemoryLength;
	//メモリ位置
	int add;
	int rem;

private:

};


template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_vector<GSum, FGSumUpToIndex, FG>::Window_vector(const int& Imax, const std::vector<int>& size, int level, bool storeGSum)
{
	this->storeGSum = storeGSum;
	this->length = 1;//データ量
	this->histoMemoryLength = sizeof(FG) * Imax;
	for (int i = 0; i < size.size() - level; i++)
	{
		this->length *= size[i];
	}
	if (this->storeGSum)
	{
		this->gsum = new GSum[this->length];
		memset(this->gsum, 0, sizeof(GSum) * length);
	}
	this->sumUpToIndex = new FGSumUpToIndex[this->length];
	memset(this->sumUpToIndex, 0, sizeof(FGSumUpToIndex) * length);
	this->histo = new FG *[this->length];
	for (int i = 0; i < this->length; i++)
	{
		this->histo[i] = (FG*)_aligned_malloc(this->histoMemoryLength, MEMORY_ALIGNMENT);
		memset(this->histo[i], 0, this->histoMemoryLength);
	}
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline Window_vector<GSum, FGSumUpToIndex, FG>::Window_vector()
{
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
void Window_vector<GSum, FGSumUpToIndex, FG>::resetPos() {
	//位置リセット
	this->add = 0;
	this->rem = 0;
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void Window_vector<GSum, FGSumUpToIndex, FG>::setZero()
{
	if (this->storeGSum)
	{
		memset(this->gsum, 0, sizeof(GSum) * length);
	}
	memset(this->sumUpToIndex, 0, sizeof(FGSumUpToIndex) * length);
	for (int i = 0; i < this->length; i++)
	{
		memset(this->histo[i], 0, this->histoMemoryLength);
	}
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_vector<GSum, FGSumUpToIndex, FG>::~Window_vector()
{
	if (this->storeGSum)
	{
		delete[] this->gsum;
	}
	delete[] this->sumUpToIndex;
	for (int i = 0; i < this->length; i++)
	{
		_aligned_free(this->histo[i]);
	}
	delete[] this->histo;
}

//doubly-linked list 用
/*
template<typename GSum, typename FGSumUpToIndex, typename FG>
class Window_vector_list
{
public:
	Window_vector_list(const int& Imax, const std::vector<int>& size, int level, bool storeGSum);
	Window_vector_list();
	~Window_vector_list();
	void resetPos();
	void setZero();

	GSum* gsum;
	FGSumUpToIndex* sumUpToIndex;
	std::list<FG>* histo;

	int length;
	bool storeGSum;
	//メモリ位置
	int add;
	int rem;

private:

};


template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_vector_list<GSum, FGSumUpToIndex, FG>::Window_vector_list(const int& Imax, const std::vector<int>& size, int level, bool storeGSum)
{
	this->storeGSum = storeGSum;
	this->length = 1;//データ量
	this->histoMemoryLength = sizeof(FG) * Imax;
	for (int i = 0; i < size.size() - level; i++)
	{
		this->length *= size[i];
	}
	if (this->storeGSum)
	{
		this->gsum = new GSum[this->length];
		memset(this->gsum, 0, sizeof(GSum) * length);
	}
	this->sumUpToIndex = new FGSumUpToIndex[this->length];
	memset(this->sumUpToIndex, 0, sizeof(FGSumUpToIndex) * length);
	this->histo = new std::list<FG> [this->length];
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline Window_vector_list<GSum, FGSumUpToIndex, FG>::Window_vector_list()
{
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
void Window_vector_list<GSum, FGSumUpToIndex, FG>::resetPos() {
	//位置リセット
	this->add = 0;
	this->rem = 0;
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void Window_vector_list<GSum, FGSumUpToIndex, FG>::setZero()
{
	if (this->storeGSum)
	{
		memset(this->gsum, 0, sizeof(GSum) * length);
	}
	memset(this->sumUpToIndex, 0, sizeof(FGSumUpToIndex) * length);
	for (int i = 0; i < this->length; i++)
	{
		this->histo[i].clear();
	}
}

template<typename GSum, typename FGSumUpToIndex, typename FG>
Window_vector_list<GSum, FGSumUpToIndex, FG>::~Window_vector_list()
{
	if (this->storeGSum)
	{
		delete[] this->gsum;
	}
	delete[] this->sumUpToIndex;
	delete[] this->histo;
}
*/



/////////////////////
//addPixelToWindow
//pixelのポインタ増加も含める
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void addPixelToWindow(Window_vector<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	addPixelToWindow(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g);
	pixel_f++;
	pixel_g++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void addPixelToWindow_gSum(Window_vector<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	addPixelToWindow_gSum(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g, window.gsum[pos]);
	pixel_f++;
	pixel_g++;
}
/*
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void addPixelToWindow_list(Window_vector_list<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	addPixelToWindow(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g);
	pixel_f++;
	pixel_g++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void addPixelToWindow_gSum_list(Window_vector_list<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	addPixelToWindow_gSum(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g, window.gsum[pos]);
	pixel_f++;
	pixel_g++;
}
*/

//removePixelFromWindow
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removePixelFromWindow(Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& pos, const int& pixel_f, const GTYPE& pixel_g) {
	removePixelFromWindow(window.histo[pos], window.sumUpToIndex[pos], pixel_f, pixel_g);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removePixelFromWindow_gSum(Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& pos, const int& pixel_f, const GTYPE& pixel_g) {
	removePixelFromWindow_gSum(window.histo[pos], window.sumUpToIndex[pos], pixel_f, pixel_g, window.gsum[pos]);
}
//よりまとめた版
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removePixelFromWindow(Window_vector<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	removePixelFromWindow(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g);
	pixel_f++;
	pixel_g++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removePixelFromWindow_gSum(Window_vector<GSum, FGSumUpToIndex, FG>& window, int*& pixel_f, GTYPE*& pixel_g) {
	const int pos = window.add;
	removePixelFromWindow_gSum(window.histo[pos], window.sumUpToIndex[pos], *pixel_f, *pixel_g, window.gsum[pos]);
	pixel_f++;
	pixel_g++;
}
//updateSubWindowAndAddToWindow
//vec to vec
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void updateSubWindowAndAddToWindow(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& winpos, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, const int& subpos) {
	updateSubWindowAndAddToWindow(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos]);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void updateSubWindowAndAddToWindow_gsum(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& winpos, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, const int& subpos) {
	updateSubWindowAndAddToWindow_gSum(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos], window.gsum[winpos], subwin.gsum[subpos]);
}
//よりまとめた版
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int winpos = window.add;
	const int subpos = subwin.add;
	updateSubWindowAndAddToWindow(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos]);
	subwin.add++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int winpos = window.add;
	const int subpos = subwin.add;
	updateSubWindowAndAddToWindow_gSum(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos], window.gsum[winpos], subwin.gsum[subpos]);
	subwin.add++;
}

//vec to single
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, int subpos) {
	updateSubWindowAndAddToWindow(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos]);
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, int subpos) {
	updateSubWindowAndAddToWindow_gSum(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos], window.gsum, subwin.gsum[subpos]);
}
//よりまとめた版
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int subpos = subwin.add;
	updateSubWindowAndAddToWindow(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos]);
	subwin.add++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int subpos = subwin.add;
	updateSubWindowAndAddToWindow_gSum(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos], window.gsum, subwin.gsum[subpos]);
	subwin.add++;
}

//removeSubWindowFromWindow
//vec to single
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, int subpos) {
	removeSubWindowFromWindow(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos]);
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow_gSum(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, int subpos) {
	removeSubWindowFromWindow_gSum(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos], window.gsum, subwin.gsum[subpos]);
}
//よりまとめた版
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int subpos = subwin.rem;
	removeSubWindowFromWindow(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos]);
	subwin.rem++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow_gSum(const int& Imax, Window_single<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int subpos = subwin.rem;
	removeSubWindowFromWindow_gSum(Imax, window.histo, subwin.histo[subpos], window.sumUpToIndex, subwin.sumUpToIndex[subpos], window.gsum, subwin.gsum[subpos]);
	subwin.rem++;
}
//vec to vec
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removeSubWindowFromWindow(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& winpos, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, const int& subpos) {
	removeSubWindowFromWindow(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos]);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE>
inline void removeSubWindowFromWindow_gsum(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, const int& winpos, Window_vector<GSum, FGSumUpToIndex, FG>& subwin, const int& subpos) {
	removeSubWindowFromWindow_gSum(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos], window.gsum[winpos], subwin.gsum[subpos]);
}
//よりまとめた版
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int winpos = window.add;
	const int subpos = subwin.rem;
	removeSubWindowFromWindow(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos]);
	subwin.rem++;
}
template<typename GSum, typename FGSumUpToIndex, typename FG>
inline void removeSubWindowFromWindow_gSum(const int& Imax, Window_vector<GSum, FGSumUpToIndex, FG>& window, Window_vector<GSum, FGSumUpToIndex, FG>& subwin) {
	const int winpos = window.add;
	const int subpos = subwin.rem;
	removeSubWindowFromWindow_gSum(Imax, window.histo[winpos], subwin.histo[subpos], window.sumUpToIndex[winpos], subwin.sumUpToIndex[subpos], window.gsum[winpos], subwin.gsum[subpos]);
	subwin.rem++;
}



//

template<typename GSum, typename FGSumUpToIndex, typename FG, typename CTYPE>
//中央値計算
inline void findMedian(const CTYPE& cx, const float& dx, const float& half, Window_single<GSum, FGSumUpToIndex, FG>& window, int& result_center) {
	findMedian(cx, dx, half, window.histo, window.sumUpToIndex, result_center);
}


