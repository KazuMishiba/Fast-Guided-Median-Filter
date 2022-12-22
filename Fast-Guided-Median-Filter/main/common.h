#pragma once


//#define HOME
//#define HPC

#define BLOCK_SIZE_1D 8

#ifdef _DEBUG
#define BLOCK_SIZE_2D 8
#else
#define BLOCK_SIZE_2D 32
#endif

//#define USE_AVX512
#define USE_AVX2

#if defined(USE_AVX512)
#define MEMORY_ALIGNMENT 64
#else
#define MEMORY_ALIGNMENT 32
#endif
