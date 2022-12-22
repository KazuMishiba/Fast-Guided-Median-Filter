//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include<iostream>
#include<string>
#include <fstream>
#include <stdexcept>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Method.h"


int main(int argc, char *argv[])
{
    /*
    SIZE_T  dwMin, dwMax;
    HANDLE hProcess;

    // プロセスIdの取得(カレントプロセス)
    DWORD dwProcessId = ::GetCurrentProcessId();

    // Retrieve a handle to the process.

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION,
        FALSE, dwProcessId);
    if (!hProcess)
    {
        printf("OpenProcess failed (%d)\n", GetLastError());
        return 1;
    }
   int ok = SetProcessWorkingSetSize
    (
        hProcess,
        200 * 1024 * 1,
        1380 * 1024
    );
   printf("success : %d\n", ok);

    // Retrieve the working set size of the process.

    if (!GetProcessWorkingSetSize(hProcess, &dwMin, &dwMax))
    {
        printf("GetProcessWorkingSetSize failed (%d)\n",
            GetLastError());
        return 1;
    }

    printf("Process ID: %d\n", dwProcessId);
    printf("Minimum working set: %lu KB\n", dwMin / 1024);
    printf("Maximum working set: %lu KB\n", dwMax / 1024);

    CloseHandle(hProcess);
    */
	try
	{

		Method method = Method();
		//method.main();
		//method.multiTest();

		return 0;
	}
	catch (...)
	{
		std::cout << "Error" << std::endl;
		return -1;
	}
}

