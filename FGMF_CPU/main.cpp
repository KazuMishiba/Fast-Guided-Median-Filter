#include "FGMF_CPU_O1.h"
#include <stdexcept>
#include <sstream>

using namespace std;

void displayParamter(const string& inputImagePath, const string& guideImagePath, int r, float epsilon, int bitDepth, int numThreads) {
    cout << "Processing image with the following parameters:" << endl;
    cout << "Input Image Path: " << inputImagePath << endl;
    cout << "Guide Image Path: " << guideImagePath << endl;
    cout << "Radius: " << r << endl;
    cout << "Epsilon: " << epsilon << endl;
    cout << "Bit Depth: " << bitDepth << endl;
    cout << "Number of Threads: " << numThreads << endl;
}

template<typename T>
T parseArgument(const string& argument) {
    T value;
    stringstream ss(argument);
    ss >> value;
    if (ss.fail()) {
        throw invalid_argument("Invalid argument format: " + argument);
    }
    return value;
}
/**
 * @brief Entry point of the program that processes images based on the given command-line arguments.
 *
 * This program processes images using the provided command-line arguments. The required argument is the input image path.
 * Other optional arguments include guide image path, radius, root of epsilon, bit depth per channel, and number of threads.
 * If the guide image path is not provided, it is set to the same path as the input image. Default values are used for
 * other arguments if they are not specified.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return 0 if the program runs successfully, 1 if there's an error.
 *
 * Usage:
 *   -i, --input <input_image_path>                (required) Input image path.
 *   -g, --guide <guide_image_path>                (optional) Guide image path. Default: same as input image path.
 *   -r, --radius <radius>                         (optional) Window radius (int). Default: 5.
 *   -e, --root-epsilon <root_epsilon>             (optional) Root of epsilon (float). Default: 2.55.
 *   -b, --bit-depth <bit_depth>                   (optional) Bit depth per channel (int). Default: 8.
 *   -t, --threads <number_of_threads>             (optional) Number of threads (int). Default: number of available processors.
 *   -s, --save <output_image_path>                (optional) Output image save path. Default: output.png in the same folder as the program.
 */
int main(int argc, char** argv) {
    try {
        string inputImagePath;
        string guideImagePath = "";
        string outputImagePath = "output.png";
        int r = 5;
        float rootEpsilon = 2.55f;
        int bitDepth = 8;
        int numThreads = omp_get_num_procs();

        for (int i = 1; i < argc; i++) {
            string arg(argv[i]);
            if (arg == "-i" || arg == "--input") {
                inputImagePath = argv[++i];
            }
            else if (arg == "-g" || arg == "--guide") {
                guideImagePath = argv[++i];
            }
            else if (arg == "-r" || arg == "--radius") {
                r = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-e" || arg == "--root-epsilon") {
                rootEpsilon = parseArgument<float>(argv[++i]);
            }
            else if (arg == "-b" || arg == "--bit-depth") {
                bitDepth = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-t" || arg == "--threads") {
                numThreads = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-s" || arg == "--save") {
                outputImagePath = argv[++i];
            }
            else {
                throw invalid_argument("Unknown option: " + arg);
            }
        }

        if (inputImagePath.empty()) {
            throw invalid_argument("Input image path is required.");
        }

        if (guideImagePath.empty()) {
            guideImagePath = inputImagePath;
        }

        float epsilon = rootEpsilon * rootEpsilon;
        int fRange = pow(2, bitDepth);

        displayParamter(inputImagePath, guideImagePath, r, epsilon, bitDepth, numThreads);

        // Read input image
        cv::Mat f_img = cv::imread(inputImagePath, cv::IMREAD_UNCHANGED);
        if (f_img.empty()) {
            throw runtime_error("Failed to load input image: " + inputImagePath);
        }

        // Read guide image
        cv::Mat g_img = cv::imread(guideImagePath, cv::IMREAD_UNCHANGED);
        if (g_img.empty()) {
            throw runtime_error("Failed to load guide image: " + guideImagePath);
        }

        // Apply CPU-O(1) filter
        cv::Mat result = FGMF_CPU_O1::filter_2d(f_img, g_img, r, epsilon, fRange, numThreads);

        // Save the result
        if (!cv::imwrite(outputImagePath, result)) {
            throw runtime_error("Failed to save output image: " + outputImagePath);
        } else {
            cout << "Output image saved successfully: " << outputImagePath << endl;
        }

    }
    catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }
    return 0;
}
