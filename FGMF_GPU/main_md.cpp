#include "FGMF_GPU_Or.h"
#include <stdexcept>
#include <sstream>

using namespace std;

string numberToString(int i, int numOfDigit) {
    ostringstream ss;
    ss << setw(numOfDigit) << setfill('0') << i;
    return ss.str();
}

void displayParamter(const string& inputImagePath, const string& guideImagePath, cv::Mat& f_img, cv::Mat& g_img, int r, float epsilon, int bitDepth, int blockSize, int digitCount, int startNumber, int useDataCount, string extension, std::vector<int> radii3DAndUp, std::vector<int> size3DAndUp, bool negativeFlag) {
    string startNum = numberToString(startNumber, digitCount);
    string endNum = numberToString(startNumber + useDataCount - 1, digitCount);
    cout << "Processing image with the following parameters:" << endl;
    cout << "Input Image Path: " << inputImagePath << startNum << extension << " - " << endNum << extension << endl;
    cout << "Input Image Channel Num: " << f_img.channels() << endl;
    cout << "Guide Image Path: " << guideImagePath << startNum << extension << " - " << endNum << extension << endl;
    cout << "Guide Image Channel Num: " << g_img.channels() << endl;
    cout << "Radius: " << r << "x" << r;
    for (int i = 0; i < radii3DAndUp.size(); i++) {
		cout << "x" << radii3DAndUp[i];
	}
    cout << endl;
    cout << "Data Size: " << f_img.cols << "x" << f_img.rows;
    for (int i = 0; i < size3DAndUp.size(); i++) {
        cout << "x" << size3DAndUp[i];
    }
    cout << endl;
    cout << "Epsilon: " << epsilon << endl;
    cout << "Input Bit Depth: " << bitDepth << endl;
    cout << "CUDA Block Size: " << blockSize << endl;
    if (negativeFlag)
        cout << "If the input image contains negative values, it does not work." << endl;
    cout << endl;
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
 *   -i, --input <input_image_path>                (required) Input image path without the numeric sequence.
 *   -d, --digit-count <sequence_digit_count>      (optional) Minimum number of digits in a sequence, padded with zeros if necessary. Default: 3.
 *   -n, --start-number <sequence_start_number>    (optional) Starting number in the sequence. Default: 0.
 *   -u, --use-data-count <use_data_count>         (optional) Number of images to use. Default: 3.
 *   -x, --extension <image_extension>             (optional) Input and output image extension. Default: .png.
 *   -g, --guide <guide_image_path>                (optional) Guide image path without the numeric sequence. Default: same as input image path.
 *   -r, --radius <radius>                         (optional) Spatial radius (int). Default: 5.
 *   --radii-3d <radius_1,radius_2,...>            (optional) Comma-separated list of filter radii for 3rd dimension and beyond. Default: 0.
 *   --size-3d <size_1,size_2,...>                 (optional) Comma-separated list of sizes for 3rd dimension and beyond. Default: 1.
 *   -e, --root-epsilon <root_epsilon>             (optional) Root of epsilon (float). Default: 2.55.
 *   -s, --save <output_image_path>                (optional) Output image save path without the numeric sequence. Default: outputXXX.png in the same folder as the program.
 *   -b, --block-size <block_size>                 (optional) Block size for CUDA thread (int). Default: 16.
 */
int main(int argc, char** argv) {
    try {
        string inputImagePath;
        string guideImagePath = "";
        string outputImagePath = "output";
        string imageExtension = ".png";
        int r = 5;
        float rootEpsilon = 2.55f;
        int blockSize = 16;

        int sequenceDigitCount = 3;
        int sequenceStartNumber = 0;
        int useDataCount = 3;

        std::vector<int> radii3DAndUp = { 0 };  // for 3rd dimension and beyond
        std::vector<int> size3DAndUp = { 1 };  // for 3rd dimension and beyond



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
            else if (arg == "-s" || arg == "--save") {
                outputImagePath = argv[++i];
            }
            else if (arg == "-b" || arg == "--block-size") {
                blockSize = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-d" || arg == "--digit-count") {
                sequenceDigitCount = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-n" || arg == "--start-number") {
                sequenceStartNumber = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-u" || arg == "--use-data-count") {
                useDataCount = parseArgument<int>(argv[++i]);
            }
            else if (arg == "-x" || arg == "--extension") {
                imageExtension = argv[++i];
            }
            else if (arg == "--radii-3d") {
                radii3DAndUp.clear();
                string values = argv[++i];
                stringstream ss(values);
                string item;
                while (getline(ss, item, ',')) {
                    radii3DAndUp.push_back(stoi(item));
                }
            }
            else if (arg == "--size-3d") {
                size3DAndUp.clear();
                string values = argv[++i];
                stringstream ss(values);
                string item;
                while (getline(ss, item, ',')) {
                    size3DAndUp.push_back(stoi(item));
                }
            }
            else
            {
                throw invalid_argument("Unknown option: " + arg);
            }
        }


        // Check that radii3DAndUp and size3DAndUp have the same number of elements
        if (radii3DAndUp.size() != size3DAndUp.size()) {
            throw invalid_argument("The values of --radii-3d and --size-3d must have the same number of elements.");
        }


        int elementNumOf2D = 1;
        for (int dimension : size3DAndUp)
            elementNumOf2D *= dimension;
        if (useDataCount != elementNumOf2D)
        {
            throw invalid_argument("The number of images to be read must match the product of the values of --size-3d.");
        }


        if (inputImagePath.empty()) {
            throw invalid_argument("Input image path is required.");
        }

        if (guideImagePath.empty()) {
            guideImagePath = inputImagePath;
        }

        std::vector<cv::Mat> f_imgs;
        std::vector<cv::Mat> g_imgs;
        std::vector<cv::Mat> results;

        //Read inputImagePath + XXX + imageExtension. 
        //XXX is a sequential number.
        //sequenceDigitCount is the number of digits, sequenceStartNumber is the reading start number, and useDataCount is the number of data to be used.
        for (int i = sequenceStartNumber; i < sequenceStartNumber + useDataCount; i++) {
            string inputImagePathWithNumber = inputImagePath;
            string guideImagePathWithNumber = guideImagePath;
            string outputImagePathWithNumber = outputImagePath;
            string number = numberToString(i, sequenceDigitCount);
            inputImagePathWithNumber += number + imageExtension;
            guideImagePathWithNumber += number + imageExtension;
            outputImagePathWithNumber += number + imageExtension;
            // Read input image
            cv::Mat f_img = cv::imread(inputImagePathWithNumber, cv::IMREAD_UNCHANGED);
            if (f_img.empty()) {
                throw runtime_error("Failed to load input image: " + inputImagePathWithNumber);
            }
            // Read guide image
            cv::Mat g_img = cv::imread(guideImagePathWithNumber, cv::IMREAD_UNCHANGED);
            if (g_img.empty()) {
                throw runtime_error("Failed to load guide image: " + guideImagePathWithNumber);
            }
            f_imgs.push_back(f_img);
            g_imgs.push_back(g_img);
        }



        float epsilon = rootEpsilon * rootEpsilon;

        // Bit depth of input image
        int inputDepth = f_imgs[0].depth();
        int bitDepth;
        bool negativeFlag;

        if (inputDepth == 0) {
            bitDepth = 8;
            negativeFlag = false;
        }
        else if (inputDepth == 1) {
            bitDepth = 8;
            negativeFlag = true;
        }
        else if (inputDepth == 2) {
            bitDepth = 16;
            negativeFlag = false;
        }
        else if (inputDepth == 3) {
            bitDepth = 16;
            negativeFlag = true;
        }
        else if (inputDepth == 4) {
            bitDepth = 32;
            negativeFlag = true;
        }
        else {
            throw runtime_error("Invalid input image.");
        }
        int fRange = pow(2, bitDepth);

        // Display settings
        displayParamter(inputImagePath, guideImagePath, f_imgs[0], g_imgs[0], r, epsilon, bitDepth, blockSize, sequenceDigitCount, sequenceStartNumber, useDataCount, imageExtension, radii3DAndUp, size3DAndUp, negativeFlag);


        // Filtering for multi-dimensional data
        results = FGMF_GPU_Or::filter_Nd(f_imgs, g_imgs, r, radii3DAndUp, size3DAndUp, epsilon, fRange, blockSize);


        // Save the result
        for (int i = sequenceStartNumber, j = 0; i < sequenceStartNumber + useDataCount; i++, j++) {
            string outputImagePathWithNumber = outputImagePath;
            string number = numberToString(i, sequenceDigitCount);
            outputImagePathWithNumber += number + imageExtension;

            // Get the processed image
            cv::Mat result_img = results[j];

            // Save the processed image
            if (!cv::imwrite(outputImagePathWithNumber, result_img)) {
                throw runtime_error("Failed to save image: " + outputImagePathWithNumber);
            }
        }
        cout << "Output image saved successfully: " << outputImagePath << numberToString(sequenceStartNumber, sequenceDigitCount) << imageExtension << " - " << numberToString(sequenceStartNumber + useDataCount - 1, sequenceDigitCount) << imageExtension << endl;


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
