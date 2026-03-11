#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

static int oddAtLeast(int v, int minv) {
    v = std::max(v, minv);
    return (v % 2) ? v : v + 1;
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static cv::Mat denoiseImage(const cv::Mat& gray, const std::string& type, int k, double sigma) {
    cv::Mat out;
    int kOdd = oddAtLeast(k, 3);

    if (type == "none") {
        return gray.clone();
    } else if (type == "median") {
        cv::medianBlur(gray, out, kOdd);
    } else if (type == "bilateral") {
        int d = std::max(1, k); // diameter
        cv::bilateralFilter(gray, out, d, sigma, sigma);
    } else { // gaussian (default)
        cv::GaussianBlur(gray, out, cv::Size(kOdd, kOdd), sigma, sigma);
    }
    return out;
}

static cv::Mat sobelGradient8U(const cv::Mat& input, const std::string& mode, int ksize) {
    if (!(ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7)) {
        ksize = 3;
    }

    cv::Mat gx, gy, g32, g8;
    cv::Sobel(input, gx, CV_32F, 1, 0, ksize);
    cv::Sobel(input, gy, CV_32F, 0, 1, ksize);

    if (mode == "x") {
        cv::absdiff(gx, cv::Scalar::all(0), g32);
    } else if (mode == "y") {
        cv::absdiff(gy, cv::Scalar::all(0), g32);
    } else { // magnitude (default)
        cv::magnitude(gx, gy, g32);
    }

    cv::normalize(g32, g8, 0, 255, cv::NORM_MINMAX, CV_8U);
    return g8;
}

static cv::Mat thresholdEdges(const cv::Mat& grad8,
                              const std::string& tmode,
                              double th,
                              bool invert,
                              int block,
                              double C) {
    cv::Mat edges;
    int type = invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;

    if (tmode == "otsu") {
        cv::threshold(grad8, edges, 0, 255, type | cv::THRESH_OTSU);
    } else if (tmode == "adaptive") {
        cv::adaptiveThreshold(
            grad8, edges, 255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            type,
            oddAtLeast(block, 3),
            C
        );
    } else { // fixed (default)
        cv::threshold(grad8, edges, th, 255, type);
    }

    return edges;
}

static void saveStep(const std::string& prefix, int idx, const std::string& name, const cv::Mat& img) {
    cv::imwrite(cv::format("%s_%02d_%s.png", prefix.c_str(), idx, name.c_str()), img);
}

int main(int argc, char** argv) {
    const char* keys =
        "{help h||Show help}"
        "{@input|street_scene.png|Input image path}"
        "{out|out|Output prefix}"
        "{denoise|gaussian|none, gaussian, median, bilateral}"
        "{k|5|Kernel size (odd for gaussian/median; diameter for bilateral)}"
        "{sigma|1.2|Sigma for gaussian or bilateral}"
        "{sobel|3|Sobel kernel size: 1,3,5,7}"
        "{smode|magnitude|Sobel output: magnitude, x, y}"
        "{tmode|fixed|Threshold mode: fixed, otsu, adaptive}"
        "{th|80|Fixed threshold value}"
        "{block|11|Adaptive block size (odd)}"
        "{c|2|Adaptive C}"
        "{invert|0|Invert threshold: 0 or 1}";

    cv::CommandLineParser p(argc, argv, keys);

    if (p.has("help")) {
        p.printMessage();
        return 0;
    }
    if (!p.check()) {
        p.printErrors();
        return 1;
    }

    std::string input  = p.get<std::string>(0);
    std::string out    = p.get<std::string>("out");
    std::string dType  = lower(p.get<std::string>("denoise"));
    std::string sMode  = lower(p.get<std::string>("smode"));
    std::string tMode  = lower(p.get<std::string>("tmode"));
    int k              = p.get<int>("k");
    double sigma       = p.get<double>("sigma");
    int sobelK         = p.get<int>("sobel");
    double th          = p.get<double>("th");
    int block          = p.get<int>("block");
    double C           = p.get<double>("c");
    bool invert        = (p.get<int>("invert") != 0);

    cv::Mat color = cv::imread(input, cv::IMREAD_COLOR);
    if (color.empty()) {
        std::cerr << "Error: cannot open image: " << input << "\n";
        return 1;
    }

    // 1) Grayscale
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    // 2) Low-pass denoising
    cv::Mat denoised = denoiseImage(gray, dType, k, sigma);

    // 3) Gradient (Sobel)
    cv::Mat gradient = sobelGradient8U(denoised, sMode, sobelK);

    // 4) Gradient thresholding
    cv::Mat edges = thresholdEdges(gradient, tMode, th, invert, block, C);

    saveStep(out, 1, "gray", gray);
    saveStep(out, 2, "denoised", denoised);
    saveStep(out, 3, "gradient", gradient);
    saveStep(out, 4, "edges", edges);

    std::cout << "Saved:\n"
              << "  " << out << "_01_gray.png\n"
              << "  " << out << "_02_denoised.png\n"
              << "  " << out << "_03_gradient.png\n"
              << "  " << out << "_04_edges.png\n";

    return 0;
}