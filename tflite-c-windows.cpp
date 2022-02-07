// tflite-c-windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <fstream>
#include "ObjectDetector.h"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat src = imread("C:/c_vscode/tflite-c-windows/test.jpg");

    int length;
    char* buffer;

    ifstream is;
    is.open("C:/c_vscode/tflite-c-windows/model&data/ssd_mobilenet_small_model.tflite", ios::binary);
    // get length of file:
    is.seekg(0, ios::end);
    length = is.tellg();
    is.seekg(0, ios::beg);
    // allocate memory:
    buffer = new char[length];
    // read data as a block:
    is.read(buffer, length);
    is.close();

    ObjectDetector detector = ObjectDetector(buffer, length, true);
    DetectResult* res = detector.detect(src);
    for (int i = 0; i < detector.DETECT_NUM; ++i) {
        int label = res[i].label;
        float score = res[i].score;
        float xmin = res[i].xmin;
        float xmax = res[i].xmax;
        float ymin = res[i].ymin;
        float ymax = res[i].ymax;
    
        rectangle(src, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0),2);
        putText(src, to_string(label), Point(xmin, ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 255, 0), 2);

        cout <<label << " " << score << endl;

    }
    imshow("test", src);
    waitKey(0);

    cout << "execuation completed!\n";
}
