#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    //load an image file and display it
    // Mat image = imread("/Users/hadley/Development/computer_vision/data/lenna.jpg");
    
    //create a new image which consists of 3 channels, depth of 8 bits
    //800 x 600 of resolution
    // Mat image(600, 800, CV_8UC3, Scalar(100,250,30));

    //create a grey image 
    Mat image(600, 800, CV_8UC1, 100);

    if(image.empty()){
        cout<<"couln't open or find the image"<<endl;
    }

    string windowName = "test";
    namedWindow(windowName);
    imshow(windowName, image);


    waitKey(0);
    destroyWindow(windowName);

    return 0;
}
