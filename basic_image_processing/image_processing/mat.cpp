#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    // The idea is that each Mat object has its own header, 
    // however a matrix may be shared between two Mat objects by having their matrix pointers point to the same address. 
    // Moreover, the copy operators will only copy the headers and the pointer to the large matrix, not the data itself.
    Mat A = imread("../../data/lenna.jpg");
    Mat B(A); //use copy constructor
    Mat C;
    C=A;  //assignment operator
    // cout<<"address of matrix A is: ">>A.at<Vec3b>(0,0)<<endl;
    // cout<<"address of matrix B is: ">>B[0][0]<<endl;
    // cout<<"address of matrix C is: ">>C[0][0]<<endl;
    if(A.empty()){
        cout<<"couln't open or find the image"<<endl;
    }

    //the different objects just provide different access methods to the same underlying data
    // roi
    Mat D(A, Rect(10,10,100,100));
    Mat E=A(Range(0,100), Range::all());

    //the last Mat object no longer needed matrix itself then it will be cleaned up, this is handled by using reference counting mechanism

    //copy matrix itself
    Mat F = A.clone();
    Mat G;
    A.copyTo(G);

    //display picture
    // string windowName = "test";
    // namedWindow(windowName);
    // imshow(windowName, G);

    // waitKey(0);
    // destroyWindow(windowName);

    //creating a Mat object explicitly
    //cv::Mat::Mat constructor
    Mat M(2,2,CV_8UC3,Scalar(0,0,255));
    cout<<M<<" "<<M.size()<<endl;
    // cout<<E<<endl;

    //use c/c++ arrays and initialize via constructor
    int sz[3] = {2,2,2};
    Mat L(3,sz,CV_8UC(1), Scalar::all(0));
    cout<<L.size()<<endl;
    // cout<<L<<endl;
    // cout<<L<<" "<<L.size()<<endl;

    //cv::Mat::create, cannot initialize matrix value using this construction
    M.create(4,4,CV_8UC(2));
    cout<<"M ="<<endl<<" "<<M<<endl;

    //cv::Mat::zeros, cv::Mat::ones, cv::Mat::eye
    Mat H = Mat::eye(4,4,CV_64F);
    cout<<"H= "<<endl<<" "<<H<<endl;

    Mat O = Mat::ones(2,2,CV_32F);
    cout<<"O= "<<endl<<" "<<O<<endl;

    Mat Z = Mat::zeros(3,3,CV_8UC1);
    cout<<"Z= "<<endl<<" "<<Z<<endl;

    //when creating 
    Mat I = (Mat_<double>(3,3) << -1,0,1,2,3,4,5,6,7,8);
    cout<<"I = "<<endl<<" "<<I<<endl;

    Mat J = (Mat_<double> ({-2,0,2,4,6,8}));
    cout<<"J = "<<endl<<" "<<J<<endl;

    // J.reshape(3,3);
    // cout<<"J = "<<endl<<" "<<J<<endl;

    //clone row of matrix
    Mat RowClone = I.row(2).clone();
    cout<<"RowClone = "<<endl<<" "<<RowClone<<endl;

    //fill out matrix with random values using cv::randu() function
    Mat R = Mat(3,2,CV_8UC3,Scalar::all(25));
    cout<<"R =  "<<endl<<" "<<R<<endl;
    randu(R,Scalar::all(0), Scalar::all(255));

    //output formatting
    //python 
    cout<<"R(python) = "<<endl<<format(R,Formatter::FMT_PYTHON) <<endl;

    //csv
    cout<<"R(csv) = "<<endl<<format(R,Formatter::FMT_CSV) <<endl;

    //numpy
    cout<<"R(numpy) = "<<endl<<format(R,Formatter::FMT_NUMPY) <<endl;
    //c
    cout<<"R(c) = "<<endl<<format(R,Formatter::FMT_C) <<endl;

    // cout<<"R =  "<<endl<<" "<<R<<endl;
    //data structure of Scalar
    cout<<Scalar::all(0)<<endl;


    //output of other common items
    //2d point
    Point2f P1(5,1);
    cout<<"Point(2D)= "<<P1<<endl;

    //3d point
    Point3f P2(2,6,7);
    cout<<"Point(3D)= "<<P2<<endl;

    //vector via cv::Mat
    vector<float> v={1,2,3,4,5};
    // cout<<v.size()<<endl;
    cout<<"vector of floats via Mat= "<<Mat(v)<<endl;

    //vector of points
    vector<Point2f> vPoints(20);
    for(size_t i = 0; i <vPoints.size(); i++){
        vPoints[i] = Point2f(i*5, i%7);
    }
    cout<<"A vector of 2d points = "<<vPoints<<endl;

    vector<int> test={1,23,4,5,67};
    // cout<<test<<endl;

    return 0;
}