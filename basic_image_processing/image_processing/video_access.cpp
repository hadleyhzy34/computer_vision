#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
 //Open the default video camera
 // VideoCapture cap(0);


 //open the video file from folder, video capture class constructor
 VideoCapture cap("../../data/oldFriend.mp4");

 // if not success, exit program
 if (cap.isOpened() == false)  
 {
  cout << "Cannot open the video camera" << endl;
  cin.get(); //wait for any key press
  return -1;
 }

 double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
 double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

 cout <<"resolution of the video: "<<dWidth<<" x "<<dHeight<<endl;

  //get the frames rate of the video
 double fps = cap.get(CAP_PROP_FPS); 
 cout << "Frames per seconds : " << fps << endl;

 string window_name = "My Camera Feed";
 namedWindow(window_name); //create a window called "My Camera Feed"
 
 while (true)
 {
  Mat frame;
  //read frames from the video file one by one and store it in the frame
  bool bSuccess = cap.read(frame); // read a new frame from video 

  //Breaking the while loop if the frames cannot be captured
  if (bSuccess == false) 
  {
   cout << "Video camera is disconnected" << endl;
   cin.get(); //Wait for any key press
   break;
  }

  //show the frame in the created window
  imshow(window_name, frame);

  //wait for for 10 ms until any key is pressed.  
  //If the 'Esc' key is pressed, break the while loop.
  //If the any other key is pressed, continue the loop 
  //If any key is not pressed withing 10 ms, continue the loop 
  if (waitKey(10) == 27)
  {
   cout << "Esc key is pressed by user. Stoppig the video" << endl;
   break;
  }
 }

 return 0;
}

