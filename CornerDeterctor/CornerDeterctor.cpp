#include <opencv2/opencv.hpp>
#include "CornerDeterctor.h"

int main() 
{
    cv::VideoCapture gif("example.gif");

    if (!gif.isOpened()) 
    {
        std::cerr << "Failed to open the GIF file." << std::endl;
        return -1;
    }


    cv::Mat firstFrame;
    gif >> firstFrame;

    cv::Mat firstGray;
    cv::cvtColor(firstFrame, firstGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(firstGray, corners, 100, 0.01, 10);


    while (true) 
    {
        cv::Mat currentFrame;
        gif >> currentFrame;

        if (currentFrame.empty())
            break;

        

        // to grayscale
        cv::Mat currentGray;
        cv::cvtColor(currentFrame, currentGray, cv::COLOR_BGR2GRAY);

        // corner tracking using KLT
        std::vector<cv::Point2f> nextCorners;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(firstGray, currentGray, corners, nextCorners, status, err);

        // print corners
        for (size_t i = 0; i < corners.size(); ++i) 
            if (status[i]) 
                cv::circle(currentFrame, nextCorners[i], 2, cv::Scalar(0, 255, 0), -1);

        // adding new corners
        std::vector<cv::Point2f> newCorners;
        cv::goodFeaturesToTrack(currentGray, newCorners, 100, 0.01, 10);

        // print new corners
        for (const auto& newCorner : newCorners) 
        {
            cv::circle(currentFrame, newCorner, 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("KLT Tracking", currentFrame);

        // Exit by 'Esc
        if (cv::waitKey(30) == 27) 
            break;

        // reusing corners to the next frame
        corners = newCorners;
        firstGray = currentGray.clone();
    }

    cv::destroyAllWindows();
    return 0;
}