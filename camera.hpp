#pragma once

#include <thread>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace tbb;

class Camera {
    public:
        vector<concurrent_queue<Mat>*> frame_queue;

        Camera(vector<int> cam_ids);
        void init_and_start();

    private:
        vector<VideoCapture*> cam_caps;
        vector<int> camera_indexes;
        vector<thread*> camera_threads;

        void cam_thread(int idx);

};