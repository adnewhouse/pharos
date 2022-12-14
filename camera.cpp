// Based off example from https://putuyuwono.wordpress.com/2015/05/29/multi-thread-multi-camera-capture-using-opencv/

#include "camera.hpp"

Camera::Camera(vector<int> cam_ids) {
    camera_indexes = cam_ids;
}

void Camera::init_and_start() {
    VideoCapture *cap;
    thread *t;
    concurrent_queue<CameraFrame> *q;

    for (int i = 0; i < camera_indexes.size(); i++) {
        int idx = camera_indexes[i];  
        cap = new VideoCapture(idx);

        cap->set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap->set(CAP_PROP_FRAME_WIDTH, 640);
        cap->set(CAP_PROP_FRAME_HEIGHT, 480);
        cap->set(CAP_PROP_FPS, 60);

        if (!cap->isOpened()) {
            cerr << "Couldn't open video capture device" << endl;
        }
        cout << "Camera Setup: " << to_string(idx) << endl;

        //Put VideoCapture to the vector
        cam_caps.push_back(cap);
        
        //Make thread instance
        t = new thread(&Camera::cam_thread, this, i);
        
        //Put thread to the vector
        camera_threads.push_back(t);
        
        //Make a queue instance
        q = new concurrent_queue<CameraFrame>;
        
        //Put queue to the vector
        frame_queue.push_back(q);
    }
}

void Camera::cam_thread(int idx) {
    while(true) {
    	Mat frame;
        if((*cam_caps[idx]).read(frame)) {
            CameraFrame cf;
            cf.frame = frame;
            cf.frame_time = chrono::high_resolution_clock::now();
            cf.cam_idx = idx;
            if(frame_queue[idx]->empty())
                frame_queue[idx]->push(cf);
            frame.release(); //We no longer need access to this, decrement ref counter
	    }
    }
}
