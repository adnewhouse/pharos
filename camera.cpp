#include "camera.hpp"

Camera::Camera(vector<int> cam_ids) {
    camera_indexes = cam_ids;
}

void Camera::init_and_start() {
    VideoCapture *cap;
    thread *t;
    concurrent_queue<Mat> *q;

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
        q = new concurrent_queue<Mat>;
        
        //Put queue to the vector
        frame_queue.push_back(q);
    }
}

void Camera::cam_thread(int idx) {
    while(true) {
    	Mat frame;
        if((*cam_caps[idx]).read(frame)) {
            frame_queue[idx]->push(frame);
            frame.release(); //We no longer need access to this, decrement ref counter
	}
    }
}
