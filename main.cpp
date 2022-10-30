#include <opencv2/opencv.hpp>
#include <thread>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/apriltag_pose.h"
}

#include "camera.hpp"

using namespace cv;
using namespace std;

Mat camera_mat = (Mat_<double>(3,3) << 1.0726025045989093e+03, 0., 6.2896842131682581e+02,
                  0, 1.0726025045989093e+03, 4.3324822128294642e+02,
                  0, 0, 1.0);


Mat dist_mat = (Mat_<double>(1,5) << 6.3546507304579905e-02, -9.2152649371735132e-02, 0., 0., 3.9583498557302213e-02);

void drawAxis(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray _rvec, InputArray _tvec, float length) {

    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert(length > 0);

    // project axis points
    vector< Point3f > axisPoints;
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(length, 0, 0));
    axisPoints.push_back(Point3f(0, length, 0));
    axisPoints.push_back(Point3f(0, 0, length));
    vector< Point2f > imagePoints;
    projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    // draw axis lines
    line(_image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 3);
    line(_image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
    line(_image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 3);
}

void process_thread(Camera& cam, int idx, concurrent_queue<CameraPose*>& pose_queue) {
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 2;
    td->nthreads = 2;

    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;

    CameraFrame cf;
    Mat img, gray;
    auto prev = chrono::high_resolution_clock::now();

    while(1) {
        //Pop frame from queue and check if the frame is valid
        if (cam.frame_queue[idx]->try_pop(cf)) {
            img = cf.frame;
            cvtColor(img, gray, COLOR_BGR2GRAY);

            image_u8_t im = { .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };

            zarray_t *detections = apriltag_detector_detect(td, &im);
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);

                vector<Point2d> uncalib_pts;
                for(int i =0; i < 4; i++) {
                    uncalib_pts.push_back(Point2d(det->p[i][0], det->p[i][1]));
                }

                vector<Point3d> objPts;
                objPts.push_back(Point3d(-0.0762, 0.0762, 0.0));
                objPts.push_back(Point3d(0.0762, 0.0762, 0.0));
                objPts.push_back(Point3d(0.0762, -0.0762, 0.0));
                objPts.push_back(Point3d(-0.0762, -0.0762, 0.0));
                Mat rvec, tvec;
                solvePnP(objPts, uncalib_pts, camera_mat, dist_mat, rvec, tvec, false, SOLVEPNP_IPPE_SQUARE);
                drawAxis(img, camera_mat, dist_mat, rvec, tvec, 0.1);
                line(img, uncalib_pts[0], uncalib_pts[1], cv::Scalar(0,0,255), 2);
                line(img, uncalib_pts[1], uncalib_pts[2], cv::Scalar(0,255,0), 2);
                line(img, uncalib_pts[2], uncalib_pts[3], cv::Scalar(255,0,0), 2);
                line(img, uncalib_pts[3], uncalib_pts[0], cv::Scalar(0,0,0), 2);
                cout << rvec << endl;
                cout << tvec << endl << endl;
            }
            apriltag_detections_destroy(detections);

            auto now = chrono::high_resolution_clock::now();
            auto elapsed = now - prev;
            prev = now;
            int f_time = chrono::duration_cast<chrono::microseconds>(elapsed).count();

            cv::putText(img, to_string(1000000/f_time), Point(40, 40),
                fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

            cv::imshow("Disp " + to_string(idx), img);
            pollKey();
        }
    }

}

int main() {
   
   cout << "C = " << endl << " " << camera_mat << endl << endl;
   cout << "D = " << endl << " " << dist_mat << endl << endl;

    vector<int> cam_ids = { 0, 2 };
    Camera cam(cam_ids);
    cam.init_and_start();

    vector<thread*> worker_threads;
    concurrent_queue<CameraPose*> pose_queue;
    thread* t;
    for (int i = 0; i < cam_ids.size(); i++) {
        t = new thread(&process_thread, ref(cam), i, ref(pose_queue));
        worker_threads.push_back(t);
    }

    while(1) {
        CameraPose* cf;
        if (pose_queue.try_pop(cf) && cf->tag_id == 1) {
            auto now = chrono::high_resolution_clock::now();
            auto pose_latency = chrono::duration_cast<chrono::milliseconds>(now - cf->frame_time);
            cout << "Tag id: " << cf->tag_id << ", \tTag latency (ms):  " << pose_latency.count() << "\t Tag error: " << cf->err << endl;
            cout << "X: " << cf->pose.t->data[0] + .1 << "\tY: " << cf->pose.t->data[1] + .1 << "\tZ: " << cf->pose.t->data[2] << endl;
            delete cf;
        }
    }


    return 0;
}
