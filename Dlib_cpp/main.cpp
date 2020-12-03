
#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <fstream>


//dlib - http://dlib.net/
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
//using namespace dlib;


//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

/* My code updates */
bool doNN = true;
bool showImages = false;
bool showResultImage = false;
int size = 32;



using net_type = dlib::loss_multiclass_log<
        dlib::fc<2,
        dlib::relu<dlib::fc<84,
        dlib::relu<dlib::fc<120,
        dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<16,5,5,1,1,
        dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<6,5,5,1,1,
        dlib::input<dlib::matrix<unsigned char>>
        >>>>>>>>>>>>;


struct space
{
    int x01, y01, x02, y02, x03, y03, x04, y04, occup;
};
int load_parking_geometry(const char* filename, space* spaces);
void extract_space(space* spaces, Mat in_mat, std::vector<Mat>& vector_images);
void draw_detection(space* spaces, Mat& frame);
void evaluation(fstream& detectorOutputFile, fstream& groundTruthFile);
void train_parking();
void test_parking(bool doNN);
void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);
int spaces_num = 56;
cv::Size space_size(80, 80);

int main(int argc, char** argv)
{
    if (doNN) {
        cout << "Train OpenCV Start" << endl;
        train_parking();
        cout << "Train OpenCV End" << endl;
    }

    cout << "Test OpenCV Start" << endl;
    test_parking(doNN);
    cout << "Test OpenCV End" << endl;

}

void train_parking()
{
    //load parking lot geometry
    space* spaces = new space[spaces_num];
    load_parking_geometry("parking_map.txt", spaces);

    std::vector<unsigned long> train_labels;
    std::vector<dlib::matrix<unsigned char>> train_images;

    fstream train_file("train_images_dlib.txt");
    string train_path;

    while (train_file >> train_path)
    {
        Mat frame;

        frame = imread(train_path, 0);
        resize(frame, frame, Size(size, size));

        dlib::cv_image<unsigned char> cimg(frame);
        dlib::matrix<unsigned char> dlibFrame = dlib::mat(cimg);

        train_images.push_back(dlibFrame);
        unsigned long label = 0;

        if (train_path.find("full") != std::string::npos) label = 1;

        train_labels.push_back(label);

    }

    delete spaces;

    cout << "Train images: " << train_images.size() << endl;
    cout << "Train labels: " << train_labels.size() << endl;

    //TODO - Train

    net_type net;
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(256);
    trainer.set_iterations_without_progress_threshold(500);
    trainer.set_max_num_epochs(20);
    trainer.be_verbose();

    trainer.train(train_images, train_labels);
    dlib::serialize("test.dat") << net;

}

void test_parking(bool doNN)
{

    space* spaces = new space[spaces_num];
    load_parking_geometry("parking_map.txt", spaces);

    fstream test_file("test_images.txt");
    ofstream out_label_file("out_prediction.txt");
    string test_path;

    net_type net;
    if(doNN){
        dlib::deserialize("test.dat") >> net;

        while (test_file >> test_path){
            Mat frame, gradX, gradY;
            frame = imread(test_path, 1);
            Mat draw_frame = frame.clone();
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            std::vector<Mat> test_images;
            extract_space(spaces, frame, test_images);

            int colNum = 0;
            for (int i = 0; i < test_images.size(); i++)
            {
                Mat frame = test_images[i];
                resize(frame, frame, Size(size, size));
                dlib::cv_image<unsigned char> cimg(frame);
                dlib::matrix<unsigned char> dlibFrame = dlib::mat(cimg);
                unsigned long percent = net(dlibFrame);
                cout << percent;
                out_label_file << percent << endl;
                spaces[i].occup = percent;
            }
        }
    }

    else {
        while (test_file >> test_path)
        {
            //cout << "test_path: " << test_path << endl;
            Mat frame, gradX, gradY;
            //read testing images
            frame = imread(test_path, 1);
            Mat draw_frame = frame.clone();
            cvtColor(frame, frame, COLOR_BGR2GRAY);

            std::vector<Mat> test_images;
            extract_space(spaces, frame, test_images);

            int colNum = 0;
            for (int i = 0; i < test_images.size(); i++)
            {
                Mat frame = test_images[i];

                //      Odstranit šum a vyhladit obrázek
                medianBlur(frame, frame, 3); //98,6 %
                //blur(frame, frame, Size(3, 3)); //97,6%
                //GaussianBlur(frame, frame, Size(3, 3), 0); //98,2%


                if (showImages) {
                    //namedWindow("test_img", WINDOW_NORMAL);
                    //resizeWindow("test_img", 200, 200);
                    imshow("test_img", frame);
                }

                //Cannyho
                Canny(frame, frame, 100, 100);

                //image brightness threshold
                int sum = 0;
                for (int x = 0; x < frame.rows; x++) {
                    for (int y = 0; y < frame.cols; y++) {
                        if (frame.at<uchar>(x, y) > 150) {
                            sum++;
                        }
                    }
                }

                int predict_label = 0;

                //cout << sum << endl;
                // Threshold ručně získán
                int threshold = 370;
                if (sum > threshold) predict_label = 1;


                out_label_file << predict_label << endl;
                spaces[i].occup = predict_label;

                if (showImages) {
                    //namedWindow("frame", WINDOW_NORMAL);
                    //resizeWindow("frame", 200, 200);
                    imshow("frame", frame);
                    waitKey(0);
                }
            }

            //draw detection
            if (showResultImage) {
                draw_detection(spaces, draw_frame);
                //namedWindow("draw_frame", 0);
                //resizeWindow("draw_frame", 700, 700);
                imshow("draw_frame", draw_frame);
                waitKey(0);
            }
        }
    }

    //evaluation    
    fstream detector_file("out_prediction.txt");
    fstream groundtruth_file("groundtruth.txt");
    evaluation(detector_file, groundtruth_file);
}

int load_parking_geometry(const char* filename, space* spaces)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) return -1;
    int ps_count, i, count;
    count = fscanf(file, "%d\n", &ps_count); // read count of polygons
    for (i = 0; i < ps_count; i++) {
        int p = 0;
        int poly_size;
        count = fscanf(file, "%d->", &poly_size); // read count of polygon vertexes
        int* row = new int[poly_size * 2];
        int j;
        for (j = 0; j < poly_size; j++) {
            int x, y;
            count = fscanf(file, "%d,%d;", &x, &y); // read vertex coordinates
            row[p] = x;
            row[p + 1] = y;
            p = p + 2;
        }
        spaces[i].x01 = row[0];
        spaces[i].y01 = row[1];
        spaces[i].x02 = row[2];
        spaces[i].y02 = row[3];
        spaces[i].x03 = row[4];
        spaces[i].y03 = row[5];
        spaces[i].x04 = row[6];
        spaces[i].y04 = row[7];
        //printf("}\n");
        free(row);
        count = fscanf(file, "\n"); // read end of line
    }
    fclose(file);
    return 1;
}

void extract_space(space* spaces, Mat in_mat, std::vector<Mat>& vector_images)
{
    for (int x = 0; x < spaces_num; x++)
    {
        Mat src_mat(4, 2, CV_32F);
        Mat out_mat(space_size, CV_8U, 1);
        src_mat.at<float>(0, 0) = spaces[x].x01;
        src_mat.at<float>(0, 1) = spaces[x].y01;
        src_mat.at<float>(1, 0) = spaces[x].x02;
        src_mat.at<float>(1, 1) = spaces[x].y02;
        src_mat.at<float>(2, 0) = spaces[x].x03;
        src_mat.at<float>(2, 1) = spaces[x].y03;
        src_mat.at<float>(3, 0) = spaces[x].x04;
        src_mat.at<float>(3, 1) = spaces[x].y04;

        Mat dest_mat(4, 2, CV_32F);
        dest_mat.at<float>(0, 0) = 0;
        dest_mat.at<float>(0, 1) = 0;
        dest_mat.at<float>(1, 0) = out_mat.cols;
        dest_mat.at<float>(1, 1) = 0;
        dest_mat.at<float>(2, 0) = out_mat.cols;
        dest_mat.at<float>(2, 1) = out_mat.rows;
        dest_mat.at<float>(3, 0) = 0;
        dest_mat.at<float>(3, 1) = out_mat.rows;

        Mat H = findHomography(src_mat, dest_mat, 0);
        warpPerspective(in_mat, out_mat, H, space_size);

        //imshow("out_mat", out_mat);
        //waitKey(0);

        vector_images.push_back(out_mat);

    }

}

void draw_detection(space* spaces, Mat& frame)
{
    int sx, sy;
    for (int i = 0; i < spaces_num; i++)
    {
        Point pt1, pt2;
        pt1.x = spaces[i].x01;
        pt1.y = spaces[i].y01;
        pt2.x = spaces[i].x03;
        pt2.y = spaces[i].y03;
        sx = (pt1.x + pt2.x) / 2;
        sy = (pt1.y + pt2.y) / 2;
        if (spaces[i].occup)
        {
            circle(frame, Point(sx, sy - 25), 12, CV_RGB(255, 0, 0), -1);
        }
        else
        {
            circle(frame, Point(sx, sy - 25), 12, CV_RGB(0, 255, 0), -1);
        }
    }
}

void evaluation(fstream& detectorOutputFile, fstream& groundTruthFile)
{
    int detectorLine, groundTruthLine;
    int falsePositives = 0;
    int falseNegatives = 0;
    int truePositives = 0;
    int trueNegatives = 0;
    while (true)
    {
        if (!(detectorOutputFile >> detectorLine)) break;
        groundTruthFile >> groundTruthLine;

        int detect = detectorLine;
        int ground = groundTruthLine;

        //false positives
        if ((detect == 1) && (ground == 0))
        {
            falsePositives++;
        }

        //false negatives
        if ((detect == 0) && (ground == 1))
        {
            falseNegatives++;
        }

        //true positives
        if ((detect == 1) && (ground == 1))
        {
            truePositives++;
        }

        //true negatives
        if ((detect == 0) && (ground == 0))
        {
            trueNegatives++;
        }

    }
    cout << "falsePositives " << falsePositives << endl;
    cout << "falseNegatives " << falseNegatives << endl;
    cout << "truePositives " << truePositives << endl;
    cout << "trueNegatives " << trueNegatives << endl;
    float acc = (float)(truePositives + trueNegatives) / (float)(truePositives + trueNegatives + falsePositives + falseNegatives);

    float TP = truePositives;
    float FP = falsePositives;
    float FN = falseNegatives;
    float TN = trueNegatives;

    float precision = TP / (TP + FP);
    float sensitivity = TP / (TP + FN);

    float F1 = 2 * precision * sensitivity / (precision + sensitivity);
    
    cout << "F1 " << F1 << endl;
    cout << "Accuracy " << acc << endl;
}

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    std::vector< Mat >::const_iterator itr = train_samples.begin();
    std::vector< Mat >::const_iterator end = train_samples.end();
    for (int i = 0; itr != end; ++itr, ++i)
    {
        CV_Assert(itr->cols == 1 ||
            itr->rows == 1);
        if (itr->cols == 1)
        {
            transpose(*(itr), tmp);
            tmp.copyTo(trainData.row(i));
        }
        else if (itr->rows == 1)
        {
            itr->copyTo(trainData.row(i));
        }
    }
}
