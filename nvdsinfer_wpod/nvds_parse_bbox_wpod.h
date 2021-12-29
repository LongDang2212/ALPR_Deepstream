#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <gst/gst.h>
#include "singular/singular.h"
#include "singular/Svd.h"

#define INPUT_W 256
#define INPUT_H 256
#define NMS_THRESH 0.6
#define CONF_THRESH 0.95
#define BATCH_SIZE 1
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))

extern "C" bool NvDsInferParseCustomWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);
void get_pos(int x, std::vector<float> &v);
float iou(float lbox[], float lbox2[], float rbox[], float rbox2[]);
void normal(float pts[][4], float side, std::vector<float> mn, float MN[]);
bool not_ok(float x);
void getRectPts(float r[][4], float tlx, float tly, float brx, float bry);
void find_T_matrix(float pts[][4], float t_pts[][4], std::vector<std::vector<float>> &r);


class Label
{
public:
    float tl[2];
    float br[2];
    float prob;
    float cl;
    float wh[2];

public:
    Label() {}
    Label(float t[], float b[], float c, float p)
    {
        tl[0] = t[0];
        tl[1] = t[1];
        br[0] = b[0];
        br[1] = b[1];
        cl = c;
        prob = p;
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
    Label(const Label &l)
    {
        tl[0] = l.tl[0];
        tl[1] = l.tl[1];
        br[0] = l.br[0];
        br[1] = l.br[1];
        cl = l.cl;
        prob = l.prob;
    }
    void cc(float *t)
    {
        t[0] = tl[0] + wh[0] / 2;
        t[1] = tl[1] + wh[1] / 2;
    }
    void tr(float *t)
    {
        t[0] = br[0];
        t[1] = tl[1];
    }
    void bl(float *t)
    {
        t[0] = tl[0];
        t[1] = br[1];
    }
    float area()
    {
        return wh[0] * wh[1];
    }
};
class DLabel : public Label
{
public:
    float pts[2][4];

public:
    DLabel() {}
    DLabel(float cl, float pts[][4], float p)
    {

        tl[0] = *std::min_element(&pts[0][0], &pts[0][4]);
        tl[1] = *std::min_element(&pts[1][0], &pts[1][4]);

        br[0] = *std::max_element(&pts[0][0], &pts[0][4]);
        br[1] = *std::max_element(&pts[1][0], &pts[1][4]);
        this->cl = cl;
        this->prob = p;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                this->pts[i][j] = pts[i][j];
            }
        }
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
    DLabel(const DLabel &l)
    {
        tl[0] = l.tl[0];
        tl[1] = l.tl[1];
        br[0] = l.br[0];
        br[1] = l.br[1];
        cl = l.cl;
        prob = l.prob;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                this->pts[i][j] = l.pts[i][j];
            }
        }
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
};

float IOU_Label(DLabel l1, DLabel l2);
bool comp(DLabel l1, DLabel l2);
void nms(std::vector<DLabel> l, float iou_threshold, std::vector<DLabel> &v);
void post_process(std::vector<DLabel> &out, float *prob, int h, int w);
static bool NvDsInferParseWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);