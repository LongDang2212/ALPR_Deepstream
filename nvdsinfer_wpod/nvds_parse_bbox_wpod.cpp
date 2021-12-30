#include "nvds_parse_bbox_wpod.h"

void get_pos(int x, std::vector<float> &v)
{

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (i * 16 + j == x)
            {
                v.push_back(j + 0.5);
                v.push_back(i + 0.5);
                return;
            }
        }
    }
}
float iou(float lbox[], float lbox2[], float rbox[], float rbox2[])
{

    float x_left = Max(lbox[0], rbox[0]);
    float x_right = Min(lbox2[0], rbox2[0]);
    float y_top = Max(lbox[1], rbox[1]);
    float y_bottom = Min(lbox2[1], rbox2[1]);
    if ((x_right < x_left) || (y_bottom < y_top))
        return 0.0;
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float bb1_area = (lbox2[0] - lbox[0]) * (lbox2[1] - lbox[1]);
    float bb2_area = (rbox2[0] - rbox[0]) * (rbox2[1] - rbox[1]);
    float iou = intersection_area / float(bb1_area + bb2_area - intersection_area);
    return iou;
}
void normal(float pts[][4], float side, std::vector<float> mn, float MN[])
{

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            pts[i][j] *= side;
        }
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[0][j] += mn[0];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[1][j] += mn[1];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[0][j] /= MN[0];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[1][j] /= MN[1];
    }
}
// void getRectPts(float r[][4], float tlx, float tly, float brx, float bry)
// {
//     r[0][0] = tlx;
//     r[0][1] = brx;
//     r[0][2] = brx;
//     r[0][3] = tlx;
//     r[1][0] = tly;
//     r[1][1] = tly;
//     r[1][2] = bry;
//     r[1][3] = bry;
//     r[2][0] = 1;
//     r[2][1] = 1;
//     r[2][2] = 1;
//     r[2][3] = 1;
// }

float IOU_Label(DLabel l1, DLabel l2)
{
    return iou(l1.tl, l1.br, l2.tl, l2.br);
}
bool comp(DLabel l1, DLabel l2)
{
    return l1.prob < l2.prob;
}
void nms(std::vector<DLabel> l, float iou_threshold, std::vector<DLabel> &v)
{
    std::sort(l.begin(), l.end(), comp);
    std::reverse(l.begin(), l.end());

    for (auto lb : l)
    {
        bool non_overlap = true;
        for (auto x : v)
        {

            if (IOU_Label(lb, x) > iou_threshold)
            {
                non_overlap = false;
                break;
            }
        }
        if (non_overlap)
            v.push_back(lb);
    }
}
void getRectPts(float r[][4], float tlx, float tly, float brx, float bry)
{
    r[0][0] = tlx;
    r[0][1] = brx;
    r[0][2] = brx;
    r[0][3] = tlx;
    r[1][0] = tly;
    r[1][1] = tly;
    r[1][2] = bry;
    r[1][3] = bry;
    r[2][0] = 1;
    r[2][1] = 1;
    r[2][2] = 1;
    r[2][3] = 1;
}
void find_T_matrix(float pts[][4], float t_pts[][4], std::vector<std::vector<float>> &r)
{
    double temp[8][9];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            temp[i][j] = 0;
        }
    }
    for (int i = 0; i < 4; i++)
    {
        float xi[3], xil[3];
        for (int j = 0; j < 3; j++)
        {
            xi[j] = t_pts[j][i];
            xil[j] = pts[j][i];
        }
        // xi = xi.T
        for (int j = 3; j < 6; j++)
        {
            temp[i * 2][j] = -xil[2] * xi[j - 3];
        }
        for (int j = 6; j < 9; j++)
        {
            temp[i * 2][j] = xil[1] * xi[j - 6];
        }
        for (int j = 0; j < 3; j++)
        {
            temp[i * 2 + 1][j] = xil[2] * xi[j];
        }
        for (int j = 6; j < 9; j++)
        {
            temp[i * 2 + 1][j] = -xil[0] * xi[j - 6];
        }
    }
    // check done
    double *mat = new double[sizeof(temp) * sizeof(double) / sizeof(float)];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            mat[i * 9 + j] = temp[i][j];
        }
    }
    singular::Matrix<8, 9> A;
    A.fill(mat);
    // check done
    singular::Svd<8, 9>::USV usv = singular::Svd<8, 9>::decomposeUSV(A);
    auto V1 = singular::Svd<8, 9>::getV(usv).clone();
    auto V2 = V1.transpose();
    auto c = V2.row(8);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            r[i][j] = c[i * 3 + j];
        }
    }
}

void post_process(std::vector<DLabel> &out, float *prob, int h, int w)
{
    float net_stride = pow(2, 4);
    float side = ((208 + 40) / 2) / net_stride;
    float MN[] = {w / net_stride, h / net_stride};
    float b[3][4] = {{-0.5, 0.5, 0.5, -0.5},
                     {-0.5, -0.5, 0.5, 0.5},
                     {1.0, 1.0, 1.0, 1.0}};
    std::vector<std::vector<float>> o;
    // #pragma omp parallel for
    for (int i = 0; i < 256; i++)
    {

        if (prob[i] > CONF_THRESH)
        {
            //  gst_print("\nprob: %f", prob[i]);
            std::vector<float> v;
            v.push_back(i);
            for (int j = 0; j < 8; j++)
            {
                v.push_back(prob[i + j * 256]);
            }
            o.push_back(v);
        }
    }
    if(o.empty())
    {
        return;
    }
    std::vector<DLabel> label;
    for (int i = 0; i < o.size(); i++)
    {
        auto v = o.at(i);
        std::vector<float> mn;
        get_pos(v[0], mn);
        float conf = v.at(1);
        float A[2][3] = {{Max(v[3], 0), v[4], v[5]}, {v[6], Max(v[7], 0), v[8]}};
        float pts[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    pts[i][j] += A[i][k] * b[k][j];
                }
            }
        }

        normal(pts, side, mn, MN);
        out.push_back(DLabel(0, pts, conf));
    }
    nms(label, NMS_THRESH, out);
}
bool not_ok(float x)
{
    return (x < 0) || (x > 1);
}
bool NvDsInferParseWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    std::vector<DLabel> out_labels;
    post_process(out_labels, (float *)outputLayersInfo[0].buffer, INPUT_H, INPUT_W);
    for (auto &l : out_labels)
    {
        if (not_ok(l.tl[0]) || not_ok(l.tl[1]) || not_ok(l.wh[0]) || not_ok(l.wh[1]))
            continue;
        NvDsInferParseObjectInfo oinfo;
        oinfo.classId = 1;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                oinfo.pts[i][j] = l.pts[i][j] * 256.0;
            }
        }
        oinfo.left = static_cast<unsigned int>(l.tl[0] * 256);
        oinfo.top = static_cast<unsigned int>(l.tl[1] * 256);
        oinfo.width = static_cast<unsigned int>(l.wh[0] * 256);
        oinfo.height = static_cast<unsigned int>(l.wh[1] * 256);
        oinfo.detectionConfidence = l.prob;
        objectList.push_back(oinfo);
        // g_print("\nprob: %f", oinfo.detectionConfidence);
        // g_print("\nleft:%f\ttop:%f\twidth:%f\theight:%f", oinfo.left, oinfo.top, oinfo.width, oinfo.height);
    }

    return true;
}
bool NvDsInferParseCustomWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseWpod(outputLayersInfo, networkInfo, detectionParams, objectList);
}