#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <iostream>
#include <vector>
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvdsinfer_custom_impl.h"
#include "nvbufsurface.h"
#include <time.h>
#include "cuda_runtime_api.h"
#include "gstnvdsinfer.h"
#include <opencv2/opencv.hpp>
#include "nvds_parse_bbox_wpod.h"
#include "gst-nvdssr.h"

#define OUTPUT_FILE "output/out.mp4"

#define INTERVAL 2
#define OSD_PROCESS_MODE 0
#define SGIE_INPUT_H 32
#define SGIE_INPUT_W 100
#define SGIE_OUTPUT_SIZE 26 * 37
#define SGIE_PAD_OUT 8

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1
#define PGIE_NET_WIDTH 256
#define PGIE_NET_HEIGHT 256
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define TILED_OUTPUT_WIDTH 1920
#define TILED_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 25000
#define MAX_DISPLAY_LEN 64
#define MEMORY_FEATURES "memory:NVMM"
#define PGIE_CONFIG_FILE "config_pgie.txt"
#define SGIE_CONFIG_FILE "config_sgie.txt"
#define START_TIME 0
#define SHOW_CRNN 0
const std::string alphabet = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// ./deepstream-ALPR file:///opt/nvidia/deepstream/deepstream-5.0/sources/alpr_ds/wpod/deepstream_app_wpod/test1.mp4

/* Duration of recording
 */
bool display = FALSE;

extern "C" bool NvDsInferParseCustomWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

std::string strDecode(std::vector<int> &preds, bool raw)
{
    std::string str;
    if (raw)
    {
        for (auto v : preds)
        {
            str.push_back(alphabet[v]);
        }
    }
    else
    {
        for (size_t i = 0; i < preds.size(); i++)
        {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i]))
                continue;
            str.push_back(alphabet[preds[i]]);
        }
    }
    return str;
}
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
    {
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    }

    case GST_MESSAGE_WARNING:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_warning(msg, &error, &debug);
        g_printerr("WARNING from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        g_free(debug);
        g_printerr("Warning: %s\n", error->message);
        g_error_free(error);
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
#ifdef PLATFORM_TEGRA
    case GST_MESSAGE_ELEMENT:
    {
        if (gst_nvmessage_is_stream_eos(msg))
        {
            guint stream_id;
            if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
            {
                g_print("Got EOS from stream %d\n", stream_id);
            }
        }
        break;
    }
#endif
    default:
        break;
    }
    return TRUE;
}

static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    NvBufSurface *surface = NULL;
    GstBuffer *inbuf = GST_PAD_PROBE_INFO_BUFFER(info);
    GstMapInfo in_map_info;
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ))
    {
        g_error("Error: Failed to map gst buffer\n");
    }
    surface = (NvBufSurface *)in_map_info.data;
    NvDsBatchMeta *batch_meta =
        gst_buffer_get_nvds_batch_meta(inbuf);
    static guint use_device_mem = 0;
    static NvDsInferNetworkInfo networkInfo{3, PGIE_NET_WIDTH, PGIE_NET_HEIGHT};
    NvDsInferParseDetectionParams detectionParams;

#ifndef PLATFORM_TEGRA
    // if (surface->memType != NVBUF_MEM_CUDA_UNIFIED)
    // {
    //     g_error("need NVBUF_MEM_CUDA_UNIFIED memory for opencv\n");
    // }
#endif
    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        cv::Mat in_mat;
        if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] == NULL)
        {
            if (NvBufSurfaceMap(surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0)
            {
                g_error("buffer map to be accessed by CPU failed\n");
            }
        }
        /* Cache the mapped data for CPU access */
        NvBufSurfaceSyncForCpu(surface, frame_meta->batch_id, 0);
        in_mat =
            cv::Mat(surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
                    surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4,
                    surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
                    surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);
        // cv::imwrite("output/in_im.jpg", in_mat);
        // cv::Mat bgr_frame = cv::Mat(cv::Size(surface->surfaceList[frame_meta->batch_id].planeParams.height[0], surface->surfaceList[frame_meta->batch_id].planeParams.width[0]), CV_8UC3);

        // cv::cvtColor(in_mat, bgr_frame, cv::COLOR_RGBA2BGR);

        // cv::imwrite("output/a.jpg", bgr_frame);
        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++)
            {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (use_device_mem && meta->out_buf_ptrs_dev[i])
                {
                    cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                               info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }
            // g_print("\n1 %d\n", meta->num_output_layers);

            /* Parse output tensor and fill detection results into objectList. */
            std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info,
                                                             meta->output_layers_info + meta->num_output_layers);
            std::vector<NvDsInferObjectDetectionInfo> objectList;
#if NVDS_VERSION_MAJOR >= 5
            if (nvds_lib_major_version >= 5)
            {
                if (meta->network_info.width != networkInfo.width ||
                    meta->network_info.height != networkInfo.height ||
                    meta->network_info.channels != networkInfo.channels)
                {
                    g_error("failed to check pgie network info\n");
                }
            }
#endif
            NvDsInferParseCustomWpod(outputLayersInfo, networkInfo,
                                     detectionParams, objectList);
            for (auto &obj : objectList)
            {

                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
                obj_meta->unique_component_id = meta->unique_id;
                obj_meta->confidence = 0.0;
                /* This is an untracked object. Set tracking_id to -1. */
                obj_meta->object_id = UNTRACKED_OBJECT_ID;
                obj_meta->class_id = 0;
                // g_print("\nleft:%f\ttop:%f\twidth:%f\theight:%f", obj.left, obj.top, obj.width, obj.height);
                cv::Rect plate_rect = cv::Rect(obj.left * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH, obj.top * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT, obj.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH, obj.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT);
                // cv::Mat bgr = in_mat(plate_rect);
                cv::Mat cropref = cv::Mat(in_mat, plate_rect);
                cv::Mat crop;
                cropref.copyTo(crop);
                //cv::imwrite("output/" + std::to_string(idx) + "raw.jpg", crop);
                cv::Size out_size;
                // bool type;
                if (obj.width / obj.height < 1.7)
                {
                    out_size.width = 280;
                    out_size.height = 200;
                    // type = 0;
                }
                else
                {
                    out_size.width = 470;
                    out_size.height = 110;
                    // type = 1;
                }

                std::vector<std::vector<float>> H = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                float t_pts[3][4];
                getRectPts(t_pts, 0, 0, out_size.width, out_size.height);
                float ptsh[3][4];
                for (int j = 0; j < 4; j++)
                {
                    ptsh[0][j] = obj.pts[0][j] * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                }
                for (int j = 0; j < 4; j++)
                {
                    ptsh[1][j] = obj.pts[1][j] * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                }
                for (int i = 0; i < 4; i++)
                {
                    ptsh[2][i] = 1.0;
                }
                find_T_matrix(t_pts, ptsh, H);
                float Hf[3][3];
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        Hf[i][j] = H[i][j];
                    }
                }
                cv::Mat H1(3, 3, CV_32F);
                std::memcpy(H1.data, Hf, 3 * 3 * sizeof(float));
                cv::Mat tmp_img;
                cv::warpPerspective(in_mat, tmp_img, H1, out_size, 1, 0);
                //cv::imwrite("output/im" + std::to_string(idx) + ".jpg", tmp_img);
                cv::resize(tmp_img, tmp_img, cv::Size(SGIE_INPUT_W, SGIE_INPUT_H));
                tmp_img.copyTo(in_mat(cv::Rect(SGIE_PAD_OUT, SGIE_PAD_OUT, SGIE_INPUT_W, SGIE_INPUT_H)));

                /* Assign bounding box coordinates. */
                NvOSD_RectParams &rect_params = obj_meta->rect_params;
                NvOSD_TextParams &text_params = obj_meta->text_params;
                NvOSD_RectParams &sgie_rect_params = obj_meta->sgie_rect_params;
                rect_params.left = obj.left * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                rect_params.top = obj.top * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                rect_params.width = obj.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                rect_params.height = obj.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                rect_params.border_width = 3;
                rect_params.has_bg_color = 0;
                rect_params.border_color = (NvOSD_ColorParams){0, 1, 0, 1};
                /* SGIE bounding box*/
                sgie_rect_params.left = SGIE_PAD_OUT;
                sgie_rect_params.top = SGIE_PAD_OUT;
                sgie_rect_params.width = SGIE_INPUT_W;
                sgie_rect_params.height = SGIE_INPUT_H;
                /* display_text requires heap allocated memory. */
                // text_params.display_text = g_strdup(type ? "Bien 1 hang" : "Bien 2 hang");
                /* Display text above the left top corner of the object. */
                text_params.x_offset = rect_params.left;
                text_params.y_offset = rect_params.top - 10;
                /* Set black background for the text. */
                text_params.set_bg_clr = 1;
                text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
                /* Font face, size and color. */
                text_params.font_params.font_name = (gchar *)"Serif";
                text_params.font_params.font_size = 20;
                text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};
                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
                NvBufSurfaceSyncForDevice(surface, frame_meta->batch_id, 0);
            }
        }
        NvBufSurfaceUnMap(surface, frame_meta->batch_id, 0);
    }
    use_device_mem = 1 - use_device_mem;
    gst_buffer_unmap(inbuf, &in_map_info);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
sgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsBatchMeta *batch_meta =
        gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));
    static guint use_device_mem = 0;
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                 l_user = l_user->next)
            {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                    continue;
                /* convert to tensor metadata */
                NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
                for (unsigned int i = 0; i < meta->num_output_layers; i++)
                {
                    NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                    info->buffer = meta->out_buf_ptrs_host[i];
                    if (use_device_mem && meta->out_buf_ptrs_dev[i])
                    {
                        cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                                   info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                    }
                }
                std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info,
                                                                 meta->output_layers_info + meta->num_output_layers);
                float *prob = (float *)outputLayersInfo[0].buffer;
                std::vector<int> preds;
                for (int i = 0; i < 26; i++)
                {
                    int maxj = 0;
                    for (int j = 1; j < 37; j++)
                    {
                        if (prob[37 * i + j] > prob[37 * i + maxj])
                            maxj = j;
                    }
                    preds.push_back(maxj);
                }
                std::string raw = strDecode(preds, true);
                std::string sim = strDecode(preds, false);
                if (SHOW_CRNN)
                {
                    g_print("\nIdx = \nRaw : %s", raw.c_str());
                    g_print("\nSim : %s", sim.c_str());
                }
                else
                {
                    obj_meta->text_params.display_text = g_strdup(sim.c_str());
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

static void
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    GstElement *source_bin = (GstElement *)data;
    GstCapsFeatures *features = gst_caps_get_features(caps, 0); //
    if (!strncmp(name, "video", 5))
    {

        if (gst_caps_features_contains(features, MEMORY_FEATURES))
        {
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                          decoder_src_pad))
            {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref(bin_ghost_pad);
        }
        else
        {
            g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}
static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                                  gchar *name, gpointer user_data)
{
    g_print("Decodebin child added: %s\n", name);
    if (g_strrstr(name, "decodebin") == name)
    {
        g_signal_connect(G_OBJECT(object), "child-added",
                         G_CALLBACK(decodebin_child_added), user_data);
    }
}

static GstElement *
create_source_bin(guint index, gchar *uri)
{
    GstElement *bin = NULL, *uri_decode_bin = NULL;

    gchar bin_name[16] = {};

    g_snprintf(bin_name, 15, "source-bin-%02d", index);
    /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
    bin = gst_bin_new(bin_name);

    /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
    uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

    /* We set the input uri to the source element */
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);
    /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */

    g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added", G_CALLBACK(cb_newpad), bin);

    g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                     G_CALLBACK(decodebin_child_added), bin);

    gst_bin_add(GST_BIN(bin), uri_decode_bin);

    /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                              GST_PAD_SRC)))
    {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    return bin;
}

int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
               *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL, *qtmux = NULL, *videoconvert = NULL,
               *nvosd = NULL, *tiler = NULL, *sgie = NULL, *queue_sgie = NULL, *nvvidconv1 = NULL,
               *h264parser = NULL, *decoder = NULL;
    GstElement *filter1 = NULL, *filter2 = NULL, *filter3 = NULL, *filter4 = NULL, *x264enc = NULL, *converter = NULL;
    GstCaps *caps1 = NULL, *caps2 = NULL, *caps3 = NULL, *caps4 = NULL;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *pgie_src_pad = NULL, *sgie_src_pad = NULL;
    guint i, num_sources;
    guint tiler_rows, tiler_columns;
    guint pgie_batch_size;

    /* Check input arguments */
    if (argc < 2)
    {
        g_printerr("Usage: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
        return -1;
    }
    num_sources = argc - 1;

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("alpr-pipeline");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    queue_sgie = gst_element_factory_make("queue", "queue_sgie");

    if (!pipeline || !streammux)
    {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }
    gst_bin_add(GST_BIN(pipeline), streammux);

    for (i = 0; i < num_sources; i++)
    {
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = {};
        GstElement *source_bin = create_source_bin(i, argv[i + 1]);

        if (!source_bin)
        {
            g_printerr("Failed to create source bin. Exiting.\n");
            return -1;
        }
        gst_bin_add(GST_BIN(pipeline), source_bin);

        g_snprintf(pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_get_request_pad(streammux, pad_name);
        if (!sinkpad)
        {
            g_printerr("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad(source_bin, "src");
        if (!srcpad)
        {
            g_printerr("Failed to get src pad of source bin. Exiting.\n");
            return -1;
        }

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
    }

    /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
    h264parser = gst_element_factory_make("h264parse", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    /* Use nvinfer to infer on batched frame. */
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    sgie = gst_element_factory_make("nvinfer", "secondary-nvinference-engine");
    /* Add queue elements between every two elements */
    queue1 = gst_element_factory_make("queue", "queue1");
    queue2 = gst_element_factory_make("queue", "queue2");
    queue3 = gst_element_factory_make("queue", "queue3");
    queue4 = gst_element_factory_make("queue", "queue4");
    queue5 = gst_element_factory_make("queue", "queue5");
    filter1 = gst_element_factory_make("capsfilter", "filter1");
    filter2 = gst_element_factory_make("capsfilter", "filter2");
    filter3 = gst_element_factory_make("capsfilter", "filter3");
    filter4 = gst_element_factory_make("capsfilter", "filter4");
    x264enc = gst_element_factory_make("x264enc", "h264 encoder");
    converter = gst_element_factory_make("videoconvert", "converter");
    videoconvert = gst_element_factory_make("videoconvert", "converter");
    qtmux = gst_element_factory_make("qtmux", "muxer");
    nvvidconv1 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
    /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
    tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    sink = gst_element_factory_make("filesink", "nvvideo-renderer");

    if (!pgie || !tiler || !nvvidconv || !nvosd || !sink)
    {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }
    if (!converter || !x264enc || !qtmux || !filter3 || !filter4)
    {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make("queue", "nvegl-transform");
    if (!transform)
    {
        g_printerr("One tegra element could not be created. Exiting.\n");
        return -1;
    }
#endif

    // #ifndef PLATFORM_TEGRA
    //     /* Set properties of the nvvideoconvert element
    //    * requires unified cuda memory for opencv blurring on CPU
    //    */
    //     g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 0, NULL);
    // #else
    //     g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 4, NULL);
    // #endif
    g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);

    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                 MUXER_OUTPUT_HEIGHT,
                 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set(G_OBJECT(pgie),
                 "config-file-path", PGIE_CONFIG_FILE, "output-tensor-meta", TRUE, NULL);
    g_object_set(G_OBJECT(sgie), "config-file-path", SGIE_CONFIG_FILE, "output-tensor-meta", TRUE, NULL);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources)
    {
        g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
                   pgie_batch_size, num_sources);
        g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
    }

    tiler_rows = (guint)sqrt(num_sources);
    tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
    /* we set the tiler properties here */
    g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
                 "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

    g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
                 "display-text", OSD_DISPLAY_TEXT, NULL);

    g_object_set(G_OBJECT(sink), "sync", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "location", OUTPUT_FILE, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    // , queue_sgie, sgie
    // h264parser, decoder,
    gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue_sgie, sgie, queue2, tiler, queue3,
                     filter1, nvvidconv, filter2, nvosd, nvvidconv1, filter3, converter, filter4,
                     x264enc, qtmux, transform, sink, NULL);
    /* we link the elements together */
    if (!gst_element_link_many(streammux, queue1, pgie, queue_sgie, sgie, queue2, tiler, queue3,
                               filter1, nvvidconv, filter2, nvosd, nvvidconv1, filter3, converter, filter4,
                               x264enc, qtmux, transform, sink, NULL))
    {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }
#else
    gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue_sgie, sgie, queue2, tiler, queue3,
                     filter1, nvvidconv, filter2, nvosd, nvvidconv1, filter3, converter, filter4,
                     x264enc, qtmux, sink, NULL);
    /* we link the elements together */
    if (!gst_element_link_many(streammux, queue1, pgie, queue_sgie, sgie, queue2, tiler, queue3,
                               filter1, nvvidconv, filter2, nvosd, nvvidconv1, filter3, converter, filter4,
                               x264enc, qtmux, sink, NULL))
    {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }
#endif
    caps1 = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    g_object_set(G_OBJECT(filter1), "caps", caps1, NULL);
    gst_caps_unref(caps1);
    caps2 = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
    g_object_set(G_OBJECT(filter2), "caps", caps2, NULL);
    gst_caps_unref(caps2);
    caps3 = gst_caps_from_string("video/x-raw, format=RGBA");
    g_object_set(G_OBJECT(filter3), "caps", caps3, NULL);
    gst_caps_unref(caps3);
    caps4 = gst_caps_from_string("video/x-raw, format=NV12");
    g_object_set(G_OBJECT(filter4), "caps", caps4, NULL);
    gst_caps_unref(caps4);

    /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
    pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_pad_buffer_probe, NULL, NULL);

    gst_object_unref(pgie_src_pad);
    sgie_src_pad = gst_element_get_static_pad(sgie, "src");
    gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      sgie_pad_buffer_probe, (gpointer)sink, NULL);
    gst_object_unref(sgie_src_pad);

    /* Set the pipeline to "playing" state */
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
    g_print("Now playing:");
    for (i = 0; i < num_sources; i++)
    {
        g_print(" %s", argv[i + 1]);
    }
    g_print("\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}