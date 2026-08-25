// rife-ncnn-vulkan-master
// accepts incoming worker connections over LAN, distributes frame pairs with
// pipelined job dispatching, automatic task re-queuing on worker drop,
// and background image save threads

#include "network_protocol.h"
#include "filesystem_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <vector>
#include <deque>
#include <queue>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

#if _WIN32
#include "wic_image.h"
#else
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif
#include "webp_image.h"

#if _WIN32
#include <locale.h>
#include <wchar.h>
#include <io.h>
#include <fcntl.h>
#endif

static std::atomic<bool> g_running(true);

static void sig_handler(int sig)
{
    (void)sig;
    g_running = false;
}

static int decode_image_to_rgb(const path_t& imagepath, unsigned char** pixels, int* w, int* h)
{
    int c = 0;
    unsigned char* pixeldata = NULL;

#if _WIN32
    FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
    FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    int length = (int)ftell(fp);
    rewind(fp);
    unsigned char* filedata = (unsigned char*)malloc(length);
    if (filedata) {
        if ((int)fread(filedata, 1, length, fp) != length) {
            free(filedata);
            filedata = NULL;
        }
    }
    fclose(fp);

    if (!filedata) return -1;

    pixeldata = webp_load(filedata, length, w, h, &c);
    if (!pixeldata) {
#if _WIN32
        pixeldata = wic_decode_image(imagepath.c_str(), w, h, &c);
#else
        pixeldata = stbi_load_from_memory(filedata, length, w, h, &c, 3);
        c = 3;
#endif
    }
    free(filedata);

    if (!pixeldata) return -1;

    *pixels = pixeldata;
    return 0;
}

static int encode_image_from_rgb(const path_t& imagepath, const unsigned char* pixels, int w, int h)
{
    path_t ext = get_file_extension(imagepath);
    int success = 0;

    if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP")) {
        success = webp_save(imagepath.c_str(), w, h, 3, pixels);
    } else if (ext == PATHSTR("png") || ext == PATHSTR("PNG")) {
#if _WIN32
        success = wic_encode_image(imagepath.c_str(), w, h, 3, pixels);
#else
        success = stbi_write_png(imagepath.c_str(), w, h, 3, pixels, 0);
#endif
    } else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")) {
#if _WIN32
        success = wic_encode_jpeg_image(imagepath.c_str(), w, h, 3, pixels);
#else
        success = stbi_write_jpg(imagepath.c_str(), w, h, 3, pixels, 100);
#endif
    }

    return success ? 0 : -1;
}

struct MasterTask {
    int id;
    path_t in0path;
    path_t in1path;
    path_t outpath;
    float timestep;
};

struct SaveTask {
    int id;
    path_t outpath;
    path_t in0path;
    path_t in1path;
    float timestep;
    int width;
    int height;
    unsigned char* pixels; // dynamically allocated, freed by save thread
    bool is_sentinel;
};

class TaskPool {
public:
    void init(const std::vector<MasterTask>& tasks)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& t : tasks) {
            pending.push_back(t.id);
        }
    }

    bool pop(int& task_id)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (pending.empty()) return false;
        task_id = pending.front();
        pending.pop_front();
        return true;
    }

    void requeue_front(const std::vector<int>& task_ids)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto it = task_ids.rbegin(); it != task_ids.rend(); ++it) {
            pending.push_front(*it);
        }
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(mtx);
        return pending.size();
    }

private:
    std::deque<int> pending;
    std::mutex mtx;
};

template <typename T>
class SafeQueue {
public:
    SafeQueue(size_t max_size = 16) : max_size(max_size), stopped(false) {}

    void put(const T& val)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [this]() { return stopped || queue.size() < max_size; });
        if (stopped) return;
        queue.push(val);
        cv_not_empty.notify_one();
    }

    bool get(T& val)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [this]() { return stopped || !queue.empty(); });
        if (queue.empty()) return false;
        val = queue.front();
        queue.pop();
        cv_not_full.notify_one();
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mtx);
        stopped = true;
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }

private:
    std::queue<T> queue;
    size_t max_size;
    bool stopped;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
};

static void print_usage(void)
{
    fprintf(stderr, "Usage: rife-ncnn-vulkan-master -p <port> -m <modeldir> [options]\n");
    fprintf(stderr, "  -p port         listen port for workers (required)\n");
    fprintf(stderr, "  -m modeldir     model directory path or name (required)\n");
    fprintf(stderr, "  -i inputdir     input image folder\n");
    fprintf(stderr, "  -o outputdir    output image folder\n");
    fprintf(stderr, "  -0 input0file   first input frame (single pair mode)\n");
    fprintf(stderr, "  -1 input1file   second input frame (single pair mode)\n");
    fprintf(stderr, "  -n numframe     target frame count (default: 2x source frames)\n");
    fprintf(stderr, "  -s timestep     timestep for single pair (default: 0.5)\n");
    fprintf(stderr, "  -f pattern      output filename pattern (default: %%08d.png)\n");
    fprintf(stderr, "  -x              enable spatial TTA\n");
    fprintf(stderr, "  -z              enable temporal TTA\n");
    fprintf(stderr, "  -u              enable UHD mode\n");
    fprintf(stderr, "  -j jobs_save    disk save threads (default: 2)\n");
    fprintf(stderr, "  -v              verbose progress output\n");
    fprintf(stderr, "  -h              show this help\n");
}

int main(int argc, char** argv)
{
    int listen_port = -1;
    path_t model;
    path_t inputpath;
    path_t outputpath;
    path_t input0path;
    path_t input1path;
    int numframe = 0;
    float timestep = 0.5f;
    int tta_mode = 0;
    int tta_temporal_mode = 0;
    int uhd_mode = 0;
    int jobs_save = 2;
    int verbose = 0;
    path_t pattern_format = PATHSTR("%08d.png");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            listen_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            model = wbuf;
#else
            model = argv[++i];
#endif
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            inputpath = wbuf;
#else
            inputpath = argv[++i];
#endif
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            outputpath = wbuf;
#else
            outputpath = argv[++i];
#endif
        } else if (strcmp(argv[i], "-0") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            input0path = wbuf;
#else
            input0path = argv[++i];
#endif
        } else if (strcmp(argv[i], "-1") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            input1path = wbuf;
#else
            input1path = argv[++i];
#endif
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numframe = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            timestep = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
#if _WIN32
            wchar_t wbuf[256];
            mbstowcs(wbuf, argv[++i], 255);
            pattern_format = wbuf;
#else
            pattern_format = argv[++i];
#endif
        } else if (strcmp(argv[i], "-x") == 0) {
            tta_mode = 1;
        } else if (strcmp(argv[i], "-z") == 0) {
            tta_temporal_mode = 1;
        } else if (strcmp(argv[i], "-u") == 0) {
            uhd_mode = 1;
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            jobs_save = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    if (listen_port <= 0 || model.empty()) {
        print_usage();
        return 1;
    }

    if (((input0path.empty() || input1path.empty()) && inputpath.empty()) || outputpath.empty()) {
        print_usage();
        return 1;
    }

    path_t pattern = get_file_name_without_extension(pattern_format);
    path_t format = get_file_extension(pattern_format);
    if (format.empty()) {
        pattern = PATHSTR("%08d");
        format = pattern_format;
    }
    if (pattern.empty()) {
        pattern = PATHSTR("%08d");
    }

    if (!path_is_directory(outputpath)) {
        path_t ext = get_file_extension(outputpath);
        if (ext == PATHSTR("png") || ext == PATHSTR("PNG")) format = PATHSTR("png");
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP")) format = PATHSTR("webp");
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")) format = PATHSTR("jpg");
    }

    bool rife_v2 = false;
    bool rife_v4 = false;
    if (model.find(PATHSTR("rife-v2")) != path_t::npos || model.find(PATHSTR("rife-v3")) != path_t::npos) {
        rife_v2 = true;
    } else if (model.find(PATHSTR("rife-v4")) != path_t::npos) {
        rife_v4 = true;
    }

    // collect input tasks
    std::vector<MasterTask> all_tasks;
    if (!inputpath.empty() && path_is_directory(inputpath)) {
        std::vector<path_t> filenames;
        if (list_directory(inputpath, filenames) != 0 || filenames.size() < 2) {
            fprintf(stderr, "failed to read input directory or less than 2 frames\n");
            return 1;
        }

        const int count = (int)filenames.size();
        if (numframe <= 0) numframe = count * 2;

        all_tasks.resize(numframe);
        double scale = (double)count / numframe;
        for (int i = 0; i < numframe; i++) {
            float fx = (float)(i * scale);
            int sx = static_cast<int>(floor(fx));
            fx -= sx;

            if (sx < 0) { sx = 0; fx = 0.f; }
            if (sx >= count - 1) { sx = count - 2; fx = 1.f; }

            path_t filename0 = filenames[sx];
            path_t filename1 = filenames[sx + 1];

#if _WIN32
            wchar_t tmp[256];
            swprintf(tmp, pattern.c_str(), i + 1);
#else
            char tmp[256];
            snprintf(tmp, sizeof(tmp), pattern.c_str(), i + 1);
#endif
            path_t out_filename = path_t(tmp) + PATHSTR('.') + format;

            all_tasks[i].id = i;
            all_tasks[i].in0path = inputpath + PATHSTR('/') + filename0;
            all_tasks[i].in1path = inputpath + PATHSTR('/') + filename1;
            all_tasks[i].outpath = outputpath + PATHSTR('/') + out_filename;
            all_tasks[i].timestep = fx;
        }
    } else {
        MasterTask t;
        t.id = 0;
        t.in0path = input0path;
        t.in1path = input1path;
        t.outpath = outputpath;
        t.timestep = timestep;
        all_tasks.push_back(t);
    }

    int total_tasks = (int)all_tasks.size();
    fprintf(stderr, "rife-ncnn-vulkan-master starting: %d total frames to render\n", total_tasks);

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    sock_init();

    sock_t listen_fd = sock_listen(listen_port, 32);
    if (listen_fd == SOCK_INVALID) {
        fprintf(stderr, "failed to bind and listen on port %d\n", listen_port);
        return 1;
    }

    fprintf(stderr, "master listening on port %d, waiting for workers...\n", listen_port);

    TaskPool task_pool;
    task_pool.init(all_tasks);

    SafeQueue<SaveTask> save_queue(32);
    std::atomic<int> completed_count(0);
    std::atomic<bool> all_done(false);

    // start save threads
    std::vector<std::thread> save_threads;
    for (int j = 0; j < jobs_save; j++) {
        save_threads.emplace_back([&save_queue, &completed_count, total_tasks, verbose]() {
            while (g_running) {
                SaveTask st;
                if (!save_queue.get(st)) break;
                if (st.is_sentinel) break;

                encode_image_from_rgb(st.outpath, st.pixels, st.width, st.height);
                free(st.pixels);

                int done = ++completed_count;
                if (verbose || done % 10 == 0 || done == total_tasks) {
                    fprintf(stderr, "[%d/%d] (%.1f%%) rendered\n",
                            done, total_tasks, (done * 100.0f) / total_tasks);
                }
            }
        });
    }

    std::mutex workers_mtx;
    std::vector<sock_t> active_worker_sockets;
    std::vector<std::thread> worker_threads;

    // listener thread accepting workers dynamically
    std::thread listener_thread([&]() {
        while (g_running && !all_done) {
            char worker_ip[64] = {0};
            sock_t worker_fd = sock_accept(listen_fd, worker_ip, sizeof(worker_ip));
            if (worker_fd == SOCK_INVALID) {
                if (!g_running || all_done) break;
                continue;
            }

            fprintf(stderr, "worker joined from %s\n", worker_ip);

            // recv HELLO
            uint32_t msg_type = 0;
            void* body = NULL;
            uint32_t body_len = 0;

            if (sock_recv_msg(worker_fd, &msg_type, &body, &body_len) < 0 || msg_type != MSG_HELLO) {
                fprintf(stderr, "failed to get HELLO from worker %s\n", worker_ip);
                if (body) free(body);
                closesocket(worker_fd);
                continue;
            }
            if (body) free(body);

            // send CONFIG
            ConfigMsg cfg;
            memset(&cfg, 0, sizeof(cfg));
            cfg.tta_mode = tta_mode;
            cfg.tta_temporal_mode = tta_temporal_mode;
            cfg.uhd_mode = uhd_mode;
            cfg.rife_v2 = rife_v2 ? 1 : 0;
            cfg.rife_v4 = rife_v4 ? 1 : 0;
#if _WIN32
            wcstombs(cfg.model_dir, model.c_str(), sizeof(cfg.model_dir) - 1);
#else
            strncpy(cfg.model_dir, model.c_str(), sizeof(cfg.model_dir) - 1);
#endif

            if (sock_send_msg(worker_fd, MSG_CONFIG, &cfg, sizeof(cfg)) < 0) {
                fprintf(stderr, "failed to send CONFIG to worker %s\n", worker_ip);
                closesocket(worker_fd);
                continue;
            }

            // recv READY
            if (sock_recv_msg(worker_fd, &msg_type, &body, &body_len) < 0 || msg_type != MSG_READY) {
                fprintf(stderr, "worker %s failed to get READY\n", worker_ip);
                if (body) free(body);
                closesocket(worker_fd);
                continue;
            }
            if (body) free(body);

            fprintf(stderr, "worker %s ready, assigning work...\n", worker_ip);

            {
                std::lock_guard<std::mutex> lk(workers_mtx);
                active_worker_sockets.push_back(worker_fd);
            }

            // spawn handler thread for this worker
            worker_threads.emplace_back([&, worker_fd, worker_ip_str = std::string(worker_ip)]() {
                std::vector<int> in_flight;

                while (g_running && !all_done) {
                    int task_id = -1;
                    if (!task_pool.pop(task_id)) {
                        // no tasks currently pending, wait briefly or check if done
                        if (completed_count >= total_tasks) break;
#ifdef _WIN32
                        Sleep(50);
#else
                        usleep(50000);
#endif
                        continue;
                    }

                    const MasterTask& task = all_tasks[task_id];

                    unsigned char* in0_pixels = NULL;
                    unsigned char* in1_pixels = NULL;
                    int w0 = 0, h0 = 0, w1 = 0, h1 = 0;

                    if (decode_image_to_rgb(task.in0path, &in0_pixels, &w0, &h0) != 0 ||
                        decode_image_to_rgb(task.in1path, &in1_pixels, &w1, &h1) != 0 ||
                        w0 != w1 || h0 != h1) {
                        fprintf(stderr, "error decoding frames for task %d\n", task_id);
                        if (in0_pixels) free(in0_pixels);
                        if (in1_pixels) free(in1_pixels);
                        continue;
                    }

                    int frame_bytes = w0 * h0 * 3;
                    uint32_t body_size = sizeof(SubmitJobMsg) + frame_bytes * 2;
                    unsigned char* job_buf = (unsigned char*)malloc(body_size);
                    if (!job_buf) {
                        free(in0_pixels);
                        free(in1_pixels);
                        task_pool.requeue_front({task_id});
                        break;
                    }

                    SubmitJobMsg* hdr = (SubmitJobMsg*)job_buf;
                    hdr->task_id = (uint32_t)task_id;
                    hdr->width = (uint32_t)w0;
                    hdr->height = (uint32_t)h0;
                    hdr->channels = 3;
                    hdr->timestep = task.timestep;

                    memcpy(job_buf + sizeof(SubmitJobMsg), in0_pixels, frame_bytes);
                    memcpy(job_buf + sizeof(SubmitJobMsg) + frame_bytes, in1_pixels, frame_bytes);

                    free(in0_pixels);
                    free(in1_pixels);

                    in_flight.push_back(task_id);

                    if (sock_send_msg(worker_fd, MSG_SUBMIT_JOB, job_buf, body_size) < 0) {
                        free(job_buf);
                        break; // socket error, handle below
                    }
                    free(job_buf);

                    // receive result
                    uint32_t res_type = 0;
                    void* res_body = NULL;
                    uint32_t res_len = 0;

                    if (sock_recv_msg(worker_fd, &res_type, &res_body, &res_len) < 0 || res_type != MSG_JOB_RESULT) {
                        if (res_body) free(res_body);
                        break; // worker dropped
                    }

                    ResultMsg* res_hdr = (ResultMsg*)res_body;
                    if (res_hdr->ret != 0 || res_len < sizeof(ResultMsg) + (uint32_t)frame_bytes) {
                        fprintf(stderr, "worker returned error %d for task %d\n", res_hdr->ret, task_id);
                        free(res_body);
                        break;
                    }

                    // remove from in_flight
                    for (auto it = in_flight.begin(); it != in_flight.end(); ++it) {
                        if (*it == task_id) {
                            in_flight.erase(it);
                            break;
                        }
                    }

                    // send to save queue
                    SaveTask st;
                    st.id = task_id;
                    st.outpath = task.outpath;
                    st.in0path = task.in0path;
                    st.in1path = task.in1path;
                    st.timestep = task.timestep;
                    st.width = w0;
                    st.height = h0;
                    st.is_sentinel = false;

                    st.pixels = (unsigned char*)malloc(frame_bytes);
                    if (st.pixels) {
                        memcpy(st.pixels, (unsigned char*)res_body + sizeof(ResultMsg), frame_bytes);
                        save_queue.put(st);
                    }
                    free(res_body);

                    if (completed_count >= total_tasks) {
                        all_done = true;
                        break;
                    }
                }

                // If any tasks were in-flight when worker dropped, re-queue them!
                if (!in_flight.empty()) {
                    fprintf(stderr, "worker %s disconnected! Re-queuing %d in-flight tasks.\n",
                            worker_ip_str.c_str(), (int)in_flight.size());
                    task_pool.requeue_front(in_flight);
                }

                closesocket(worker_fd);
            });
        }
    });

    // main loop waiting for completion
    while (g_running && completed_count < total_tasks) {
#ifdef _WIN32
        Sleep(200);
#else
        usleep(200000);
#endif
    }

    all_done = true;

    // notify all connected workers of shutdown
    {
        std::lock_guard<std::mutex> lk(workers_mtx);
        for (sock_t s : active_worker_sockets) {
            sock_send_msg(s, MSG_BYE, NULL, 0);
        }
    }

    // close listen socket so accept() unblocks
    closesocket(listen_fd);

    if (listener_thread.joinable())
        listener_thread.join();

    for (auto& th : worker_threads) {
        if (th.joinable()) th.join();
    }

    // signal save threads to stop
    for (int j = 0; j < jobs_save; j++) {
        SaveTask sentinel;
        sentinel.is_sentinel = true;
        save_queue.put(sentinel);
    }
    save_queue.stop();

    for (auto& th : save_threads) {
        if (th.joinable()) th.join();
    }

    sock_cleanup();

    if (completed_count >= total_tasks) {
        fprintf(stderr, "Render complete! %d/%d frames rendered successfully.\n",
                completed_count.load(), total_tasks);
        return 0;
    } else {
        fprintf(stderr, "Render interrupted! %d/%d frames rendered.\n",
                completed_count.load(), total_tasks);
        return 1;
    }
}
