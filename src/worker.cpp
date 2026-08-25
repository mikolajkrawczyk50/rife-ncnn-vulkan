// rife-ncnn-vulkan-worker
// connects to master, receives model configuration, loads model on GPU/CPU,
// runs load/proc/send pipelined queues for maximum GPU utilization

#include "rife.h"
#include "network_protocol.h"
#include "filesystem_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static std::atomic<bool> g_running(true);

static void sig_handler(int sig)
{
    (void)sig;
    g_running = false;
}

template <typename T>
class SafeQueue {
public:
    SafeQueue(size_t max_size = 4) : max_size(max_size), stopped(false) {}

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

    void reset()
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::queue<T> empty;
        std::swap(queue, empty);
        stopped = false;
    }

private:
    std::queue<T> queue;
    size_t max_size;
    bool stopped;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
};

struct WorkerJob {
    uint32_t task_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    float timestep;
    std::vector<unsigned char> in0_pixels;
    std::vector<unsigned char> in1_pixels;
    bool is_sentinel;
};

struct WorkerResult {
    uint32_t task_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    int32_t ret;
    ncnn::Mat out;
    bool is_sentinel;
};

static void print_usage(void)
{
    fprintf(stderr, "Usage: rife-ncnn-vulkan-worker -c <master_host:port> [options]\n");
    fprintf(stderr, "  -c host:port    master address to connect to (required)\n");
    fprintf(stderr, "  -g gpuid        GPU device id (default 0, -1 for CPU)\n");
    fprintf(stderr, "  -t num_threads  ncnn threads (default 1)\n");
    fprintf(stderr, "  -m modeldir     local model directory override (optional)\n");
    fprintf(stderr, "  -h              show this help\n");
}

int main(int argc, char** argv)
{
    std::string master_addr;
    int gpuid = 0;
    int num_threads = 1;
    std::string local_model_override;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            master_addr = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gpuid = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            local_model_override = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    if (master_addr.empty()) {
        print_usage();
        return 1;
    }

    char master_host[128];
    int master_port = 0;
    if (sscanf(master_addr.c_str(), "%127[^:]:%d", master_host, &master_port) != 2 || master_port <= 0) {
        fprintf(stderr, "invalid master address: %s (expected host:port)\n", master_addr.c_str());
        return 1;
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    sock_init();

    fprintf(stderr, "worker starting (GPU: %d, Threads: %d, Target: %s:%d)\n",
            gpuid, num_threads, master_host, master_port);

    while (g_running) {
        fprintf(stderr, "connecting to master %s:%d...\n", master_host, master_port);
        sock_t fd = sock_connect(master_host, master_port);
        if (fd == SOCK_INVALID) {
            fprintf(stderr, "connect failed, retrying in 2 seconds...\n");
#ifdef _WIN32
            Sleep(2000);
#else
            sleep(2);
#endif
            continue;
        }

        fprintf(stderr, "connected to master!\n");

        // send HELLO
        HelloMsg hello;
        memset(&hello, 0, sizeof(hello));
        snprintf(hello.worker_name, sizeof(hello.worker_name), "worker-gpu%d", gpuid);
        hello.max_in_flight = 4;
        if (sock_send_msg(fd, MSG_HELLO, &hello, sizeof(hello)) < 0) {
            fprintf(stderr, "failed to send HELLO\n");
            closesocket(fd);
            continue;
        }

        // wait for CONFIG from master
        uint32_t msg_type = 0;
        void* body = NULL;
        uint32_t body_len = 0;

        if (sock_recv_msg(fd, &msg_type, &body, &body_len) < 0 || msg_type != MSG_CONFIG || body_len < sizeof(ConfigMsg)) {
            fprintf(stderr, "expected CONFIG from master\n");
            if (body) free(body);
            closesocket(fd);
            continue;
        }

        ConfigMsg cfg;
        memcpy(&cfg, body, sizeof(cfg));
        free(body);

        std::string model_dir = local_model_override.empty() ? cfg.model_dir : local_model_override;
        fprintf(stderr, "received config: model=%s (tta=%d, temp=%d, uhd=%d, v2=%d, v4=%d)\n",
                model_dir.c_str(), cfg.tta_mode, cfg.tta_temporal_mode, cfg.uhd_mode, cfg.rife_v2, cfg.rife_v4);

        // detect rife version if not set
        bool rife_v2 = cfg.rife_v2 != 0;
        bool rife_v4 = cfg.rife_v4 != 0;
        if (!rife_v2 && !rife_v4) {
            if (model_dir.find("rife-v2") != std::string::npos || model_dir.find("rife-v3") != std::string::npos)
                rife_v2 = true;
            else if (model_dir.find("rife-v4") != std::string::npos)
                rife_v4 = true;
        }

        ncnn::create_gpu_instance();

        RIFE* rife = new RIFE(gpuid, cfg.tta_mode, cfg.tta_temporal_mode,
                              cfg.uhd_mode, num_threads, rife_v2, rife_v4);

        path_t model_path = sanitize_dirpath(model_dir.c_str());
        int ret = rife->load(model_path);
        if (ret != 0) {
            fprintf(stderr, "failed to load model from %s (error %d)\n", model_dir.c_str(), ret);
            delete rife;
            ncnn::destroy_gpu_instance();
            closesocket(fd);
#ifdef _WIN32
            Sleep(2000);
#else
            sleep(2);
#endif
            continue;
        }

        fprintf(stderr, "model loaded successfully, sending READY\n");
        if (sock_send_msg(fd, MSG_READY, NULL, 0) < 0) {
            fprintf(stderr, "failed to send READY\n");
            delete rife;
            ncnn::destroy_gpu_instance();
            closesocket(fd);
            continue;
        }

        // initialize pipeline queues: Load (Recv) -> Proc (GPU) -> Save (Send)
        SafeQueue<WorkerJob> to_proc(4);
        SafeQueue<WorkerResult> to_send(4);
        std::atomic<bool> session_active(true);

        // 1. Recv Thread (Network Load)
        std::thread recv_thread([&]() {
            while (session_active && g_running) {
                uint32_t m_type = 0;
                void* m_body = NULL;
                uint32_t m_len = 0;

                if (sock_recv_msg(fd, &m_type, &m_body, &m_len) < 0) {
                    fprintf(stderr, "master connection closed (recv)\n");
                    session_active = false;
                    break;
                }

                if (m_type == MSG_BYE) {
                    fprintf(stderr, "master requested shutdown (MSG_BYE)\n");
                    if (m_body) free(m_body);
                    session_active = false;
                    break;
                }

                if (m_type != MSG_SUBMIT_JOB || m_len < sizeof(SubmitJobMsg)) {
                    fprintf(stderr, "unexpected message type: %u\n", m_type);
                    if (m_body) free(m_body);
                    session_active = false;
                    break;
                }

                SubmitJobMsg* job_hdr = (SubmitJobMsg*)m_body;
                uint32_t w = job_hdr->width;
                uint32_t h = job_hdr->height;
                uint32_t ch = job_hdr->channels;
                uint32_t frame_bytes = w * h * ch;

                if (m_len < sizeof(SubmitJobMsg) + frame_bytes * 2) {
                    fprintf(stderr, "payload size too small for frame pair\n");
                    free(m_body);
                    session_active = false;
                    break;
                }

                const unsigned char* pixels = (const unsigned char*)m_body + sizeof(SubmitJobMsg);

                WorkerJob job;
                job.task_id = job_hdr->task_id;
                job.width = w;
                job.height = h;
                job.channels = ch;
                job.timestep = job_hdr->timestep;
                job.in0_pixels.assign(pixels, pixels + frame_bytes);
                job.in1_pixels.assign(pixels + frame_bytes, pixels + frame_bytes * 2);
                job.is_sentinel = false;

                free(m_body);
                to_proc.put(job);
            }

            WorkerJob sentinel;
            sentinel.is_sentinel = true;
            to_proc.put(sentinel);
            to_proc.stop();
        });

        // 2. Proc Thread (GPU inference)
        std::thread proc_thread([&]() {
            while (g_running) {
                WorkerJob job;
                if (!to_proc.get(job)) break;
                if (job.is_sentinel) {
                    WorkerResult sentinel_res;
                    sentinel_res.is_sentinel = true;
                    to_send.put(sentinel_res);
                    break;
                }

                WorkerResult res;
                res.task_id = job.task_id;
                res.width = job.width;
                res.height = job.height;
                res.channels = job.channels;
                res.is_sentinel = false;
                res.out = ncnn::Mat(job.width, job.height, (size_t)3, 3);

                ncnn::Mat in0(job.width, job.height, (void*)job.in0_pixels.data(), (size_t)3, 3);
                ncnn::Mat in1(job.width, job.height, (void*)job.in1_pixels.data(), (size_t)3, 3);

                res.ret = rife->process(in0, in1, job.timestep, res.out);
                to_send.put(res);
            }
            to_send.stop();
        });

        // 3. Send Thread (Network Save)
        std::thread send_thread([&]() {
            while (g_running) {
                WorkerResult res;
                if (!to_send.get(res)) break;
                if (res.is_sentinel) break;

                ResultMsg hdr;
                hdr.task_id = res.task_id;
                hdr.width = res.width;
                hdr.height = res.height;
                hdr.channels = res.channels;
                hdr.ret = res.ret;

                uint32_t frame_bytes = (res.ret == 0) ? (res.width * res.height * res.channels) : 0;
                uint32_t total_size = sizeof(ResultMsg) + frame_bytes;

                unsigned char* out_buf = (unsigned char*)malloc(total_size);
                if (!out_buf) {
                    session_active = false;
                    break;
                }

                memcpy(out_buf, &hdr, sizeof(ResultMsg));
                if (res.ret == 0 && res.out.data) {
                    memcpy(out_buf + sizeof(ResultMsg), (const unsigned char*)res.out.data, frame_bytes);
                }

                int s_ret = sock_send_msg(fd, MSG_JOB_RESULT, out_buf, total_size);
                free(out_buf);

                if (s_ret < 0) {
                    fprintf(stderr, "failed to send result for task %u\n", res.task_id);
                    session_active = false;
                    break;
                }
            }
        });

        recv_thread.join();
        to_proc.stop();
        proc_thread.join();
        to_send.stop();
        send_thread.join();

        delete rife;
        ncnn::destroy_gpu_instance();
        closesocket(fd);

        fprintf(stderr, "session ended, ready for next connection...\n");
#ifdef _WIN32
        Sleep(1000);
#else
        sleep(1);
#endif
    }

    sock_cleanup();
    fprintf(stderr, "worker shutdown\n");
    return 0;
}
