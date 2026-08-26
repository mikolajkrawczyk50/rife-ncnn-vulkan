// network protocol for master-worker communication
// raw TCP, cross-platform (POSIX + Windows Winsock)

#ifndef NETWORK_PROTOCOL_H
#define NETWORK_PROTOCOL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET sock_t;
#define SOCK_INVALID INVALID_SOCKET
#define SOCK_ERR() WSAGetLastError()
#ifndef EINTR
#define EINTR WSAEINTR
#endif
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
typedef int sock_t;
#define SOCK_INVALID (-1)
#define SOCK_ERR() errno
#define closesocket close
#endif

#define RIFE_NET_MAGIC 0x52494645 // "RIFE" in ASCII

// protocol message types
enum MsgType : uint32_t {
    MSG_HELLO       = 1, // worker -> master: worker announces itself
    MSG_CONFIG      = 2, // master -> worker: sends model name/path + flags
    MSG_READY       = 3, // worker -> master: model loaded and ready
    MSG_SUBMIT_JOB  = 4, // master -> worker: frame pair + timestep
    MSG_JOB_RESULT  = 5, // worker -> master: interpolated frame
    MSG_BYE         = 6, // clean shutdown
    MSG_ERROR       = 7, // error alert
};

#pragma pack(push, 1)

struct MsgHeader {
    uint32_t magic;      // RIFE_NET_MAGIC
    uint32_t msg_type;
    uint32_t body_len;
};

struct HelloMsg {
    char    worker_name[64];
    int32_t max_in_flight;
};

struct ConfigMsg {
    int32_t tta_mode;
    int32_t tta_temporal_mode;
    int32_t uhd_mode;
    int32_t rife_v2;
    int32_t rife_v4;
    char    model_dir[256]; // model name or path
};

struct SubmitJobMsg {
    uint32_t task_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    float    timestep;
};

struct ResultMsg {
    uint32_t task_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    int32_t  ret;
};

#pragma pack(pop)

// maximum message body size sanity limit (512 MB)
#define MAX_MSG_BODY_LEN (512 * 1024 * 1024)

// set low-latency TCP_NODELAY, SO_KEEPALIVE, and send/recv timeouts
static inline void sock_set_options(sock_t fd, int timeout_seconds = 15)
{
    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (const char*)&yes, sizeof(yes));

#ifdef _WIN32
    DWORD timeout_ms = (DWORD)(timeout_seconds * 1000);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv;
    tv.tv_sec = timeout_seconds;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif
}

// send exactly len bytes, handles partial writes and signal interruptions
static inline int sock_send_all(sock_t fd, const void* buf, int len)
{
    const char* p = (const char*)buf;
    while (len > 0) {
#if defined(MSG_NOSIGNAL)
        int n = (int)send(fd, p, len, MSG_NOSIGNAL);
#else
        int n = (int)send(fd, p, len, 0);
#endif
        if (n < 0) {
            if (SOCK_ERR() == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

// recv exactly len bytes, handles partial reads and signal interruptions
static inline int sock_recv_all(sock_t fd, void* buf, int len)
{
    char* p = (char*)buf;
    while (len > 0) {
        int n = (int)recv(fd, p, len, 0);
        if (n < 0) {
            if (SOCK_ERR() == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1; // connection closed (EOF)
        p += n;
        len -= n;
    }
    return 0;
}

// send a message: header + body
static inline int sock_send_msg(sock_t fd, uint32_t msg_type, const void* body, uint32_t body_len)
{
    MsgHeader hdr;
    hdr.magic = RIFE_NET_MAGIC;
    hdr.msg_type = msg_type;
    hdr.body_len = body_len;
    if (sock_send_all(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (body_len > 0 && body) {
        if (sock_send_all(fd, body, body_len) < 0) return -1;
    }
    return 0;
}

// recv a message: returns msg_type, allocates *body (caller must free if body_len > 0)
static inline int sock_recv_msg(sock_t fd, uint32_t* msg_type, void** body, uint32_t* body_len)
{
    if (body) *body = NULL;
    if (msg_type) *msg_type = 0;
    if (body_len) *body_len = 0;

    MsgHeader hdr;
    if (sock_recv_all(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (hdr.magic != RIFE_NET_MAGIC) {
        return -1; // reject non-RIFE protocol / port scanners
    }
    if (hdr.body_len > MAX_MSG_BODY_LEN) return -1; // sanity limit check

    *msg_type = hdr.msg_type;
    *body_len = hdr.body_len;

    if (hdr.body_len == 0) {
        *body = NULL;
        return 0;
    }

    *body = malloc(hdr.body_len);
    if (!*body) return -1;

    if (sock_recv_all(fd, *body, hdr.body_len) < 0) {
        free(*body);
        *body = NULL;
        return -1;
    }
    return 0;
}

// connect to host:port, returns socket fd or SOCK_INVALID
static inline sock_t sock_connect(const char* host, int port)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0)
        return SOCK_INVALID;

    sock_t fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == SOCK_INVALID) {
        freeaddrinfo(res);
        return SOCK_INVALID;
    }

    if (connect(fd, res->ai_addr, (int)res->ai_addrlen) < 0) {
        closesocket(fd);
        freeaddrinfo(res);
        return SOCK_INVALID;
    }

    freeaddrinfo(res);

    sock_set_options(fd, 15);
    return fd;
}

// bind + listen, returns socket fd or SOCK_INVALID
static inline sock_t sock_listen(int port, int backlog = 32)
{
    sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCK_INVALID) return SOCK_INVALID;

    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(fd);
        return SOCK_INVALID;
    }

    if (listen(fd, backlog) < 0) {
        closesocket(fd);
        return SOCK_INVALID;
    }

    return fd;
}

// accept incoming connection, returns socket fd or SOCK_INVALID
static inline sock_t sock_accept(sock_t listen_fd, char* client_ip = NULL, int max_ip_len = 0)
{
    struct sockaddr_in addr;
#ifdef _WIN32
    int addrlen = sizeof(addr);
#else
    socklen_t addrlen = sizeof(addr);
#endif
    sock_t fd = accept(listen_fd, (struct sockaddr*)&addr, &addrlen);
    if (fd != SOCK_INVALID) {
        sock_set_options(fd, 15);
        if (client_ip && max_ip_len > 0) {
            inet_ntop(AF_INET, &addr.sin_addr, client_ip, max_ip_len);
        }
    }
    return fd;
}

// init networking (Winsock startup on Windows, ignore SIGPIPE on Linux)
static inline void sock_init(void)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
}

static inline void sock_cleanup(void)
{
#ifdef _WIN32
    WSACleanup();
#endif
}

#endif // NETWORK_PROTOCOL_H
