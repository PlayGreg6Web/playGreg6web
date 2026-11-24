#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace playgta6::os {

struct HttpfsMailbox;
class HttpfsMailboxClient;

void log(std::string_view message);

// Optionally attach an in-process or shared-memory HTTPFS mailbox (or client).
// When attached, httpfs_open / httpfs_read / httpfs_close route through the
// mailbox protocol; otherwise they fall back to Emscripten EM_IMPORT(env)
// host calls (or return an error on native builds). Pass nullptr to detach.
void httpfs_attach_mailbox(HttpfsMailbox* mailbox);
void httpfs_attach_client(HttpfsMailboxClient* client);
bool httpfs_has_attached_mailbox();

int32_t httpfs_open(std::string_view rel_path);
int64_t httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len);
void httpfs_close(int32_t handle);
void wgpu_submit(const uint8_t* data, size_t len);

// Sees every packet batch handed to wgpu_submit (complete packets) before it is forwarded, on any
// build (native builds otherwise discard them). One observer; nullptr detaches. Used by the
// native harness's frame capture (gpu/frame_capture.h).
using WgpuSubmitObserver = void (*)(const uint8_t* data, size_t len);
void set_wgpu_submit_observer(WgpuSubmitObserver observer);

}  // namespace playgta6::os
