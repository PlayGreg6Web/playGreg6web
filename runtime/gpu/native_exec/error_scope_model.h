// Diagnostic error-scope attribution (--gpu-error-scopes): Dawn uncaptured device errors carry no shader or
// resource context, so the executor wraps each packet (pipeline creation, bind group creation, pass
// commands, the present's encode + submit) in Push/PopErrorScope and records the error with the phase, the
// present index and the shader hashes / resource ids in scope. This header is the pure part: the phase of
// an opcode, the on/off policy and the bounded, deduplicated log. No GPU involved.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::native_exec {

inline constexpr size_t kErrorScopeMaxEntries = 64;     // distinct (phase, message, context) rows kept
inline constexpr size_t kErrorScopeMaxMessages = 128;   // distinct messages counted
inline constexpr uint64_t kErrorScopeAutoErrors = 200;  // auto mode stops after this many scoped errors

// The phase an opcode belongs to: where the error was raised.
inline const char* error_scope_phase(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::CreateRenderPipeline:
        case GpuOpcode::CreateComputePipeline:
        case GpuOpcode::CreateRenderPipeline2:
        case GpuOpcode::CreateComputePipeline2:
        case GpuOpcode::CreatePipelineLayout:
        case GpuOpcode::CreateShaderModule:
            return "pipeline_create";
        case GpuOpcode::CreateBindGroup:
        case GpuOpcode::CreateBindGroup2:
        case GpuOpcode::CreateBindGroupLayout:
            return "bind_group_create";
        case GpuOpcode::Present:
            return "present_submit";  // the frame's passes are encoded, finished and submitted here
        case GpuOpcode::BeginRenderPass:
        case GpuOpcode::EndRenderPass:
        case GpuOpcode::Draw:
        case GpuOpcode::DrawIndexed:
        case GpuOpcode::DrawIndirect:
        case GpuOpcode::DrawIndexedIndirect:
        case GpuOpcode::DispatchCompute:
        case GpuOpcode::DispatchIndirect:
        case GpuOpcode::SetPipeline:
        case GpuOpcode::SetComputePipeline:
        case GpuOpcode::SetBindGroup:
        case GpuOpcode::SetBindGroup2:
            return "pass";
        default:
            return "other";
    }
}

// Scoping is explicit (--gpu-error-scopes), or armed automatically once a device error has been seen and
// disarmed again after kErrorScopeAutoErrors scoped errors, so a healthy run pays nothing.
inline bool error_scope_active(bool explicit_on, uint64_t device_errors, uint64_t scoped_errors) {
    if (explicit_on) return true;
    return device_errors > 0 && scoped_errors < kErrorScopeAutoErrors;
}

struct ErrorScopeEntry {
    std::string message;
    std::string kind;  // validation / out-of-memory / internal
    std::string phase;
    std::string context;  // "vs=<hash> fs=<hash> pipeline=<id> bind_group=<id> ..."
    uint64_t present_index = 0;
    uint64_t packet = 0;
    uint16_t opcode = 0;
    uint64_t count = 0;
};

class ErrorScopeLog {
public:
    void record(const std::string& kind, const std::string& message, const char* phase, const std::string& context,
                uint64_t present_index, uint64_t packet, uint16_t opcode) {
        std::lock_guard<std::mutex> lock(mu_);
        ++total_;
        std::string msg = message.size() > 400 ? message.substr(0, 400) : message;
        if (by_message_.size() < kErrorScopeMaxMessages || by_message_.count(msg)) ++by_message_[msg];
        else ++messages_dropped_;
        for (ErrorScopeEntry& e : entries_) {
            if (e.message == msg && e.phase == phase && e.context == context) {
                ++e.count;
                return;
            }
        }
        if (entries_.size() >= kErrorScopeMaxEntries) {
            ++entries_dropped_;
            return;
        }
        ErrorScopeEntry e;
        e.message = std::move(msg);
        e.kind = kind;
        e.phase = phase;
        e.context = context;
        e.present_index = present_index;
        e.packet = packet;
        e.opcode = opcode;
        e.count = 1;
        entries_.push_back(std::move(e));
    }

    uint64_t total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_;
    }
    std::vector<ErrorScopeEntry> entries() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_;
    }
    std::map<std::string, uint64_t> by_message() const {
        std::lock_guard<std::mutex> lock(mu_);
        return by_message_;
    }
    uint64_t entries_dropped() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_dropped_;
    }

    static std::string json_escape(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"' || c == '\\') {
                o += '\\';
                o += c;
            } else if (c == '\n') {
                o += "\\n";
            } else if (static_cast<unsigned char>(c) >= 0x20) {
                o += c;
            }
        }
        return o;
    }

    // {"total": N, "entries_dropped": N, "by_message": [{"count", "text"}...], "entries": [...]}
    std::string json() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::string j = "{\"total\": " + std::to_string(total_) + ", \"entries_dropped\": " + std::to_string(entries_dropped_) +
                        ", \"messages_dropped\": " + std::to_string(messages_dropped_) + ", \"by_message\": [";
        std::vector<std::pair<std::string, uint64_t>> rows(by_message_.begin(), by_message_.end());
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(rows[i].second) + ", \"text\": \"" + json_escape(rows[i].first) + "\"}";
        }
        j += "], \"entries\": [";
        for (size_t i = 0; i < entries_.size(); ++i) {
            const ErrorScopeEntry& e = entries_[i];
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(e.count) + ", \"kind\": \"" + e.kind + "\", \"phase\": \"" + e.phase +
                 "\", \"present_index\": " + std::to_string(e.present_index) + ", \"packet\": " + std::to_string(e.packet) +
                 ", \"opcode\": " + std::to_string(e.opcode) + ", \"context\": \"" + json_escape(e.context) +
                 "\", \"message\": \"" + json_escape(e.message) + "\"}";
        }
        j += "]}";
        return j;
    }

private:
    mutable std::mutex mu_;
    std::vector<ErrorScopeEntry> entries_;
    std::map<std::string, uint64_t> by_message_;
    uint64_t total_ = 0, entries_dropped_ = 0, messages_dropped_ = 0;
};

}  // namespace playgta6::gpu::native_exec
