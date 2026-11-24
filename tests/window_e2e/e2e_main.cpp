// O29 end-to-end driver (tools/run_window_e2e.py, wasm64): boots the program that
// `playgta6-recomp build-program --memory-model window` made from guest.c - its descriptor
// selects the memory model - from the program's PGIM image, then prints the boot summary, the
// report's memory section and the guest's exit code (0x600D when every check in guest.c passed).

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "guest/boot.h"
#include "guest/program_loader.h"

extern "C" const Pg6Program pg6_program;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: e2e <main/image/guest.pgim>\n");
        return 64;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 64;
    }
    std::vector<uint8_t> image;
    uint8_t buf[65536];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        image.insert(image.end(), buf, buf + n);
    }
    std::fclose(f);
    playgta6::guest::ProgramBoot pb;
    const playgta6::guest::ProgramError e = playgta6::guest::prepare_program_boot(
        pg6_program, playgta6::guest::memory_image_source({{"main", image}}), {}, pb);
    if (!e.ok()) {
        std::fprintf(stderr, "prepare_program_boot: %s: %s\n", playgta6::guest::program_error_name(e.code),
                     e.detail.c_str());
        return 2;
    }
    const playgta6::guest::BootResult r = playgta6::guest::run_boot(pb.config);
    std::printf("%s\n", r.summary.c_str());
    const std::string& j = r.report_json;
    const size_t m = j.find("\"memory\"");
    if (m != std::string::npos) {
        std::printf("%s\n", j.substr(m, j.find('\n', m) - m).c_str());
    }
    std::printf("exit code 0x%x\n", static_cast<unsigned>(r.stats.exit_code));
    return r.stats.exit_code == 0x600D ? 0 : 1;
}
