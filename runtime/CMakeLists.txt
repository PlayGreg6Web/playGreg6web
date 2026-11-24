cmake_minimum_required(VERSION 3.24)
project(playgta6_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(playgta6_core STATIC
    os/os_shim.cpp
    os/vmem.cpp
    os/vmem_backing.cpp
    os/threading.cpp
    os/game_input.cpp
    os/httpfs_mailbox.cpp
    os/win32_kernel.cpp
    os/win32_sync.cpp
    os/win32_compat_file.cpp
    os/win32_compat_sys.cpp
    os/win32_compat_text.cpp
    os/win32_compat_misc.cpp
    os/win32_rtl.cpp
    os/win32_rtl_text.cpp
    os/win32_rtl_file.cpp
    os/xgameruntime.cpp
    os/xasync_provider.cpp
    os/xbox_offline.cpp
    os/xbox_scarlett.cpp
    os/xbox_middleware.cpp
    os/win32_loader.cpp
    os/guest_heap.cpp
    os/ucrt_bridge.cpp
    os/crt_guest.cpp
    os/ucrt_ext.cpp
    os/crt_format.cpp
    os/crt_scan.cpp
    os/shim_exports.cpp
    os/image_loader.cpp
    os/winsock_ordinals.cpp
    gpu/cmd_encoder.cpp
    gpu/pso_cache.cpp
    gpu/descriptor_heap.cpp
    gpu/state_bridge.cpp
    gpu/d3d12_guest_objects.cpp
    gpu/d3d12_xs_exports.cpp
    gpu/d3d12_guest_resources.cpp
    gpu/d3d12_guest_command_list.cpp
    gpu/d3d12_xs_xbox.cpp
    gpu/d3d12_xs_dump.cpp
    gpu/shader_pack_index.cpp
    gpu/stage_reflection.cpp
    gpu/gfx10_descriptors.cpp
    gpu/gfx10_detile.cpp
    gpu/descriptor_words.cpp
    gpu/frame_capture.cpp
    os/xmem_windows.cpp
    os/guest_window.cpp
    guest/guest_runtime.cpp
    guest/guest_exceptions.cpp
    guest/com_bridge.cpp
    guest/guest_callbacks.cpp
    guest/export_thunks.cpp
    guest/guest_crt_exit.cpp
    guest/guest_rtl.cpp
    guest/cpu_helpers.cpp
    guest/guest_memory.cpp
    guest/memory_model.cpp
    guest/boot.cpp
    wasm_boot.cpp
    os/xgameruntime_provider.cpp
    os/xbox_scarlett_guest.cpp
    guest/program_loader.cpp
    guest/shard_dispatch.cpp
    os/xbox_scarlett_guest_input.cpp
    os/acp_hal_guest.cpp
)
target_include_directories(playgta6_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Local-only wasm64 test build (tools/run_wasm_tests.py): keeps the CTest executables below enabled
# under Emscripten instead of building the browser `runtime` target. Off by default.
option(PLAYGTA6_WASM_TESTS "Build the CTest executables as wasm64 + pthreads Node tests (Emscripten only)" OFF)

if(EMSCRIPTEN AND NOT PLAYGTA6_WASM_TESTS)
    add_executable(runtime main.cpp)
    target_link_libraries(runtime PRIVATE playgta6_core)
    set_target_properties(runtime PROPERTIES
        OUTPUT_NAME "runtime"
        SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../web/dist"
    )
    target_compile_options(runtime PRIVATE -pthread -sMEMORY64=1)
    target_link_options(runtime PRIVATE
        -pthread
        -sMEMORY64=1
        -sPTHREAD_POOL_SIZE=4
        -sMODULARIZE=1
        -sEXPORT_NAME=createRuntimeModule
        -sALLOW_MEMORY_GROWTH=1
        # Guest address space (os/vmem.h): linear memory [4 GiB, 16 GiB) is the identity-mapped
        # guest pool, grown on commit; sbrk is fenced below 4 GiB so malloc never enters it.
        -sMAXIMUM_MEMORY=16GB
        -Wl,--wrap=sbrk
        -sENVIRONMENT=worker
        -sERROR_ON_UNDEFINED_SYMBOLS=0
        -lexports.js
        # Pull the lifted-code helpers (guest/guest_runtime.cpp, EMSCRIPTEN_KEEPALIVE) out of the
        # static library even before recompiled code references them.
        -Wl,--undefined=__playgta6_indirect_call
        -Wl,--undefined=__playgta6_trap
        -Wl,--undefined=__playgta6_unimplemented
        -Wl,--undefined=__playgta6_divide_error
        -Wl,--undefined=__playgta6_cpuid
        -Wl,--undefined=__playgta6_rdtsc
        -Wl,--undefined=__playgta6_rdtscp
        -Wl,--undefined=__playgta6_xgetbv
    )
else()
    if(NOT EMSCRIPTEN)
        find_package(Threads REQUIRED)
        target_link_libraries(playgta6_core PUBLIC Threads::Threads)
    endif()

    enable_testing()
    add_executable(runtime_tests ../tests/test_runtime_abi.cpp)
    target_link_libraries(runtime_tests PRIVATE playgta6_core)
    add_test(NAME runtime_abi_tests COMMAND runtime_tests)

    add_executable(os_shim_tests ../tests/test_os_shims.cpp)
    target_link_libraries(os_shim_tests PRIVATE playgta6_core)
    add_test(NAME os_shim_tests COMMAND os_shim_tests)

    add_executable(test_xgameruntime ../tests/test_xgameruntime.cpp)
    target_link_libraries(test_xgameruntime PRIVATE playgta6_core)
    add_test(NAME playgta6_xgameruntime_tests COMMAND test_xgameruntime)

    add_executable(test_xbox_offline ../tests/test_xbox_offline.cpp)
    target_link_libraries(test_xbox_offline PRIVATE playgta6_core)
    add_test(NAME playgta6_xbox_offline_tests COMMAND test_xbox_offline)

    add_executable(test_win32_loader ../tests/test_win32_loader.cpp)
    target_link_libraries(test_win32_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_loader_tests COMMAND test_win32_loader)

    add_executable(test_shim_exports ../tests/test_shim_exports.cpp)
    target_link_libraries(test_shim_exports PRIVATE playgta6_core)
    add_test(NAME playgta6_shim_exports_tests COMMAND test_shim_exports)

    add_executable(test_shim_audit ../tests/test_shim_audit.cpp)
    target_link_libraries(test_shim_audit PRIVATE playgta6_core)
    add_test(NAME playgta6_shim_audit_tests COMMAND test_shim_audit)

    add_executable(test_crt_format ../tests/test_crt_format.cpp)
    target_link_libraries(test_crt_format PRIVATE playgta6_core)
    add_test(NAME playgta6_crt_format_tests COMMAND test_crt_format)

    add_executable(test_crt_scan ../tests/test_crt_scan.cpp)
    target_link_libraries(test_crt_scan PRIVATE playgta6_core)
    add_test(NAME playgta6_crt_scan_tests COMMAND test_crt_scan)

    add_executable(test_xmem_hooks ../tests/test_xmem_hooks.cpp)
    target_link_libraries(test_xmem_hooks PRIVATE playgta6_core)
    add_test(NAME playgta6_xmem_hooks_tests COMMAND test_xmem_hooks)

    add_executable(test_image_loader ../tests/test_image_loader.cpp)
    target_link_libraries(test_image_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_image_loader_tests COMMAND test_image_loader)

    add_executable(test_pso_cache ../tests/test_pso_cache.cpp)
    target_link_libraries(test_pso_cache PRIVATE playgta6_core)
    add_test(NAME playgta6_pso_cache_tests COMMAND test_pso_cache)

    add_executable(test_descriptor_heap ../tests/test_descriptor_heap.cpp)
    target_link_libraries(test_descriptor_heap PRIVATE playgta6_core)
    add_test(NAME descriptor_heap_tests COMMAND test_descriptor_heap)

    add_executable(test_gfx10_descriptors ../tests/test_gfx10_descriptors.cpp)
    target_link_libraries(test_gfx10_descriptors PRIVATE playgta6_core)
    add_test(NAME gfx10_descriptors_tests COMMAND test_gfx10_descriptors)

    add_executable(test_gfx10_detile ../tests/test_gfx10_detile.cpp)
    target_link_libraries(test_gfx10_detile PRIVATE playgta6_core)
    add_test(NAME test_gfx10_detile COMMAND test_gfx10_detile)

    add_executable(test_console_layout_sizes ../tests/test_console_layout_sizes.cpp)
    target_link_libraries(test_console_layout_sizes PRIVATE playgta6_core)
    add_test(NAME test_console_layout_sizes COMMAND test_console_layout_sizes)

    add_executable(test_stage_reflection ../tests/test_stage_reflection.cpp)
    target_link_libraries(test_stage_reflection PRIVATE playgta6_core)
    add_test(NAME test_stage_reflection COMMAND test_stage_reflection)

    add_executable(test_state_bridge ../tests/test_state_bridge.cpp)
    target_link_libraries(test_state_bridge PRIVATE playgta6_core)
    target_compile_definitions(test_state_bridge PRIVATE
        PG6_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")
    add_test(NAME state_bridge_tests COMMAND test_state_bridge)

    add_executable(test_win32_compat ../tests/test_win32_compat.cpp)
    target_link_libraries(test_win32_compat PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_compat_tests COMMAND test_win32_compat)

    add_executable(test_win32_rtl ../tests/test_win32_rtl.cpp)
    target_link_libraries(test_win32_rtl PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_rtl_tests COMMAND test_win32_rtl)

    add_executable(test_win32_sync ../tests/test_win32_sync.cpp)
    target_link_libraries(test_win32_sync PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_sync_tests COMMAND test_win32_sync)

    add_executable(test_xbox_scarlett ../tests/test_xbox_scarlett.cpp)
    target_link_libraries(test_xbox_scarlett PRIVATE playgta6_core)
    add_test(NAME playgta6_xbox_scarlett_tests COMMAND test_xbox_scarlett)

    add_executable(test_xbox_middleware ../tests/test_xbox_middleware.cpp)
    target_link_libraries(test_xbox_middleware PRIVATE playgta6_core)
    add_test(NAME test_xbox_middleware COMMAND test_xbox_middleware)

    add_executable(test_ucrt_bridge ../tests/test_ucrt_bridge.cpp)
    target_link_libraries(test_ucrt_bridge PRIVATE playgta6_core)
    add_test(NAME playgta6_ucrt_bridge_tests COMMAND test_ucrt_bridge)

    add_executable(test_guest_heap ../tests/test_guest_heap.cpp)
    target_link_libraries(test_guest_heap PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_heap_tests COMMAND test_guest_heap)

    add_executable(test_vmem_contiguous ../tests/test_vmem_contiguous.cpp)
    target_link_libraries(test_vmem_contiguous PRIVATE playgta6_core)
    add_test(NAME playgta6_vmem_contiguous_tests COMMAND test_vmem_contiguous)

    add_executable(test_guest_address_space ../tests/test_guest_address_space.cpp)
    target_link_libraries(test_guest_address_space PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_address_space_tests COMMAND test_guest_address_space)

    add_executable(test_guest_runtime ../tests/test_guest_runtime.cpp)
    target_link_libraries(test_guest_runtime PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_runtime_tests COMMAND test_guest_runtime)

    add_executable(test_guest_exceptions ../tests/test_guest_exceptions.cpp)
    target_link_libraries(test_guest_exceptions PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_exceptions_tests COMMAND test_guest_exceptions)

    add_executable(test_com_bridge ../tests/test_com_bridge.cpp)
    target_link_libraries(test_com_bridge PRIVATE playgta6_core)
    add_test(NAME playgta6_com_bridge_tests COMMAND test_com_bridge)

    add_executable(test_cpu_helpers ../tests/test_cpu_helpers.cpp)
    target_link_libraries(test_cpu_helpers PRIVATE playgta6_core)
    add_test(NAME playgta6_cpu_helpers_tests COMMAND test_cpu_helpers)

    # Native first-boot harness without lifted code (tools/build_native_boot.py links the full one).
    add_executable(playgta6_boot guest/boot_main.cpp)
    target_link_libraries(playgta6_boot PRIVATE playgta6_core)

    add_executable(test_guest_boot ../tests/test_guest_boot.cpp)
    target_link_libraries(test_guest_boot PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_boot_tests COMMAND test_guest_boot)

    add_executable(test_winsock_ordinals ../tests/test_winsock_ordinals.cpp)
    target_link_libraries(test_winsock_ordinals PRIVATE playgta6_core)
    add_test(NAME playgta6_winsock_ordinals_tests COMMAND test_winsock_ordinals)

    add_executable(test_xgameruntime_provider ../tests/test_xgameruntime_provider.cpp)
    target_link_libraries(test_xgameruntime_provider PRIVATE playgta6_core)
    add_test(NAME playgta6_xgameruntime_provider_tests COMMAND test_xgameruntime_provider)

    add_executable(test_guest_crt_exit ../tests/test_guest_crt_exit.cpp)
    target_link_libraries(test_guest_crt_exit PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_crt_exit_tests COMMAND test_guest_crt_exit)

    add_executable(test_guest_callback_unwind ../tests/test_guest_callback_unwind.cpp)
    target_link_libraries(test_guest_callback_unwind PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_callback_unwind_tests COMMAND test_guest_callback_unwind)

    add_executable(test_xbox_scarlett_guest ../tests/test_xbox_scarlett_guest.cpp)
    target_link_libraries(test_xbox_scarlett_guest PRIVATE playgta6_core)
    add_test(NAME playgta6_xbox_scarlett_guest_tests COMMAND test_xbox_scarlett_guest)

    add_executable(test_frame_capture ../tests/test_frame_capture.cpp)
    target_link_libraries(test_frame_capture PRIVATE playgta6_core)
    add_test(NAME playgta6_frame_capture_tests COMMAND test_frame_capture)

    add_executable(test_program_loader ../tests/test_program_loader.cpp)
    target_link_libraries(test_program_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_program_loader_tests COMMAND test_program_loader)

    # R18: links the checked-in output of `playgta6-recomp export-thunks --only ...`
    # (tests/fixtures/export_thunks, kept in sync with the generator by a Rust test).
    add_executable(test_export_thunks
        ../tests/test_export_thunks.cpp
        ../tests/fixtures/export_thunks/pg6_export_thunks.cpp)
    target_include_directories(test_export_thunks PRIVATE ../tests/fixtures/export_thunks)
    target_link_libraries(test_export_thunks PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_export_thunks_tests COMMAND test_export_thunks)

    add_executable(test_xasync_provider ../tests/test_xasync_provider.cpp)
    target_link_libraries(test_xasync_provider PRIVATE playgta6_core)
    add_test(NAME playgta6_xasync_provider_tests COMMAND test_xasync_provider)

    add_executable(test_xmem_windows ../tests/test_xmem_windows.cpp)
    target_link_libraries(test_xmem_windows PRIVATE playgta6_core)
    add_test(NAME playgta6_xmem_windows_tests COMMAND test_xmem_windows)

    add_executable(test_guest_window ../tests/test_guest_window.cpp)
    target_link_libraries(test_guest_window PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_window_tests COMMAND test_guest_window)

    # O29: the checked-in output of `playgta6-recomp export-thunks --memory-model window --only ...`
    # (tests/fixtures/export_thunks_window, kept in sync with the generator by a Rust test).
    add_executable(test_window_marshal
        ../tests/test_window_marshal.cpp
        ../tests/fixtures/export_thunks_window/pg6_export_thunks.cpp)
    target_include_directories(test_window_marshal PRIVATE ../tests/fixtures/export_thunks_window)
    target_link_libraries(test_window_marshal PRIVATE playgta6_core)
    add_test(NAME playgta6_window_marshal_tests COMMAND test_window_marshal)

    add_executable(test_guest_window_e2e ../tests/test_guest_window_e2e.cpp)
    target_link_libraries(test_guest_window_e2e PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_window_e2e_tests COMMAND test_guest_window_e2e)

    add_executable(test_d3d12_xs_exports ../tests/test_d3d12_xs_exports.cpp)
    target_link_libraries(test_d3d12_xs_exports PRIVATE playgta6_core)
    add_test(NAME playgta6_d3d12_xs_exports_tests COMMAND test_d3d12_xs_exports)

    add_executable(test_wasm_boot ../tests/test_wasm_boot.cpp)
    target_link_libraries(test_wasm_boot PRIVATE playgta6_core)
    add_test(NAME test_wasm_boot COMMAND test_wasm_boot)

    add_executable(test_shard_dispatch ../tests/test_shard_dispatch.cpp)
    target_link_libraries(test_shard_dispatch PRIVATE playgta6_core)
    add_test(NAME test_shard_dispatch COMMAND test_shard_dispatch)
endif()

if(EMSCRIPTEN AND PLAYGTA6_WASM_TESTS)
    # Same memory model as the browser runtime (MEMORY64, pthreads, 16 GiB max, sbrk fenced below
    # the guest pool); -fexceptions because test fault handlers throw. The host_* imports that
    # os_shim.cpp declares are resolved by a Node-side JS library instead of the page's glue.
    set(PLAYGTA6_WASM_TEST_JS_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../tools/wasm_test_support/host_imports.js"
        CACHE FILEPATH "JS library providing the env.host_* imports for Node-run wasm64 tests")
    target_compile_options(playgta6_core PUBLIC -pthread -sMEMORY64=1 -fexceptions)
    # The browser's process arena is the whole 12 GiB guest pool, which leaves no room for the extra
    # arenas several tests create next to it (offset-arena, rebase and multi-arena checks): tests
    # get a 4 GiB process arena instead (vmem.h: sizes are build parameters), 8 GiB stay free.
    target_compile_definitions(playgta6_core PUBLIC PLAYGTA6_PROCESS_GUEST_SPACE_SIZE=4294967296ULL)
    get_directory_property(_pg6_wasm_targets BUILDSYSTEM_TARGETS)
    foreach(_t IN LISTS _pg6_wasm_targets)
        get_target_property(_type ${_t} TYPE)
        if(_type STREQUAL "EXECUTABLE")
            set_target_properties(${_t} PROPERTIES SUFFIX ".js")
            target_link_options(${_t} PRIVATE
                -pthread -sMEMORY64=1 -fexceptions
                -sPTHREAD_POOL_SIZE=24
                -sPROXY_TO_PTHREAD=1
                -sEXIT_RUNTIME=1
                -sSTACK_SIZE=1MB
                -sALLOW_MEMORY_GROWTH=1
                -sMAXIMUM_MEMORY=16GB
                -Wl,--wrap=sbrk
                -sENVIRONMENT=node
                -sERROR_ON_UNDEFINED_SYMBOLS=1
                -sNODERAWFS=1  # tests read tests/fixtures from the host, like the native build
                --js-library=${PLAYGTA6_WASM_TEST_JS_LIBRARY}
            )
            set_property(TARGET ${_t} APPEND PROPERTY LINK_DEPENDS "${PLAYGTA6_WASM_TEST_JS_LIBRARY}")
        endif()
    endforeach()
endif()
