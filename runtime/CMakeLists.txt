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
    os/xgameruntime.cpp
    os/xbox_offline.cpp
    os/xbox_scarlett.cpp
    os/xbox_middleware.cpp
    os/win32_loader.cpp
    os/guest_heap.cpp
    os/ucrt_bridge.cpp
    os/shim_exports.cpp
    os/image_loader.cpp
    os/winsock_ordinals.cpp
    gpu/cmd_encoder.cpp
    gpu/pso_cache.cpp
    gpu/descriptor_heap.cpp
    gpu/state_bridge.cpp
    guest/guest_runtime.cpp
    guest/guest_exceptions.cpp
    guest/com_bridge.cpp
    guest/guest_callbacks.cpp
    guest/guest_crt_exit.cpp
    guest/cpu_helpers.cpp
    guest/boot.cpp
    os/xgameruntime_provider.cpp
)
target_include_directories(playgta6_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(EMSCRIPTEN)
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
    find_package(Threads REQUIRED)
    target_link_libraries(playgta6_core PUBLIC Threads::Threads)

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

    add_executable(test_image_loader ../tests/test_image_loader.cpp)
    target_link_libraries(test_image_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_image_loader_tests COMMAND test_image_loader)

    add_executable(test_pso_cache ../tests/test_pso_cache.cpp)
    target_link_libraries(test_pso_cache PRIVATE playgta6_core)
    add_test(NAME playgta6_pso_cache_tests COMMAND test_pso_cache)

    add_executable(test_descriptor_heap ../tests/test_descriptor_heap.cpp)
    target_link_libraries(test_descriptor_heap PRIVATE playgta6_core)
    add_test(NAME descriptor_heap_tests COMMAND test_descriptor_heap)

    add_executable(test_state_bridge ../tests/test_state_bridge.cpp)
    target_link_libraries(test_state_bridge PRIVATE playgta6_core)
    target_compile_definitions(test_state_bridge PRIVATE
        PG6_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")
    add_test(NAME state_bridge_tests COMMAND test_state_bridge)

    add_executable(test_win32_compat ../tests/test_win32_compat.cpp)
    target_link_libraries(test_win32_compat PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_compat_tests COMMAND test_win32_compat)

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
endif()
