cmake_minimum_required(VERSION 3.24)
project(playgta6_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(playgta6_core STATIC
    os/os_shim.cpp
    os/vmem.cpp
    os/threading.cpp
    os/game_input.cpp
    os/httpfs_mailbox.cpp
    os/win32_kernel.cpp
    os/xgameruntime.cpp
    os/xbox_offline.cpp
    os/win32_loader.cpp
    gpu/cmd_encoder.cpp
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
        -sENVIRONMENT=worker
        -sERROR_ON_UNDEFINED_SYMBOLS=0
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
endif()
