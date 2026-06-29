cmake_minimum_required(VERSION 3.25)
project(rdna4_llama LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Use glslc from Vulkan SDK (works with MSVC)
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Set VULKAN_SDK environment variable.")
endif()
message(STATUS "glslc: ${GLSLC}")

set(HOST_SRC
    src/host/context.cpp
    src/host/mailbox.cpp
    src/host/memory.cpp
    src/host/scheduler.cpp
    src/host/weight_uploader.cpp
    src/host/kv_cache.cpp
    src/host/pipeline_builder.cpp
    src/host/tokenizer.cpp
    src/host/inference_engine.cpp
    src/host/profiler.cpp
    src/host/allocator.cpp
    src/host/vgpr_stub.cpp
    main.cpp
)

add_executable(rdna4_llama ${HOST_SRC})
target_include_directories(rdna4_llama PRIVATE include)
target_link_libraries(rdna4_llama PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(rdna4_llama PRIVATE /W3 /EHsc)
    target_compile_definitions(rdna4_llama PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(rdna4_llama PRIVATE -Wall -Wextra)
endif()

set(KERNELS
    src/kernels/gemm.comp
    src/kernels/attention.comp
    src/kernels/rope.comp
    src/kernels/add.comp
    src/kernels/rms_norm.comp
    src/kernels/embed.comp
    src/kernels/kv_cache_write.comp
    src/kernels/silu_mul.comp
)

set(SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})

set(SPV_FILES "")
foreach(kernel ${KERNELS})
    get_filename_component(name ${kernel} NAME_WE)
    set(spv "${SHADER_DIR}/${name}.spv")
    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/${kernel} -o ${spv}
                --target-env=vulkan1.3 -O
        DEPENDS ${CMAKE_SOURCE_DIR}/${kernel}
        COMMENT "Compiling ${name} -> SPIR-V"
    )
    list(APPEND SPV_FILES ${spv})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_FILES})
add_dependencies(rdna4_llama shaders)

# --- test_inference ---
set(TEST_INFERENCE_SRC
    src/host/context.cpp
    src/host/mailbox.cpp
    src/host/memory.cpp
    src/host/scheduler.cpp
    src/host/weight_uploader.cpp
    src/host/kv_cache.cpp
    src/host/pipeline_builder.cpp
    src/host/tokenizer.cpp
    src/host/inference_engine.cpp
    src/host/profiler.cpp
    src/host/allocator.cpp
    src/host/vgpr_stub.cpp
    test_inference.cpp
    cpu_reference.cpp
)

add_executable(test_inference ${TEST_INFERENCE_SRC})
target_include_directories(test_inference PRIVATE include)
target_link_libraries(test_inference PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)
add_dependencies(test_inference shaders)

if(MSVC)
    target_compile_options(test_inference PRIVATE /W3 /EHsc)
    target_compile_definitions(test_inference PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_inference PRIVATE -Wall -Wextra)
endif()

# --- test_cpu_ref (CPU-only) ---
add_executable(test_cpu_ref test_cpu_ref.cpp cpu_reference.cpp)
target_include_directories(test_cpu_ref PRIVATE include)
target_link_libraries(test_cpu_ref PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(test_cpu_ref PRIVATE /W3 /EHsc)
    target_compile_definitions(test_cpu_ref PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_cpu_ref PRIVATE -Wall -Wextra)
endif()

# --- test_turboquant (compile-time assert test) ---
add_executable(test_turboquant test/test_turboquant.cpp)
target_include_directories(test_turboquant PRIVATE include)
if(MSVC)
    target_compile_options(test_turboquant PRIVATE /W3 /EHsc)
    target_compile_definitions(test_turboquant PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_turboquant PRIVATE -Wall -Wextra)
endif()

# --- test_compression (compile-time assert test) ---
add_executable(test_compression test/test_compression.cpp)
target_include_directories(test_compression PRIVATE include)
if(MSVC)
    target_compile_options(test_compression PRIVATE /W3 /EHsc)
    target_compile_definitions(test_compression PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_compression PRIVATE -Wall -Wextra)
endif()

install(TARGETS rdna4_llama DESTINATION bin)
install(FILES ${SPV_FILES} DESTINATION bin/shaders)
