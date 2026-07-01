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
    src/host/turboquant_dispatch.cpp
    src/host/multi_model_manager.cpp
    src/host/model_router.cpp
    src/host/mtp_engine.cpp
    src/host/cli_parser.cpp
    src/host/web_server.cpp
    src/host/agent_node.cpp
    src/host/graphify_client.cpp
    src/loaders/gguf.cpp
    src/engine/vulkan_device.cpp
    src/engine/ring_allocator_adapter.cpp
    src/engine/vulkan_compute_engine.cpp
    src/engine/tokenizer_adapter.cpp
    src/engine/model_adapter.cpp
    src/engine/kv_cache_adapter.cpp
    src/engine/vulkan_shader_library.cpp
    src/engine/vulkan_descriptor_manager.cpp
    src/engine/vulkan_debug_context.cpp
    src/engine/shader_compiler.cpp
    main.cpp
)

# Find threading library
find_package(Threads REQUIRED)

add_executable(rdna4_llama ${HOST_SRC})
target_include_directories(rdna4_llama PRIVATE include src)
target_link_libraries(rdna4_llama PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json Threads::Threads)

if(MSVC)
    target_compile_options(rdna4_llama PRIVATE /W3 /EHsc)
    target_compile_definitions(rdna4_llama PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(rdna4_llama PRIVATE -Wall -Wextra)
endif()

# NOTE: Compute shaders and inference tests are disabled pending rewrite.
# To restore GPU inference, re-enable the KERNELS block and test executables below.
# set(KERNELS ...)
# add_custom_target(shaders ...)
# add_dependencies(rdna4_llama shaders)

# --- test_inference ---
# Disabled: test_inference.cpp and cpu_reference.cpp pending rewrite.
# set(TEST_INFERENCE_SRC ...)
# add_executable(test_inference ...)

# --- test_cpu_ref (CPU-only) ---
# Disabled: test_cpu_ref.cpp and cpu_reference.cpp pending rewrite.
# add_executable(test_cpu_ref test_cpu_ref.cpp cpu_reference.cpp)

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

# --- test_engine (modular interface sanity test) ---
add_executable(test_engine test/test_engine.cpp
    src/host/context.cpp
    src/host/allocator.cpp
    src/engine/vulkan_device.cpp
    src/engine/ring_allocator_adapter.cpp
    src/engine/vulkan_compute_engine.cpp
    src/engine/vulkan_shader_library.cpp
    src/engine/vulkan_descriptor_manager.cpp
    src/engine/vulkan_debug_context.cpp
    src/engine/shader_compiler.cpp
)
target_include_directories(test_engine PRIVATE include)
target_link_libraries(test_engine PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)
if(MSVC)
    target_compile_options(test_engine PRIVATE /W3 /EHsc)
    target_compile_definitions(test_engine PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_engine PRIVATE -Wall -Wextra)
endif()

add_executable(test_shader_compile test/test_shader_compile.cpp
    src/engine/shader_compiler.cpp
)
target_include_directories(test_shader_compile PRIVATE include)
if(MSVC)
    target_compile_options(test_shader_compile PRIVATE /W3 /EHsc)
    target_compile_definitions(test_shader_compile PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

install(TARGETS rdna4_llama DESTINATION bin)
file(GLOB SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/*.comp")
install(FILES ${SHADER_SRC} DESTINATION bin/shaders)

# ============================================================================
# Kernel dispatch test targets (CPU reference + GPU verification)
# ============================================================================

set(KERNEL_TEST_SRC
    src/host/context.cpp
    src/engine/shader_compiler.cpp
)

set(KERNEL_TEST_COMPILE_FLAGS
    /W3 /EHsc
)
set(KERNEL_TEST_DEFINES
    NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
)

function(add_kernel_test NAME)
    add_executable(${NAME} test/${NAME}.cpp ${KERNEL_TEST_SRC})
    target_include_directories(${NAME} PRIVATE include)
    target_link_libraries(${NAME} PRIVATE Vulkan::Vulkan)
    if(MSVC)
        target_compile_options(${NAME} PRIVATE ${KERNEL_TEST_COMPILE_FLAGS})
        target_compile_definitions(${NAME} PRIVATE ${KERNEL_TEST_DEFINES})
    endif()
endfunction()

add_kernel_test(test_rms_norm)
add_kernel_test(test_rope)
add_kernel_test(test_silu_mul)
add_kernel_test(test_embed)

# ============================================================================
# WMMA Inference Engine Targets
# ============================================================================

set(WMMA_SHADERS
    shaders/src/gemm_fp32.comp
    shaders/src/gemm_q4_0.comp
    shaders/src/gemm_q8_0.comp
    shaders/src/convert_f32_f16.comp
    shaders/src/token_embedding.comp
    shaders/src/attention_scores.comp
    shaders/src/attention_softmax.comp
    shaders/src/attention_value.comp
    shaders/src/rms_norm.comp
    shaders/src/rope.comp
    shaders/src/silu_mul.comp
    shaders/src/add.comp
    shaders/src/argmax.comp
)

# Compile inference shaders to SPIR-V
set(WMMA_SPV_DIR "${CMAKE_BINARY_DIR}/shaders_spv")
file(MAKE_DIRECTORY ${WMMA_SPV_DIR})

foreach(SHADER ${WMMA_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_OUT "${WMMA_SPV_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${GLSLC} -c ${CMAKE_SOURCE_DIR}/${SHADER} --target-spv=spv1.3 -o ${SPV_OUT}
        DEPENDS ${CMAKE_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER}"
    )
    list(APPEND WMMA_SPV_FILES ${SPV_OUT})
endforeach()

set(WMMA_SRC
    src/vulkan/vk_device.cpp
    src/vulkan/vk_shader.cpp
    src/vulkan/vk_backend.cpp
    src/vulkan/inference_engine.cpp
)

# --- test_gemm_q4 (Q4_0 on-the-fly dequant GEMM correctness) ---
add_executable(test_gemm_q4
    tests/test_gemm_q4.cpp
    ${WMMA_SRC}
)
target_include_directories(test_gemm_q4 PRIVATE include)
target_link_libraries(test_gemm_q4 PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)
if(MSVC)
    target_compile_options(test_gemm_q4 PRIVATE /W3 /EHsc)
    target_compile_definitions(test_gemm_q4 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_gemm_q4 PRIVATE -Wall -Wextra)
endif()
add_dependencies(test_gemm_q4 wmma_shaders)
add_custom_command(TARGET test_gemm_q4 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${WMMA_SPV_DIR} $<TARGET_FILE_DIR:test_gemm_q4>/shaders_spv
    COMMENT "Copying SPIR-V to test output directory"
)

# --- test_multi_token (autoregressive multi-token inference) ---
add_executable(test_multi_token
    tests/test_multi_token.cpp
    ${WMMA_SRC}
)
target_include_directories(test_multi_token PRIVATE include)
target_link_libraries(test_multi_token PRIVATE Vulkan::Vulkan nlohmann_json::nlohmann_json)
if(MSVC)
    target_compile_options(test_multi_token PRIVATE /W3 /EHsc)
    target_compile_definitions(test_multi_token PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(test_multi_token PRIVATE -Wall -Wextra)
endif()
add_dependencies(test_multi_token wmma_shaders)
add_custom_command(TARGET test_multi_token POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${WMMA_SPV_DIR} $<TARGET_FILE_DIR:test_multi_token>/shaders_spv
    COMMENT "Copying SPIR-V to test output directory"
)

# Custom target for all WMMA shaders
add_custom_target(wmma_shaders ALL DEPENDS ${WMMA_SPV_FILES})
