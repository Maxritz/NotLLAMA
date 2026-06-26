#include "rdna4_vulkan.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace rdna4 {

bool VulkanContext::init() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RDNA4 LLaMA";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VkResult r = vkCreateInstance(&instInfo, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << r << "\n";
        return false;
    }

    uint32_t pdevCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdevCount, nullptr);
    if (pdevCount == 0) {
        std::cerr << "No Vulkan physical devices found\n";
        return false;
    }

    std::vector<VkPhysicalDevice> pdevs(pdevCount);
    vkEnumeratePhysicalDevices(instance, &pdevCount, pdevs.data());

    for (auto& pdev : pdevs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pdev, &props);
        if (props.vendorID == 0x1002) {
            physicalDevice = pdev;
            std::cout << "Selected AMD GPU: " << props.deviceName << "\n";
            break;
        }
    }
    if (!physicalDevice) {
        physicalDevice = pdevs[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "Selected GPU: " << props.deviceName << " (vendor=" << props.vendorID << ")\n";
    }

    // Find compute queue family with at least 4 queues
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());

    for (uint32_t i = 0; i < qfCount; ++i) {
        if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfProps[i].queueCount >= 4) {
            queueFamilyIndex = i;
            std::cout << "Queue family " << i << " has " << qfProps[i].queueCount << " queues (using 4)\n";
            break;
        }
    }

    // Fallback: any compute queue family
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                std::cout << "Queue family " << i << " has " << qfProps[i].queueCount 
                          << " queues (fallback, using min(4, available))\n";
                break;
            }
        }
    }

    if (queueFamilyIndex == 0xFFFFFFFF) {
        std::cerr << "No compute queue family found\n";
        return false;
    }

    uint32_t queueCount = std::min(4u, qfProps[queueFamilyIndex].queueCount);
    float priorities[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = queueFamilyIndex;
    qInfo.queueCount = queueCount;
    qInfo.pQueuePriorities = priorities;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures = {};
    bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &bdaFeatures;

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.pNext = &features2;

    const char* devExts[] = { VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME };
    devInfo.enabledExtensionCount = 1;
    devInfo.ppEnabledExtensionNames = devExts;

    r = vkCreateDevice(physicalDevice, &devInfo, nullptr, &device);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed: " << r << "\n";
        return false;
    }

    for (uint32_t i = 0; i < queueCount; ++i) {
        vkGetDeviceQueue(device, queueFamilyIndex, i, &queues[i]);
    }
    for (uint32_t i = queueCount; i < 4; ++i) {
        queues[i] = queues[0];  // Duplicate if fewer than 4 queues
    }

    return true;
}

void VulkanContext::cleanup() {
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
}

} // namespace rdna4
