// vk_cm_info.cpp -- 枚举 VK_KHR_cooperative_matrix 支持的所有 (M,N,K,类型) 组合。
// 张量核这条路必须先知道硬件到底支持哪些 fragment 形状。
#include <vulkan/vulkan.h>

#include <cstdio>
#include <vector>

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{};
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    VkPhysicalDevice pd = devs[0];
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pd = d;
            break;
        }
    }
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(pd, &p);
    std::printf("device: %s\n", p.deviceName);

    auto fn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fn) {
        std::printf("!! 没有 vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n");
        return 1;
    }
    uint32_t cnt = 0;
    VkResult r = fn(pd, &cnt, nullptr);
    std::printf("查询返回 %d, 组合数 %u\n", (int)r, cnt);
    if (r != VK_SUCCESS || cnt == 0) return 0;
    std::vector<VkCooperativeMatrixPropertiesKHR> props(cnt);
    for (auto& c : props) {
        c = VkCooperativeMatrixPropertiesKHR{
            VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    }
    r = fn(pd, &cnt, props.data());
    if (r != VK_SUCCESS) {
        std::printf("第二次查询失败 %d\n", (int)r);
        return 1;
    }
    auto tname = [](VkComponentTypeKHR t) {
        switch (t) {
            case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
            case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
            case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
            case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
            case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
            case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
            case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
            case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
            default: return "?";
        };
    };
    for (uint32_t i = 0; i < cnt; ++i) {
        const auto& c = props[i];
        std::printf("  M=%2u N=%2u K=%2u   A=%-4s B=%-4s C=%-4s Result=%-4s  saturating=%d scope=%d\n",
                    c.MSize, c.NSize, c.KSize, tname(c.AType), tname(c.BType), tname(c.CType),
                    tname(c.ResultType), (int)c.saturatingAccumulation, (int)c.scope);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
