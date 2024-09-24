#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/serialize.h>
#include <iostream>
#include <fstream>
#include <memory>

int main() {
    torch::manual_seed(42);

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/mnt/i/project/slam/thirdparty/slam_components/checkpoints/droid_update.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded successfully\n";


    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA is not available, using CPU instead.\n";
        device = torch::Device(torch::kCPU);
    }

    torch::Tensor x1 = torch::randn({1, 1, 128, 1, 1}).to(device);
    torch::Tensor x2 = torch::randn({1, 1, 128, 1, 1}).to(device);
    torch::Tensor x3 = torch::randn({1, 1, 196, 1, 1}).to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x1);
    inputs.push_back(x2);
    inputs.push_back(x3);

    torch::jit::IValue output = module.forward(inputs);
    auto outputs = output.toTuple();

    torch::Tensor output1 = outputs->elements()[1].toTensor();
    torch::Tensor output2 = outputs->elements()[2].toTensor(); 

    std::ofstream out("/mnt/i/project/slam/thirdparty/slam_components/checkpoints/cpp_output2.txt");
    for (int i = 0; i < 2; ++i) {
        out << output1[0][0][0][0][i].item<float>() << " ";
        out << output2[0][0][0][0][i].item<float>() << std::endl;
    }
    out.close();

    return 0;
}
