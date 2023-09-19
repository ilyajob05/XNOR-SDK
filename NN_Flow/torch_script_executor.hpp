#ifndef TORCH_SCRIPT_EXECUTOR_HPP
#define TORCH_SCRIPT_EXECUTOR_HPP

#include <memory>
#include <iostream>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

class TSE
{
    public:
        TSE(const char* script_path, float th = 0.0f)
        {
            try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                this->module = torch::jit::load(script_path);
            } catch (const c10::Error& e) {
                std::cerr << "Error loading the model" << std::endl;
            }

            std::cout << "Model loaded fine\n" << std::endl;
        }

        int predict(at::Tensor input)
        {
            // vector of inputs.
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            at::Tensor output = module.forward(inputs).toTensor();
            int num = output.argmax(1).item().toInt();
            return num;
        }

    private:
        // script module
        torch::jit::script::Module module;

};


#endif // TORCH_SCRIPT_EXECUTOR_HPP
