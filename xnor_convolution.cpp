#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


// at::Tensor convolution(
//     const at::Tensor& input,
//     const at::Tensor& weight,
//     const at::Tensor& bias,
//     c10::ArrayRef<int64_t> stride,
//     c10::ArrayRef<int64_t> padding,
//     c10::ArrayRef<int64_t> dilation,
//     int64_t groups) 
//     {
//         auto output = torch::matmul(input, weight.t()) + bias;
//         return output;
//     }



//
//torch::Tensor convolution(
//    const torch::Tensor& input,
//    const torch::Tensor& weight,
//    int stride,
//    int padding)
//{
//    int batch_size = input.size(0);
//    int in_channels = input.size(1);
//    int input_height = input.size(2);
//    int input_width = input.size(3);
//    int out_channels = weight.size(0);
//    int kernel_size_h = weight.size(2);
//    int kernel_size_w = weight.size(3);
//    int output_height = (input_height - kernel_size_h + 2 * padding) / stride + 1;
//    int output_width = (input_width - kernel_size_w + 2 * padding) / stride + 1;
//    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width});
//
//    for (int b = 0; b < batch_size; b++) {
//        for (int oc = 0; oc < out_channels; oc++) {
//            for (int oh = 0; oh < output_height; oh++) {
//                for (int ow = 0; ow < output_width; ow++) {
//                    float sum = 0.0;
//
//                    for (int ic = 0; ic < in_channels; ic++) {
//                        for (int kh = 0; kh < kernel_size_h; kh++) {
//                            for (int kw = 0; kw < kernel_size_w; kw++) {
//                                int ih = oh * stride + kh - padding;
//                                int iw = ow * stride + kw - padding;
//
//                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
//                                    sum += input[b][ic][ih][iw].item<float>() * weight[oc][ic][kh][kw].item<float>();
//                                }
//                            }
//                        }
//                    }
//
//                    output[b][oc][oh][ow] = sum;
//                }
//            }
//        }
//    }
//    return output;
//}
//



torch::Tensor convolution(
    const torch::Tensor input_,
    const torch::Tensor weight_,
    int stride,
    int padding)
{
    const auto input = input_.accessor<float, 4>();
    const auto weight = weight_.accessor<float, 4>();
    //    const auto input = input_.to(torch::kFloat32);
    //    const auto weight = weight_.to(torch::kFloat32);
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size_h = weight.size(2);
    int kernel_size_w = weight.size(3);
    int output_height = (input_height - kernel_size_h + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size_w + 2 * padding) / stride + 1;
    torch::Tensor output_ = torch::zeros({batch_size, out_channels, output_height, output_width}, torch::dtype(torch::kFloat32));
    auto output = output_.accessor<float, 4>();

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    float sum = 0.0;

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size_h; kh++) {
                            for (int kw = 0; kw < kernel_size_w; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    sum += input[b][ic][ih][iw] * weight[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    output[b][oc][oh][ow] = sum;
                }
            }
        }
    }
    return output_;
}




torch::Tensor convolution_v(
    const torch::Tensor input_,
    const torch::Tensor weight_,
    int stride,
    int padding)
{
    const auto input = input_.to(torch::kFloat32).contiguous();
    const auto weight = weight_.to(torch::kFloat32).contiguous();
    auto input_ptr = input.data_ptr<float>();
    float* weight_ptr;
    weight_ptr = weight.data_ptr<float>();
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size_h = weight.size(2);
    int kernel_size_w = weight.size(3);
    int output_height = (input_height - kernel_size_h + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size_w + 2 * padding) / stride + 1;
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, torch::dtype(torch::kFloat32)).contiguous();
    auto output_ptr = output.data_ptr<float>();

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    float sum = 0.0;

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size_h; kh++) {
                            for (int kw = 0; kw < kernel_size_w; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    const int index_i1 = ih * iw + iw;
                                    const int index_i2 = ic * index_i1 + index_i1;
                                    const int index_i3 = b * index_i2 + index_i2;
                                    const int index_w1 = kh * kw + kw;
                                    const int index_w2 = ic * index_w1 + index_w1;
                                    const int index_w3 = oc * index_w2 + index_w2;
                                    //  sum += input_ptr[b][ic][ih][iw] * weight_ptr[oc][ic][kh][kw];
                                    sum += input_ptr[index_i3] * weight_ptr[index_w3];
                                }
                            }
                        }
                    }

                    const int index_o1 = oh * ow + ow;
                    const int index_o2 = oc * index_o1 + index_o1;
                    const int index_o3 = b * index_o2 + index_o2;
                    output_ptr[index_o3] = sum;
                    //                    output_ptr[b][oc][oh][ow] = sum;
                }
            }
        }
    }

    return output;
}






std::vector<torch::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding) 
    {
        auto grad_input = torch::matmul(grad_output, weight);
        auto grad_weight = torch::matmul(grad_output.t(), input);
        auto grad_bias = torch::sum(grad_output, {0});
        
        return {grad_input, grad_weight.t(), grad_bias};
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution", &convolution, "convolution");
    m.def("convolution_v", &convolution_v, "convolution_v");
    m.def("convolution_backward", &convolution_backward, "convolution backward");
}
