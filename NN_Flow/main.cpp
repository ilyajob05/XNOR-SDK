#include <iostream>
#include <vector>
#include <torch/torch.h>




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



using namespace std;

int main()
{
    cout << "Hello World!" << endl;
    auto image_batch = torch::zeros({10, 1, 28, 28});
    auto weights = torch::zeros({32, 1, 3, 3});
    image_batch.fill_(0.5);
    weights.fill_(0.5);
    auto t = convolution(image_batch, weights, 1, 1);
    torch::save(t, "one_t.pt");
    auto t2 = convolution_v(image_batch, weights, 1, 1);
    torch::save(t2, "two_t.pt");
    auto t3 = t - t2;
    print(t3[0]);
    return 0;
}
