/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "mkldnn.hpp"

using namespace mkldnn;

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - dilation (\b optional, default 1). The filter
   *  dilation, given by dilation_size for equal dimensions for different
   *  dilation. By default the convolution has dilation 1.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Convolution"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:
  engine* cpu_engine;
  vector<primitive> pipeline_fwd;
  vector<primitive> pipeline_bwd_data;
  vector<primitive> pipeline_bwd_wgts;

  vector<int> src_dims;
  vector<Dtype> src_zero_slice;
  vector<Dtype> dst_zero_slice;
  vector<Dtype> conv_weight;
  vector<Dtype> conv_bias;
  vector<Dtype> conv_srcs;
  vector<Dtype> conv_dsts;

  vector<Dtype> conv_weight_bwd;
  vector<Dtype> conv_weight_diff;
  vector<Dtype> conv_bias_diff;
  vector<Dtype> conv_srcs_diff;
  vector<Dtype> conv_dsts_diff;

  vector<memory> conv_src_mem;
  vector<memory> conv_weights_mem;
  vector<memory> conv_bias_mem;
  vector<memory> conv_dst_mem;
  vector<memory> sum_dst_mem;

  vector<memory> conv_wgts_bwd_mem;
  vector<memory> conv_wgts_diff_mem;
  vector<memory> conv_bias_diff_mem;
  vector<memory> conv_src_diff_mem;
  vector<memory> conv_dst_diff_mem;
  vector<memory> sum_dst_bwd_data_mem;
  vector<memory> sum_dst_bwd_wgts_mem;
  vector<memory> sum_dst_bwd_bias_mem;

  vector<primitive> conv_fwds;
  vector<primitive> sums_fwds;

  vector<primitive> conv_bwds_data;
  vector<primitive> sums_bwds_data;
  vector<primitive> conv_bwds_wgts;
  vector<primitive> sums_bwds_wgts;
  vector<primitive> sums_bwds_bias;

  int useAVX_t;
  int checkAVX();
  bool srcsync;
  void Reorder(Dtype* output, Blob<Dtype>* data_blob, int reorder_t, int useAVX_t, bool reverse = false, bool isdiff = false);
  void ReshapeForMKLdnn(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void Forward_3D(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void Backward_data_3D(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void Backward_weights_3D(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
};


}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
