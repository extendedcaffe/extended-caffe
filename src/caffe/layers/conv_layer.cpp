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

#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "mkldnn.hpp"
#include <stdio.h>


using namespace mkldnn;

namespace caffe {

template <typename Dtype>
int ConvolutionLayer<Dtype>::checkAVX() {
  const int* ker_dims = this->kernel_shape_.cpu_data();
  const int* dil_dims = this->dilation_.cpu_data();
  const int* str_dims = this->stride_.cpu_data();
  int ic = this->channels_;
  int oc = this->num_output_;
  /*
  long threshold = (long)1e12;
  long size = (long)oc * ic * ker_dims[0] * ker_dims[1] * ker_dims[2] *
              ic * ker_dims[0] * ker_dims[1] * ker_dims[2] *
              src_dims[2] * src_dims[3] * src_dims[4];
  */

  bool no_stride = (str_dims[0] == 1) &&
                   (str_dims[1] == 1) && (str_dims[2] == 1);
  bool no_dilation = (dil_dims[0] == 1) &&
                     (dil_dims[1] == 1) && (dil_dims[2] == 1);
  bool no_group = (this->group_ == 1);
  int kernel_size = 1;
  for (int i = 0; i < this->kernel_shape_.count(); i++) {
    kernel_size *= ker_dims[i];
  }
  bool no_1x1x1_kernel = (kernel_size > 1);

  bool ok = true && no_dilation && no_stride && no_1x1x1_kernel &&
            no_group && (ic % 8 == 0) && (oc % 8 == 0);


  if (!ok) {
    return 0;
  } else {
    int type = ((ic % 16 == 0) && (oc % 16 == 0)) ? 1 : 2;
    return type;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom,top);
  const int* bottom_dims = bottom[0]->shape().data();
  for (int i = 0; i < this->num_spatial_axes_ + 2; ++i) {
    src_dims.push_back(bottom_dims[i]);
  }
  if (this->num_spatial_axes_ == 3) {
    useAVX_t = checkAVX();
    // LOG(ERROR) << "Setup for AVX engine: " << useAVX_t;
  } else {
    useAVX_t = 0;
  }
  src_dims.clear();
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Reorder(Dtype* output, Blob<Dtype>* data_blob,
                                      int reorder_t, int useAVX_t, bool reverse, bool isdiff) {
  std::vector<int> dims = data_blob->shape();
  Dtype *input = isdiff ? data_blob->mutable_cpu_diff() : data_blob->mutable_cpu_data();
  int cblk = (useAVX_t == 1) ? 16 : 8;
  int num_slices = this->kernel_shape_.cpu_data()[0];

  switch (reorder_t) {
    case 0:
      // reorder fwd src
      {
        int nblk_size_i = dims[1] * dims[2] * dims[3] * dims[4];
        int cblk_size_i = dims[2] * dims[3] * dims[4];
        int Cblk_size_i = cblk * cblk_size_i;
        int dblk_size_i = dims[3] * dims[4];
        int hblk_size_i = dims[4];

        int nblk_size_o = dims[1] * dims[3] * dims[4];
        int Cblk_size_o = cblk * dims[3] * dims[4];
        int dblk_size_o = dims[0] * dims[1] * dims[3] * dims[4];
        int hblk_size_o = cblk * dims[4];

        #pragma omp parallel for collapse(6) schedule(static)
        for (int d = 0; d < dims[2]; d++) {
          for (int n = 0; n < dims[0]; ++n) {
            for (int C = 0; C < dims[1] / cblk; ++C) {
              for (int h = 0; h < dims[3]; ++h) {
                for (int w = 0; w < dims[4]; ++w) {
                  for (int c = 0; c < cblk; ++c) {
                    int off_i = n * nblk_size_i + C * Cblk_size_i + d * dblk_size_i +
                                h * hblk_size_i + c * cblk_size_i + w;
                    int off_o = d * dblk_size_o + n * nblk_size_o + C * Cblk_size_o +
                                h * hblk_size_o + w * cblk + c;
                    if (!reverse) {
                      output[off_o] = input[off_i];
                    } else {
                      input[off_i] = output[off_o];
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 1:
      // reorder fwd weight
      {
        int oblk_size_i = dims[1] * dims[2] * dims[3] * dims[4];
        int Oblk_size_i = cblk * oblk_size_i;
        int iblk_size_i = dims[2] * dims[3] * dims[4];
        int Iblk_size_i = cblk * iblk_size_i;
        int hblk_size_i = dims[4];
        int dblk_size_i = dims[3] * dims[4];

        int wblk_size_o = cblk * cblk;
        int hblk_size_o = dims[4] * wblk_size_o;
        int Iblk_size_o = dims[3] * hblk_size_o;
        int Oblk_size_o = dims[1] / cblk * Iblk_size_o;
        int dblk_size_o = dims[0] * dims[1] * dims[3] * dims[4];

        #pragma omp parallel for collapse(7) schedule(static)
        for (int d = 0; d < dims[2]; ++d) {
          for (int O = 0; O < dims[0] / cblk; ++O) {
            for (int I = 0; I < dims[1] / cblk; ++I) {
              for (int h = 0; h < dims[3]; ++h) {
                for (int w  = 0; w < dims[4]; ++w) {
                  for (int ic = 0; ic < cblk; ++ic) {
                    for (int oc = 0; oc < cblk; ++oc) {
                      int off_i = O * Oblk_size_i + I * Iblk_size_i + d * dblk_size_i +
                                  h * hblk_size_i + ic * iblk_size_i + oc * oblk_size_i + w;
                      int off_o = d * dblk_size_o + O * Oblk_size_o + I * Iblk_size_o +
                                  h * hblk_size_o + w * wblk_size_o + ic * cblk + oc;
                      if (!reverse) {
                        output[off_o] = input[off_i];
                      } else {
                        input[off_i] = isdiff ? input[off_i] + output[off_o] : output[off_o];
                      }
		    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 2:
      // reorder fwd bias
      for (int oc = 0; oc < dims[0]; oc++) {
        if (!reverse) {
          output[oc] = input[oc] / (Dtype)num_slices;
        } else {
          input[oc] = isdiff ? input[oc] + output[oc] : output[oc];
        }
      }
      break;
    case 3:
      // reorder bwd weight
      {
        int oblk_size_i = dims[1] * dims[2] * dims[3] * dims[4];
        int Oblk_size_i = cblk * oblk_size_i;
        int iblk_size_i = dims[2] * dims[3] * dims[4];
        int Iblk_size_i = cblk * iblk_size_i;
        int hblk_size_i = dims[4];
        int dblk_size_i = dims[3] * dims[4];

        int wblk_size_o = cblk * cblk;
        int hblk_size_o = dims[4] * wblk_size_o;
        int Iblk_size_o = dims[3] * hblk_size_o;
        int Oblk_size_o = dims[1] / cblk * Iblk_size_o;
        int dblk_size_o = dims[0] * dims[1] * dims[3] * dims[4];

        #pragma omp parallel for collapse(7) schedule(static)
        for (int d = 0; d < dims[2]; ++d) {
          for (int O = 0; O < dims[0] / cblk; ++O) {
            for (int I = 0; I < dims[1] / cblk; ++I) {
              for (int h = 0; h < dims[3]; ++h) {
                for (int w  = 0; w < dims[4]; ++w) {
                  for (int oc = 0; oc < cblk; ++oc) {
                    for (int ic = 0; ic < cblk; ++ic) {
                      int off_i = O * Oblk_size_i + I * Iblk_size_i + d * dblk_size_i +
                                  h * hblk_size_i + ic * iblk_size_i + oc * oblk_size_i + w;
                      int off_o = d * dblk_size_o + O * Oblk_size_o + I * Iblk_size_o +
                                  h * hblk_size_o + w * wblk_size_o + oc * cblk + ic;
                      if (!reverse) {
                        output[off_o] = input[off_i];
                      } else {
                        input[off_i] = output[off_o];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    default:
      LOG(ERROR) << "undefined reorder type";
      break;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::ReshapeForMKLdnn(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // get cpu engine
  cpu_engine = new engine(engine::cpu, 0);

  // clear internal memory & primitive
  conv_src_mem.clear();
  conv_weights_mem.clear();
  conv_bias_mem.clear();
  conv_dst_mem.clear();
  sum_dst_mem.clear();

  conv_wgts_bwd_mem.clear();
  conv_wgts_diff_mem.clear();
  conv_bias_diff_mem.clear();
  conv_src_diff_mem.clear();
  conv_dst_diff_mem.clear();
  sum_dst_bwd_data_mem.clear();
  sum_dst_bwd_wgts_mem.clear();
  sum_dst_bwd_bias_mem.clear();

  conv_fwds.clear();
  sums_fwds.clear();

  conv_bwds_data.clear();
  sums_bwds_data.clear();
  conv_bwds_wgts.clear();
  sums_bwds_wgts.clear();
  sums_bwds_bias.clear();

  // data resize
  srcsync = false;

  conv_srcs.resize(bottom.size() * bottom[0]->count());
  conv_dsts.resize(top.size() * top[0]->count());
  conv_weight.resize(this->blobs_[0]->count());
  if (this->bias_term_) {
    conv_bias.resize(this->blobs_[1]->count());
  } else {
    conv_bias.resize(this->num_output_, 0.);
  }

  conv_weight_bwd.resize(this->blobs_[0]->count());
  conv_weight_diff.resize(this->blobs_[0]->count());
  if (this->bias_term_) {
    conv_bias_diff.resize(this->blobs_[1]->count());
  } else {
    conv_bias_diff.resize(this->num_output_, 0.);
  }
  conv_srcs_diff.resize(bottom.size() * bottom[0]->count());
  conv_dsts_diff.resize(top.size() * top[0]->count());

  // 2D memory reshape
  const int* out_dims = this->output_shape_.data();
  const int* ker_dims = this->kernel_shape_.cpu_data();
  const int* str_dims = this->stride_.cpu_data();
  const int* pad_dims = this->pad_.cpu_data();
  int src_slicesize = src_dims[0] * src_dims[1] * src_dims[3] * src_dims[4];
  int wgt_slicesize = this->num_output_ * src_dims[1] * ker_dims[1] * ker_dims[2];
  int dst_slicesize = src_dims[0] * this->num_output_ * out_dims[1] * out_dims[2];
  src_zero_slice.resize(src_slicesize, 0.);
  dst_zero_slice.resize(dst_slicesize, 0.);
  memory::dims conv_src_nchw = {src_dims[0], src_dims[1], src_dims[3], src_dims[4]};
  memory::dims conv_weights_oihw = {this->num_output_, src_dims[1], ker_dims[1], ker_dims[2]};
  memory::dims conv_bias_x = {this->num_output_};
  memory::dims conv_dst_nchw = {src_dims[0], this->num_output_, out_dims[1], out_dims[2]};
  memory::dims conv_strides = {str_dims[1], str_dims[2]};
  memory::dims conv_padding = {pad_dims[1], pad_dims[2]};

  // create memory descriptor
  auto conv_src_md = (useAVX_t == 1) ? memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::nChw16c) :
                                       memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::nChw8c);
  auto conv_weights_md = (useAVX_t == 1) ? memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw16i16o) :
                                           memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw8i8o);
  auto conv_bias_md = memory::desc({conv_bias_x}, memory::data_type::f32, memory::format::x);
  auto conv_dst_md = (useAVX_t == 1) ? memory::desc({conv_dst_nchw}, memory::data_type::f32,memory::format::nChw16c) :
                                       memory::desc({conv_dst_nchw}, memory::data_type::f32,memory::format::nChw8c);
  auto conv_weights_bwd_md = (useAVX_t == 1) ? memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw16o16i) :
                                               memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw8o8i);

  // create convolution descriptor
  auto conv_desc = (this->bias_term_) ?
       convolution_forward::desc(prop_kind::forward, convolution_direct,
                                 conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
                                 conv_strides, conv_padding, conv_padding, padding_kind::zero) :
       convolution_forward::desc(prop_kind::forward, convolution_direct,
                                 conv_src_md, conv_weights_md, conv_dst_md,
                                 conv_strides, conv_padding, conv_padding, padding_kind::zero);
  auto conv_fwd_pd = new convolution_forward::primitive_desc(conv_desc, *cpu_engine);

  auto conv_bwd_data_desc = convolution_backward_data::desc(convolution_direct,
                                                            conv_src_md, conv_weights_bwd_md, conv_dst_md, conv_strides,
                                                            conv_padding, conv_padding, padding_kind::zero);
  auto conv_bwd_data_pd = new convolution_backward_data::primitive_desc(conv_bwd_data_desc, *cpu_engine, *conv_fwd_pd);

  auto conv_bwd_wgts_desc = (this->bias_term_) ?
       convolution_backward_weights::desc(convolution_direct,
                                          conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
                                          conv_strides, conv_padding, conv_padding, padding_kind::zero) :
       convolution_backward_weights::desc(convolution_direct,
                                          conv_src_md, conv_weights_md, conv_dst_md,
                                          conv_strides, conv_padding, conv_padding, padding_kind::zero);
  auto conv_bwd_wgts_pd = new convolution_backward_weights::primitive_desc(conv_bwd_wgts_desc, *cpu_engine, *conv_fwd_pd);

  // create slices sum descriptor
  auto sum_src_pd = conv_fwd_pd->dst_primitive_desc();
  auto sum_dst_md = sum_src_pd.desc();
  vector<memory::primitive_desc> sum_srcs_pd(ker_dims[0], sum_src_pd);
  vector<double> sum_scale(ker_dims[0], 1.0);
  auto sum_fwd_pd = new sum::primitive_desc(sum_dst_md, sum_scale, sum_srcs_pd);

  auto sum_src_bwd_data_pd = conv_bwd_data_pd->diff_src_primitive_desc();
  auto sum_dst_bwd_data_md = sum_src_bwd_data_pd.desc();
  vector<memory::primitive_desc> sum_srcs_bwd_data_pd(ker_dims[0], sum_src_bwd_data_pd);
  auto sum_bwd_data_pd = new sum::primitive_desc(sum_dst_bwd_data_md, sum_scale, sum_srcs_bwd_data_pd);

  auto sum_src_bwd_wgts_pd = conv_bwd_wgts_pd->diff_weights_primitive_desc();
  auto sum_dst_bwd_wgts_md = sum_src_bwd_wgts_pd.desc();
  vector<memory::primitive_desc> sum_srcs_bwd_wgts_pd(bottom.size() * out_dims[0], sum_src_bwd_wgts_pd);
  vector<double> sum_bwd_wgts_scale(bottom.size() * out_dims[0], 1.0);
  auto sum_bwd_wgts_pd = new sum::primitive_desc(sum_dst_bwd_wgts_md, sum_bwd_wgts_scale, sum_srcs_bwd_wgts_pd);

  auto sum_src_bwd_bias_pd = (this->bias_term_) ? conv_bwd_wgts_pd->diff_bias_primitive_desc() :
                                                 conv_bwd_wgts_pd->diff_weights_primitive_desc();
  auto sum_dst_bwd_bias_md = sum_src_bwd_bias_pd.desc();
  vector<memory::primitive_desc> sum_srcs_bwd_bias_pd(bottom.size() * out_dims[0], sum_src_bwd_bias_pd);
  vector<double> sum_bwd_bias_scale(bottom.size() * ker_dims[0] * out_dims[0], 1.0);
  auto sum_bwd_bias_pd = new sum::primitive_desc(sum_dst_bwd_bias_md, sum_bwd_bias_scale, sum_srcs_bwd_bias_pd);

  // init forward memory & pipeline
  pipeline_fwd.clear();
  auto src_idx = [&](int od, int kd){ return od * str_dims[0] + kd - pad_dims[0]; };

  for (int i = 0; i < bottom.size(); ++i) {
    for (int od = 0; od < out_dims[0]; ++od) {
      vector<primitive::at> sum_inputs;
      for (int kd = 0; kd < ker_dims[0]; ++kd) {
        int id = src_idx(od, kd);
        auto conv_src_tmp = (id >= 0 && id < src_dims[2]) ?
            memory({conv_src_md, *cpu_engine}, conv_srcs.data() + i * bottom[0]->count() + id * src_slicesize):
            memory({conv_src_md, *cpu_engine}, src_zero_slice.data());
        conv_src_mem.push_back(conv_src_tmp);
        conv_weights_mem.push_back(memory({conv_weights_md, *cpu_engine},
                                   conv_weight.data() + kd * wgt_slicesize));
        if (this->bias_term_) {
          conv_bias_mem.push_back(memory({conv_bias_md, *cpu_engine}, conv_bias.data()));
        }
        conv_dst_mem.push_back(memory(conv_fwd_pd->dst_primitive_desc()));
        auto conv_fwd = (this->bias_term_) ?
             convolution_forward(*conv_fwd_pd, conv_src_mem.back(),
                                 conv_weights_mem.back(), conv_bias_mem.back(), conv_dst_mem.back()) :
             convolution_forward(*conv_fwd_pd, conv_src_mem.back(),
                                 conv_weights_mem.back(), conv_dst_mem.back());
        conv_fwds.push_back(conv_fwd);
        sum_inputs.push_back(conv_dst_mem.back());
        pipeline_fwd.push_back(conv_fwds.back());
      }
      sum_dst_mem.push_back(memory(sum_fwd_pd->dst_primitive_desc(),
                            conv_dsts.data() + i * top[0]->count() + od * dst_slicesize));
      sums_fwds.push_back(sum(*sum_fwd_pd, sum_inputs, sum_dst_mem.back()));
      pipeline_fwd.push_back(sums_fwds.back());
    }
  }

  // init backward data memory & pipeline
  pipeline_bwd_data.clear();
  auto dst_idx = [&](int id, int kd) {
    return (id+pad_dims[0] - kd) % str_dims[0] ? -1 : (id + pad_dims[0] - kd) / str_dims[0];
  };

  for (int i = 0; i < bottom.size(); ++i) {
    for (int id = 0; id < src_dims[2]; ++id) {
      vector<primitive::at> sum_inputs;
      for (int kd = 0; kd < ker_dims[0]; ++kd) {
        int od = dst_idx(id, kd);
        auto conv_dst_diff_tmp = (od >= 0 && od < out_dims[0]) ?
            memory({conv_dst_md, *cpu_engine},
            conv_dsts_diff.data() + i * top[0]->count() + od * dst_slicesize):
            memory({conv_dst_md, *cpu_engine}, dst_zero_slice.data());
        conv_src_diff_mem.push_back(memory(conv_bwd_data_pd->diff_src_primitive_desc()));
        conv_wgts_bwd_mem.push_back(memory({conv_weights_bwd_md, *cpu_engine},
                                    conv_weight_bwd.data() + kd * wgt_slicesize));
        conv_dst_diff_mem.push_back(conv_dst_diff_tmp);
        conv_bwds_data.push_back(convolution_backward_data(*conv_bwd_data_pd,
                                                           conv_dst_diff_mem.back(),
                                                           conv_wgts_bwd_mem.back(),
                                                           conv_src_diff_mem.back()));
        sum_inputs.push_back(conv_src_diff_mem.back());
        pipeline_bwd_data.push_back(conv_bwds_data.back());
      }
      sum_dst_bwd_data_mem.push_back(memory(sum_bwd_data_pd->dst_primitive_desc(),
                                     conv_srcs_diff.data() + i * bottom[0]->count() + id * src_slicesize));
      sums_bwds_data.push_back(sum(*sum_bwd_data_pd, sum_inputs, sum_dst_bwd_data_mem.back()));
      pipeline_bwd_data.push_back(sums_bwds_data.back());
    }
  }

  // init backward wgts and bias memory & pipeline
  pipeline_bwd_wgts.clear();

  vector<primitive::at> sum_bias_inputs;
  for (int kd = 0; kd < ker_dims[0]; ++kd) {
    vector<primitive::at> sum_wgts_inputs;
    for (int i = 0; i < bottom.size(); ++i) {
      for (int od = 0; od < out_dims[0]; ++od) {
        int id = src_idx(od, kd);
        auto conv_src_tmp = (id >= 0 && id < src_dims[2]) ?
            memory({conv_src_md, *cpu_engine},
            conv_srcs.data() + i * bottom[0]->count() + id * src_slicesize):
            memory({conv_src_md, *cpu_engine}, src_zero_slice.data());
        conv_src_mem.push_back(conv_src_tmp);
        conv_dst_diff_mem.push_back(memory({conv_dst_md, *cpu_engine},
            conv_dsts_diff.data() + i * top[0]->count() + od * dst_slicesize));
        conv_wgts_diff_mem.push_back(memory(conv_bwd_wgts_pd->diff_weights_primitive_desc()));
        if (this->bias_term_) {
          conv_bias_diff_mem.push_back(memory(conv_bwd_wgts_pd->diff_bias_primitive_desc()));
        }
        auto conv_bwd_wgts = (this->bias_term_) ?
             convolution_backward_weights(*conv_bwd_wgts_pd,
             conv_src_mem.back(), conv_dst_diff_mem.back(),
             conv_wgts_diff_mem.back(), conv_bias_diff_mem.back()) :
             convolution_backward_weights(*conv_bwd_wgts_pd,
             conv_src_mem.back(), conv_dst_diff_mem.back(),
             conv_wgts_diff_mem.back());
        conv_bwds_wgts.push_back(conv_bwd_wgts);
        if (this->bias_term_) {
          sum_bias_inputs.push_back(conv_bias_diff_mem.back());
        }
        sum_wgts_inputs.push_back(conv_wgts_diff_mem.back());
        pipeline_bwd_wgts.push_back(conv_bwds_wgts.back());
      }
    }
    sum_dst_bwd_wgts_mem.push_back(memory(sum_bwd_wgts_pd->dst_primitive_desc(),
                                          conv_weight_diff.data() + kd * wgt_slicesize));
    sums_bwds_wgts.push_back(sum(*sum_bwd_wgts_pd, sum_wgts_inputs, sum_dst_bwd_wgts_mem.back()));
    pipeline_bwd_wgts.push_back(sums_bwds_wgts.back());
  }
  if (this->bias_term_) {
    sum_dst_bwd_bias_mem.push_back(memory(sum_bwd_bias_pd->dst_primitive_desc(), conv_bias_diff.data()));
    sums_bwds_bias.push_back(sum(*sum_bwd_bias_pd, sum_bias_inputs, sum_dst_bwd_bias_mem.back()));
    pipeline_bwd_wgts.push_back(sums_bwds_bias.back());
  }
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  bool reinitialize = false;
  const int* bottom_dims = bottom[0]->shape().data();
  for (int i = 0; i < this->num_spatial_axes_ + 2; ++i){
    if (src_dims.size()) {
      if (bottom_dims[i] != src_dims[i]) {
        reinitialize = true;
        LOG(ERROR) << "reinitialize";
        break;
      }
    } else {
      reinitialize = true;
    }
  }

  if (reinitialize == true) {
    BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

    src_dims.clear();
    for (int i = 0; i < this->num_spatial_axes_ + 2; ++i) {
      src_dims.push_back(bottom_dims[i]);
    }
    if (useAVX_t != 0) {
      try {
        this->col_buffer_.Reshape(1, 1, 1, 1);
        vector<Dtype>().swap(this->col_buffer_mt_);
        vector<Dtype>().swap(this->weight_diff_mt_);

        ReshapeForMKLdnn(bottom, top);
      } catch (error& e) {
        if (e.status == mkldnn_out_of_memory) {
          LOG(ERROR) << "Out of memory, ";
        }
        LOG(ERROR) << "Error details: " << e.message;
        throw e;
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    CHECK_GT(output_dim, 0) << "Output dim should be greater than 0.";
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_3D(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* weight = this->blobs_[0].get();
  Blob<Dtype>* bias;
  if (this->bias_term_) {
    bias = this->blobs_[1].get();
  } else {
    bias = NULL;
  }

  Reorder(conv_weight.data(), weight, 1, useAVX_t, false, false);

  if (this->bias_term_) {
    Reorder(conv_bias.data(), bias, 2, useAVX_t, false, false);
  }

  for (int i = 0; i < bottom.size(); ++i) {
    Reorder(conv_srcs.data() + i * bottom[0]->count(), bottom[i], 0, useAVX_t, false, false);
  }
  srcsync = true;
  stream(stream::kind::eager).submit(pipeline_fwd).wait();
  for (int i = 0; i < bottom.size(); ++i) {
    Reorder(conv_dsts.data() + i * top[0]->count(), top[i], 0, useAVX_t, true, false);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_data_3D(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* weight = this->blobs_[0].get();

  Reorder(conv_weight_bwd.data(), weight, 3, useAVX_t, false);
  for (int i = 0; i < top.size(); ++i) {
    Reorder(conv_dsts_diff.data() + i * top[0]->count(), top[i], 0, useAVX_t, false, true);
  }
  stream(stream::kind::eager).submit(pipeline_bwd_data).wait();
  for (int i = 0; i < bottom.size(); ++i) {
    Reorder(conv_srcs_diff.data() + i * bottom[0]->count(), bottom[i], 0, useAVX_t, true, true);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_weights_3D(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* weight = this->blobs_[0].get();
  Blob<Dtype>* bias;
  if (this->bias_term_) {
    bias = this->blobs_[1].get();
  } else {
    bias = NULL;
  }

  if (!srcsync) {
    for (int i = 0; i < bottom.size(); ++i) {
      Reorder(conv_srcs.data() + i * bottom[0]->count(), bottom[i], 0, useAVX_t, false, false);
    }
  }
  stream(stream::kind::eager).submit(pipeline_bwd_wgts).wait();
  Reorder(conv_weight_diff.data(), weight, 1, useAVX_t, true, true);
  if (this->bias_term_) {
    Reorder(conv_bias_diff.data(), bias, 2, useAVX_t, true, true);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // If we have more threads available than batches to be prcessed then
  // we are wasting resources (lower batches than 36 on XeonE5)
  // So we instruct MKL
  if (this->num_spatial_axes_ == 3 && useAVX_t != 0) {
    Forward_3D(bottom,top);
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
  #ifdef _OPENMP
      #pragma omp parallel if(this->num_of_threads_ > 1) num_threads(this->num_of_threads_)
  #endif
      {
  #ifdef _OPENMP
        #pragma omp for
  #endif
        for (int n = 0; n < this->num_; ++n) {
          this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                 weight,
                                 top_data + n * this->top_dim_);
          if (this->bias_term_) {
            const Dtype* bias = this->blobs_[1]->cpu_data();
            this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
          }
        }
      }
    }
  }

  // dump conv output
#if DUMP_LAYER_IO
  LOG(ERROR) << this->layer_param_.name();
  FILE *fp = NULL;
  char dump_name[256] = {0};
  std::string layer_name = boost::replace_all_copy(this->layer_param().name(), "/", "-");
  
  // weights
  sprintf(dump_name, "./%s_cpu_weights.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  // LOG(ERROR) << "[" << this->blobs_[0]->shape(0) << ", " << this->blobs_[0]->shape(1) << ", " << this->blobs_[0]->shape(2) << ", " << this->blobs_[0]->shape(3) << "]";
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < this->blobs_[0]->channels(); c++) {
      for (int h = 0; h < this->blobs_[0]->height(); h++) {
        for (int w = 0; w < this->blobs_[0]->width(); w++) {
           fprintf(fp, "%f, ", this->blobs_[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  if (this->bias_term_) {
    sprintf(dump_name, "./%s_cpu_biases.txt", layer_name.c_str());
    fp = fopen(dump_name, "ab+");
    for (int n = 0; n < this->blobs_[1]->count(); n++) {
       fprintf(fp, "%f, ", this->blobs_[1]->cpu_data()[n]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    fp = NULL;
  }

  sprintf(dump_name, "./%s_cpu_bottom.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  for (int n = 0; n < bottom[0]->num(); n++) {
    for (int c = 0; c < bottom[0]->channels(); c++) {
      for (int h = 0; h < this->blobs_[0]->height(); h++) {
        for (int w = 0; w < this->blobs_[0]->width(); w++) {
          fprintf(fp, "%f, ", bottom[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  sprintf(dump_name, "./%s_cpu_top.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  for (int n = 0; n < top[0]->num(); n++) {
    for (int c = 0; c < 1; c++) {
      for (int h = 0; h < 1; h++) {
        for (int w = 0; w < 1; w++) {
          fprintf(fp, "%f, ", top[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  if (isnan(bottom[0]->data_at(0, 0, 0, 0)) || bottom[0]->data_at(0, 0, 0, 0) > 1000 || bottom[0]->data_at(0, 0, 0, 0) < -1000) {
    LOG(ERROR) << "bottom abnormal";
    exit(-1);
  }

  if (isnan(top[0]->data_at(0, 0, 0, 0)) || top[0]->data_at(0, 0, 0, 0) > 1000 || top[0]->data_at(0, 0, 0, 0) < -1000) {
    LOG(ERROR) << "top abnormal";
    exit(-1);
  }
#endif
  // LOG(ERROR) << "forward total takes: " << timer.MicroSeconds() / 1000. << " ms";
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {

  if (this->num_spatial_axes_ == 3 && useAVX_t != 0) {
    Backward_data_3D(bottom,top);
    Backward_weights_3D(bottom,top);
  } else {
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->cpu_diff();
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
        }
      }

      // OpenMP path is using bigger separate buffer to accumulate
      // weight diffs, which are lateron add to weight_diff
      // so bigger buffer (weight_diff_mt) hase to be cleared out
      // before GEMM ops and results has to be summed up after GEMM ops.

      if (this->param_propagate_down_[0]) {
      #ifdef _OPENMP
        if (this->num_of_threads_ > 1) {
          this->clear_weight_mt();
        }
        #pragma omp parallel if(this->num_of_threads_ > 1) num_threads(this->num_of_threads_)
      #endif
        {
      #ifdef _OPENMP
          #pragma omp for
      #endif
          for (int n = 0; n < this->num_; ++n) {
            // gradient w.r.t. weight. Note that we will accumulate diffs.
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                  top_diff + n * this->top_dim_, weight_diff);
          }

      #ifdef _OPENMP
          if (this->num_of_threads_ > 1) {
            this->sum_weight_mt(weight_diff);
          }
      #endif
        }
      }

      if (propagate_down[i]) {
      #ifdef _OPENMP
        #pragma omp parallel if(this->num_of_threads_ > 1) num_threads(this->num_of_threads_)
        {
          #pragma omp for
      #endif
          for (int n = 0; n < this->num_; ++n) {
            // gradient w.r.t. bottom data, if necessary.
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                    bottom_diff + n * this->bottom_dim_);
          }
      #ifdef _OPENMP
        }
      #endif
      }
    }
  }

#if DUMP_LAYER_IO
  LOG(ERROR) << this->layer_param_.name();
  FILE *fp = NULL;
  char dump_name[256] = {0};
  std::string layer_name = boost::replace_all_copy(this->layer_param().name(), "/", "-");
  
  // weights
  sprintf(dump_name, "./%s_cpu_weights_diff.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  // LOG(ERROR) << "[" << this->blobs_[0]->num() << ", " << this->blobs_[0]->channels() << ", " << this->blobs_[0]->height() << ", " << this->blobs_[0]->width() << "]";
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < this->blobs_[0]->channels(); c++) {
      for (int h = 0; h < this->blobs_[0]->height(); h++) {
        for (int w = 0; w < this->blobs_[0]->width(); w++) {
           fprintf(fp, "%f, ", this->blobs_[0]->diff_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  // top diff
  sprintf(dump_name, "./%s_cpu_top_diff.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < 1; c++) {
      for (int h = 0; h < this->blobs_[0]->height(); h++) {
        for (int w = 0; w < this->blobs_[0]->width(); w++) {
           fprintf(fp, "%f, ", top[0]->diff_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  // bottom diff
  sprintf(dump_name, "./%s_cpu_bottom_diff.txt", layer_name.c_str());
  fp = fopen(dump_name, "ab+");
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < 1; c++) {
      for (int h = 0; h < 1; h++) {
        for (int w = 0; w < 1; w++) {
           fprintf(fp, "%f, ", bottom[0]->diff_at(n, c, h, w));
        }
      }
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
  fp = NULL;

  if (isnan(this->blobs_[0]->diff_at(0, 0, 0, 0)) || this->blobs_[0]->diff_at(0, 0, 0, 0) > 1000 || this->blobs_[0]->diff_at(0, 0, 0, 0) < -1000) {
    LOG(ERROR) << "weight diff abnormal";
    exit(-1);
  }

  if (isnan(top[0]->diff_at(0, 0, 0, 0)) || top[0]->diff_at(0, 0, 0, 0) > 1000 || top[0]->diff_at(0, 0, 0, 0) < -1000) {
    LOG(ERROR) << "top diff abnormal";
    exit(-1);
  }

  if (isnan(bottom[0]->diff_at(0, 0, 0, 0)) || bottom[0]->diff_at(0, 0, 0, 0) > 1000 || bottom[0]->diff_at(0, 0, 0, 0) < -1000) {
    LOG(ERROR) << "bottom diff abnormal";
    exit(-1);
  }
#endif
}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
