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

namespace caffe {

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
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    CHECK_GT(output_dim, 0) << "Output dim should be greater than 0.";
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // If we have more threads available than batches to be prcessed then
  // we are wasting resources (lower batches than 36 on XeonE5)
  // So we instruct MKL
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
        this->forward_cpu_gemm(bottom_data + n*this->bottom_dim_,
                               weight,
                               top_data + n*this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
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
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
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

#ifdef USE_MLSL
  this->on_delinp_ready(propagate_down);
#endif /* USE_MLSL */

}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
