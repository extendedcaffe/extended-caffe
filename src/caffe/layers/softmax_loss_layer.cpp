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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_MLSL
using namespace MLSL;
#endif /* USE_MLSL */

namespace caffe {

#define _EPSILON Dtype(1e-8)

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

#ifdef USE_MLSL

    int ic = bottom[0]->channels();
    int iw = bottom[0]->width();
    int ih = bottom[0]->height();

    DataType dt = (sizeof(Dtype) == 4)? DT_FLOAT : DT_DOUBLE;
  	ComputeOpRegInfo *myRegInfo;
  	myRegInfo = new ComputeOpRegInfo(COMP_OP_TYPE_EVAL);
  	myRegInfo->SetName(this->layer_param_.name().c_str());
  	myRegInfo->AddInputFeatureMap(ic, iw*ih, dt);
  	myRegInfo->AddInputFeatureMap(bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), dt);

    myRegInfo->Validate();
  	this->layerOp = new ComputeOp(myRegInfo, caffe::internode::data_parallelism);
    delete myRegInfo;

#endif /* USE_MLSL */

}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  // LOG(ERROR) << this->layer_param_.name() << " normalizer is: " << normalizer;
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = Dtype(0);

  if (bottom.size() == 3) {
      const Dtype* weights = bottom[2]->cpu_data();
      Dtype weighted_sum = Dtype(0);
      Dtype weighted_sum_local = Dtype(0);
      Dtype loss_local = Dtype(0);

      for (int i = 0; i < outer_num_; ++i) {
        weighted_sum_local = Dtype(0);
        loss_local = Dtype(0);

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+: loss_local, weighted_sum_local) if(inner_num_ > 1)
        #endif
        for (int j = 0; j < inner_num_; j++) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            continue;
          }

          DCHECK_GE(label_value, 0);
          DCHECK_LT(label_value, prob_.shape(softmax_axis_));
          Dtype p = prob_data[i * dim + label_value * inner_num_ + j];
          loss_local += weights[i * inner_num_ + j] * log(std::max(Dtype(FLT_MIN), std::min(p, Dtype(1.0 - FLT_MIN))));
          weighted_sum_local += weights[i * inner_num_ + j];
        }

        weighted_sum += weighted_sum_local;
        loss -= loss_local;
      }
      // LOG(ERROR) << "outer num: " << outer_num_ << ", inner_num_: " << inner_num_;
      // LOG(ERROR) << "count: " << count << ", loss: " << loss;

      top[0]->mutable_cpu_data()[0] = loss / weighted_sum;
      if (top.size() == 2) {
        top[1]->ShareData(prob_);
      } 
  } else {
      int count_local = 0;
      Dtype loss_local = Dtype(0);

      for (int i = 0; i < outer_num_; ++i) {
        count_local = 0;
        loss_local = Dtype(0);

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+: loss_local, count_local) if(inner_num_ > 1)
        #endif
        for (int j = 0; j < inner_num_; j++) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            continue;
          }

          DCHECK_GE(label_value, 0);
          DCHECK_LT(label_value, prob_.shape(softmax_axis_));
          Dtype p = prob_data[i * dim + label_value * inner_num_ + j];
          loss_local += log(std::max(Dtype(FLT_MIN), std::min(p, Dtype(1.0 - FLT_MIN))));
          ++count_local;
        }

        count += count_local;
        loss -= loss_local;
      }
      // LOG(ERROR) << "outer num: " << outer_num_ << ", inner_num_: " << inner_num_;
      // LOG(ERROR) << "count: " << count << ", loss: " << loss;

      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, outer_num_, inner_num_, count);
      top[0]->mutable_cpu_data()[0] = loss / normalizer;
      if (top.size() == 2) {
        top[1]->ShareData(prob_);
      }
  }

#if DUMP_LAYER_IO
  // LOG(ERROR) << "softmax loss";
  FILE *fp = NULL;
  char dump_name[256] = {0};

  // weights
  if (weights) {
    sprintf(dump_name, "./softmax_loss_weights.txt");
    fp = fopen(dump_name, "wb+");
    for (int i = 0 ; i < outer_num_ * inner_num_; i++) {
       fprintf(fp, "%f, ", weights[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
  }

  // prob
  sprintf(dump_name, "./softmax_loss_prob.txt");
  fp = fopen(dump_name, "wb+");
  Dtype max_prob = -1;
  for (int i = 0; i < prob_.count(); i++) {
       fprintf(fp, "%f, ", prob_data[i]);
       if (max_prob < prob_data[i] || isnan(prob_data[i])) {
         max_prob = prob_data[i];
       }
  }
  fprintf(fp, "\n");
  fclose(fp);

  LOG(ERROR) << "max prob: " << max_prob;

  if (isnan(loss)) {
    LOG(ERROR) << "loss is nan, exit";
    exit(-1);
  }
#endif
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if (bottom.size() == 3) {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* prob_data = prob_.cpu_data();
        caffe_copy(prob_.count(), prob_data, bottom_diff);
        const Dtype* label = bottom[1]->cpu_data();
        int dim = prob_.count() / outer_num_;
        Dtype weight_sum = Dtype(0);
        const Dtype* weights = bottom[2]->cpu_data();
        for (int i = 0; i < outer_num_; ++i) {
          for (int j = 0; j < inner_num_; ++j) {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if (has_ignore_label_ && label_value == ignore_label_) {
              for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
                bottom_diff[i * dim + c * inner_num_ + j] = 0;
              }
            } else {
              bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
              for (int c = 0; c < bottom[0]->shape(1); ++c) {
                bottom_diff[i * dim + c * inner_num_ + j] *= weights[i * inner_num_ + j];
              }
              weight_sum += weights[i * inner_num_ + j];
            }
          }
        }
        // Scale gradient
        Dtype loss_weight = top[0]->cpu_diff()[0] / weight_sum;
        caffe_scal(prob_.count(), loss_weight, bottom_diff); 
    } else {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* prob_data = prob_.cpu_data();
        caffe_copy(prob_.count(), prob_data, bottom_diff);
        const Dtype* label = bottom[1]->cpu_data();
        int dim = prob_.count() / outer_num_;
        int count = 0;
        for (int i = 0; i < outer_num_; ++i) {
          for (int j = 0; j < inner_num_; ++j) {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if (has_ignore_label_ && label_value == ignore_label_) {
              for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
                bottom_diff[i * dim + c * inner_num_ + j] = 0;
              }
            } else {
              bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
              ++count;
            }
          }
        }
        // Scale gradient
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, outer_num_, inner_num_, count);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(prob_.count(), loss_weight, bottom_diff);
    }
}

#if DUMP_LAYER_IO
  FILE *fp = NULL;
  char dump_name[256] = {0};
  bool has_nan = false;

  // LOG(ERROR) << "top diff: " << top[0]->cpu_diff()[0];
  
  // bottom diff
  sprintf(dump_name, "./softmax_loss_botton_diff.txt");
  fp = fopen(dump_name, "wb+");
  Dtype min_diff = 1;
  for (int i = 0 ; i < bottom[0]->count(); i++) {
     fprintf(fp, "%f, ", bottom[0]->cpu_diff()[i]);
     if (isnan(bottom[0]->cpu_diff()[i])) {
       has_nan = true;
     }
     if (min_diff > bottom[0]->cpu_diff()[i]){
       min_diff = bottom[0]->cpu_diff()[i];
     }
  }
  fprintf(fp, "\n");
  fclose(fp);

  LOG(ERROR) << "bottom min diff: " << min_diff;
  if (has_nan) {
    LOG(ERROR) << "bottom diff has nan";
    exit(-1);
  }
#endif
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
