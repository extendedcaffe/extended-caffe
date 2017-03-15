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

#if defined(MKL2017_SUPPORTED)
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKLBatchNormLayer<Dtype>::~MKLBatchNormLayer() {
  dnnDelete<Dtype>(batchNormFwd);
  dnnDelete<Dtype>(batchNormBwd);

  dnnLayoutDelete<Dtype>(layout_usr_);
  dnnReleaseBuffer<Dtype>(scaleShift_buffer_);
  dnnReleaseBuffer<Dtype>(scaleShift_diff_);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Init(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  eps_ = this->layer_param_.batch_norm_param().eps();
  use_weight_bias_ = this->layer_param_.batch_norm_param().use_weight_bias();
  bias_term_ = this->layer_param_.batch_norm_param().bias_term();
  use_global_stats_ = this->layer_param_.batch_norm_param().use_global_stats();


  // LOG(ERROR) << "BN layer: " << this->layer_param_.name() << " use_weight_bias: " << use_weight_bias_ << ", use_global_stats: " << use_global_stats_ << ", bias_term_: " << bias_term_;
  size_t dim = 4, sizes[4], strides[4];

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  sizes[0] = width_;
  sizes[1] = height_;
  sizes[2] = channels_;
  sizes[3] = num_;

  strides[0] = 1;
  strides[1] = sizes[0];
  strides[2] = sizes[0]*sizes[1];
  strides[3] = sizes[0]*sizes[1]*sizes[2];

  // Names are for debugging only
  fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data->name =    "fwd_top_data      @ " + this->layer_param_.name();
  bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->layer_param_.name();
  bwd_top_diff->name =    "bwd_top_diff      @ " + this->layer_param_.name();

  // TODO: Make a cleanup routine to avoid
  // copy of following code in the Destructor

  dnnError_t e;
  dnnLayoutDelete<Dtype>(layout_usr_);
  e = dnnLayoutCreate<Dtype>(&layout_usr_, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  fwd_bottom_data->create_user_layout(dim, sizes, strides, false);
  fwd_top_data   ->create_user_layout(dim, sizes, strides, false);
  bwd_bottom_diff->create_user_layout(dim, sizes, strides, false);
  bwd_top_diff   ->create_user_layout(dim, sizes, strides, false);

  dnnReleaseBuffer<Dtype>(scaleShift_buffer_);
  dnnReleaseBuffer<Dtype>(scaleShift_diff_);
  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.

  // Primitives will be allocated during the first fwd pass
  dnnDelete<Dtype>(batchNormFwd);
  dnnDelete<Dtype>(batchNormBwd);

  // blobs_ layout: 0 is scale, 1 is bias,
  //                2 is mean, 3 is variance, 4 is moving average fraction
  // Matrix: don't flush cache if initialized
  if (blobs_initialized_ && this->blobs_.size() != 0 && channels_ == this->blobs_[0]->shape(0)) {
      // LOG(ERROR) << "use blobs_ cache rather than re-initialize";
      return;
  }

  flags_ = (dnnBatchNormalizationFlag_t)(0);
  // LOG(ERROR) << "use global stats: " << use_global_stats_ << ", use weight bias: " << use_weight_bias_;
  if (use_global_stats_) {
      flags_ = (dnnBatchNormalizationFlag_t)(dnnUseInputMeanVariance | dnnUseScaleShift);
  } else if (use_weight_bias_) {
      flags_ = (dnnBatchNormalizationFlag_t)dnnUseScaleShift;
  }
  // LOG(ERROR) << "blob size to 5";
  this->blobs_.resize(5);

  // Initialize scale and shift
  vector<int> scaleshift_shape(1);
  scaleshift_shape[0] = channels_;

  this->blobs_[0].reset(new Blob<Dtype>(scaleshift_shape));
  FillerParameter filler_param(
          this->layer_param_.batch_norm_param().filler());
  if (!this->layer_param_.batch_norm_param().has_filler()) {
    filler_param.set_type("constant");
    filler_param.set_value(1);
  }
  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
  filler->Fill(this->blobs_[0].get());

  if (this->blobs_.size() > 1) {
      if (bias_term_) {
        this->blobs_[1].reset(new Blob<Dtype>(scaleshift_shape));
        FillerParameter bias_filler_param(
          this->layer_param_.batch_norm_param().bias_filler());
        if (!this->layer_param_.batch_norm_param().has_bias_filler()) {
          bias_filler_param.set_type("constant");
          bias_filler_param.set_value(0);
        }
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
      bias_filler->Fill(this->blobs_[1].get());
    } else {
      this->blobs_[1].reset(new Blob<Dtype>(scaleshift_shape));
      caffe_set(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_cpu_data());
    }
  }

  // Initialize mean, variance and moving average fraction
  if (this->blobs_.size() > 2) {
    mean_.Reshape(scaleshift_shape);
    variance_.Reshape(scaleshift_shape);
    stdvar_.Reshape(scaleshift_shape);
    caffe_set(mean_.count(), Dtype(0), mean_.mutable_cpu_data());
    caffe_set(variance_.count(), Dtype(0), variance_.mutable_cpu_data());
    caffe_set(stdvar_.count(), Dtype(0), stdvar_.mutable_cpu_data());
    this->blobs_[2].reset(new Blob<Dtype>(scaleshift_shape));
    this->blobs_[3].reset(new Blob<Dtype>(scaleshift_shape));
    for (int i = 2; i < 4; i++) {
        caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_data());
    }

    scaleshift_shape[0] = 1;
    this->blobs_[4].reset(new Blob<Dtype>(scaleshift_shape));
    this->blobs_[4]->mutable_cpu_data()[0] = Dtype(1.0);
  }

  blobs_initialized_ = true;
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Init(bottom, top);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bool reshaping = true;
  if ((num_ == bottom[0]->num()) &&
      channels_ == bottom[0]->channels() &&
      height_ == bottom[0]->height() &&
      width_ == bottom[0]->width()) {
    reshaping = false;
  }

  if (bottom[0] == top[0]) {  // in-place computation
    temp_.ReshapeLike(*bottom[0]);
  } else {
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_ = bottom[0]->num();
    top[0]->Reshape(num_, channels_, height_, width_);
  }

  if (reshaping == true) {
    Init(bottom, top);
  }
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  void* bottom_data =
    reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->prv_data()));
  int is_first_pass = 0;

  if (NULL != bottom_data) {
    // Is it the first pass? Create a primitive.
    // LOG(ERROR) << "Using private data in " << this->layer_param_.name();
    if (batchNormFwd == NULL) {
      is_first_pass = 1;

      CHECK((bottom[0]->get_prv_data_descriptor())->get_descr_type() ==
        PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLData<Dtype> >(
           bottom[0]->get_prv_data_descriptor());
      CHECK(mem_descr != NULL);

      /*
      LOG(ERROR) << "Using layout of " << mem_descr->name
              << " as input layout for " << this->layer_param_.name();
      */

      fwd_bottom_data = mem_descr;

      dnnError_t e;
      e = dnnBatchNormalizationCreateForward_v2<Dtype>(
        &batchNormFwd, NULL, mem_descr->layout_int, eps_, flags_);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data   ->create_internal_layout(batchNormFwd, dnnResourceDst);
      bwd_top_diff   ->create_internal_layout(batchNormFwd, dnnResourceDst);
      bwd_bottom_diff->create_internal_layout(batchNormFwd, dnnResourceSrc);

      e = dnnBatchNormalizationCreateBackward_v2<Dtype>(
        &batchNormBwd, NULL, mem_descr->layout_int, eps_, flags_);
      CHECK_EQ(e, E_SUCCESS);
    }
  } else {
    // LOG(ERROR) << "Using cpu_data in " << this->layer_param_.name();
    if (batchNormFwd == NULL) {
      // First pass
      is_first_pass = 1;

      dnnError_t e;
      e = dnnBatchNormalizationCreateForward_v2<Dtype>(
        &batchNormFwd, NULL, layout_usr_, eps_, flags_);
      CHECK_EQ(e, E_SUCCESS);

      e = dnnBatchNormalizationCreateBackward_v2<Dtype>(
        &batchNormBwd, NULL, layout_usr_, eps_, flags_);
      CHECK_EQ(e, E_SUCCESS);
    }
    bottom_data = reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->cpu_data()));
  }
  if (is_first_pass == 1) {
      dnnError_t e;

      dnnLayout_t scaleShift_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<Dtype>(
        &scaleShift_buffer_l, batchNormFwd, dnnResourceScaleShift);
      CHECK_EQ(e, E_SUCCESS);

      e = dnnAllocateBuffer<Dtype>(
        reinterpret_cast<void**>(&scaleShift_buffer_), scaleShift_buffer_l);
      CHECK_EQ(e, E_SUCCESS);

      e = dnnAllocateBuffer<Dtype>(reinterpret_cast<void**>(&scaleShift_diff_), scaleShift_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(scaleShift_buffer_l);

      if (!use_weight_bias_) {
         for (int i = 0; i < channels_; i++) {
            scaleShift_buffer_[i] = 1.0;
            scaleShift_diff_[i] = 0;
            scaleShift_buffer_[channels_ + i] = 0;
            scaleShift_diff_[channels_ + i] = 0;
         }
      }

      if (use_global_stats_) {
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[4]->cpu_data()[0] == 0 ?
          0 : 1 / this->blobs_[4]->cpu_data()[0];
        // LOG(ERROR) << "scale_factor: " << scale_factor << ", mean count: " << mean_.count();
        caffe_cpu_scale(mean_.count(), scale_factor,
          this->blobs_[2]->cpu_data(), mean_.mutable_cpu_data());
        caffe_cpu_scale(variance_.count(), scale_factor,
          this->blobs_[3]->cpu_data(), variance_.mutable_cpu_data());
        // LOG(ERROR) << "mean: ";
        // for (int i = 0; i < mean_.count(); i++) {
        //    LOG(ERROR) << mean_.cpu_data()[i] << ", ";
        // }

        // back propagation scalar
        caffe_copy(variance_.count(), variance_.cpu_data(), stdvar_.mutable_cpu_data());
        caffe_add_scalar(stdvar_.count(), eps_, stdvar_.mutable_cpu_data());
        caffe_powx(stdvar_.count(), stdvar_.cpu_data(), Dtype(0.5), stdvar_.mutable_cpu_data());
        caffe_div(stdvar_.count(), this->blobs_[0]->cpu_data(), stdvar_.cpu_data(), stdvar_.mutable_cpu_data());
      }
  }

  if (use_weight_bias_) {
    // Fill ScaleShift buffer
    for (int i = 0; i < channels_; i++) {
      scaleShift_buffer_[i] = this->blobs_[0]->cpu_data()[i];
      scaleShift_buffer_[channels_ + i] = 0;
      if (bias_term_) {
         scaleShift_buffer_[channels_ + i] = this->blobs_[1]->cpu_data()[i];
      }
    }
  }

  if (bottom[0] == top[0] && this->phase_ == TRAIN) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we skip this if not
    // doing Backward
    caffe_copy(bottom[0]->count(), static_cast<Dtype*>(bottom_data),
                                                      temp_.mutable_cpu_data());
  }

  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber] = {NULL};
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;
  BatchNorm_res[dnnResourceMean] = use_global_stats_ ? mean_.mutable_cpu_data() : NULL;
  BatchNorm_res[dnnResourceVariance] = use_global_stats_ ? variance_.mutable_cpu_data() : NULL;
  // LOG(ERROR) << "mean: " << BatchNorm_res[dnnResourceMean] << ", variance: " << BatchNorm_res[dnnResourceVariance];
  if (fwd_top_data->conversion_needed()) {
    top[0]->set_prv_data_descriptor(fwd_top_data);
    BatchNorm_res[dnnResourceDst] = reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
    BatchNorm_res[dnnResourceDst] = reinterpret_cast<void *>(top[0]->mutable_cpu_data());
    DLOG(INFO) << "Using cpu_data for top in DnnBatchNorm.";
  }

  e = dnnExecute<Dtype>(batchNormFwd, BatchNorm_res);
  CHECK_EQ(e, E_SUCCESS);

#if DUMP_LAYER_IO
  if (1) {
    LOG(ERROR) << this->layer_param_.name();
    FILE *fp = NULL;
    char dump_name[256] = {0};

#if 1
   // print top diff
   sprintf(dump_name, "./%s_mkl_scaleshift.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < channels_ * 2; n++) {
      fprintf(fp, "%f, ", scaleShift_buffer_[n]);
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print bottom
   sprintf(dump_name, "./%s_mkl_bottom.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < 1; n++) {
     for (int c = 0; c < 1; c++) {
       for (int h = 0; h < 1; h++) {
         for (int w = 0; w < 1; w++) {
            fprintf(fp, "%f, ", bottom[0]->data_at(n, c, h, w));
         }
       }
     }
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif
   if (isnan(bottom[0]->data_at(0, 0, 0, 0)) || bottom[0]->data_at(0, 0, 0, 0) > 1000 || bottom[0]->data_at(0, 0, 0, 0) < -1000) {
     LOG(ERROR) << "bottom abnormal";
     exit(-1);
   }
   if (isnan(top[0]->data_at(0, 0, 0, 0)) || top[0]->data_at(0, 0, 0, 0) > 1000 || top[0]->data_at(0, 0, 0, 0) < -1000) {
     LOG(ERROR) << "top abnormal";
     exit(-1);
   }
  }
#endif

}

#define ENABLE_MKL_BWDBN
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
#ifdef ENABLE_MKL_BWDBN
  void *bottom_data = NULL;
  if (bottom[0] == top[0]) {
    bottom_data = reinterpret_cast<void *>(
                        const_cast<Dtype*>(temp_.cpu_data()));
  } else {
    bottom_data =
            reinterpret_cast<void *>(
                        const_cast<Dtype*>(bottom[0]->prv_data()));
    if (NULL == bottom_data) {
	  // LOG(ERROR) << "use cpu bottom data";
      bottom_data =
            reinterpret_cast<void *>(
                        const_cast<Dtype*>(bottom[0]->cpu_data()));
    } else {
	// LOG(ERROR) << "use prv bottom data";
      }
  }

  for (int i = 0; i < channels_; i++) {
    scaleShift_diff_[i] = 0;
    scaleShift_diff_[channels_ + i] = 0;
  }

  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber] = {NULL};
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;
  BatchNorm_res[dnnResourceDiffScaleShift] = scaleShift_diff_;
  BatchNorm_res[dnnResourceMean] = use_global_stats_ ? mean_.mutable_cpu_data() : NULL;
  BatchNorm_res[dnnResourceVariance] = use_global_stats_ ? variance_.mutable_cpu_data() : NULL;

  BatchNorm_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(top[0], true);
  // LOG(ERROR) << this->layer_param_.name() << " diff dst is " << BatchNorm_res[dnnResourceDiffDst];
  if (bwd_bottom_diff->conversion_needed()) {
    // LOG(ERROR) << this->layer_param_.name() << " use prv diff";
    bottom[0]->set_prv_diff_descriptor(bwd_bottom_diff);
    BatchNorm_res[dnnResourceDiffSrc] = bottom[0]->mutable_prv_diff();
  } else {
    // LOG(ERROR) << this->layer_param_.name() << " directly use cpu diff";
    BatchNorm_res[dnnResourceDiffSrc] = reinterpret_cast<void *>(bottom[0]->mutable_cpu_diff());
  }

  e = dnnExecute<Dtype>(batchNormBwd, BatchNorm_res);
  CHECK_EQ(e, E_SUCCESS);

  if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
    // Store ScaleShift blobs
    LOG(ERROR) << "BN layer: " << this->layer_param_.name() << " need weight propagation";
    Dtype* diff_scale = this->blobs_[0]->mutable_cpu_diff();
    Dtype* diff_shift = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < channels_; i++) {
      diff_scale[i] =  scaleShift_diff_[i];
      diff_shift[i] =  0;
      if (bias_term_) {
         diff_shift[i] = scaleShift_diff_[channels_ + i];
      }
    }
  }
#else
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* coef = stdvar_.cpu_data();
    const int pixels_per_plane = top[0]->width() * top[0]->height();
    const int pixels_per_image = pixels_per_plane * top[0]->channels();
    // LOG(ERROR) << "BN num: " << top[0]->num() << " channels: " << top[0]->channels();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
    for (int n = 0; n < top[0]->num(); ++n) {
#ifdef _OPENMP
#pragma omp for
#endif
      for (int c = 0; c < top[0]->channels(); ++c) {
        const int offset = n * pixels_per_image + c * pixels_per_plane;
        caffe_cpu_scale(pixels_per_plane, coef[c], top_diff + offset, bottom_diff + offset);
      }
    }
#ifdef _OPENMP
    }
#endif
  }
#endif

#if DUMP_LAYER_IO
  if (1) {
    LOG(ERROR) << this->layer_param_.name();
    FILE *fp = NULL;
    char dump_name[256] = {0};


#if 1
   // print bottom
   sprintf(dump_name, "./%s_mkl_bottom_bwd.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < 1; n++) {
     for (int c = 0; c < 1; c++) {
       for (int h = 0; h < 1; h++) {
         for (int w = 0; w < 1; w++) {
            fprintf(fp, "%f, ", bottom[0]->data_at(n, c, h, w));
         }
       }
     }
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print mean
   sprintf(dump_name, "./%s_mkl_mean_bwd.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < mean_.count(); n++) {
      fprintf(fp, "%f, ", mean_.cpu_data()[n]);
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print variance
   sprintf(dump_name, "./%s_mkl_variance_bwd.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < variance_.count(); n++) {
      fprintf(fp, "%f, ", variance_.cpu_data()[n]);
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print scaleshift data
   sprintf(dump_name, "./%s_mkl_scaleshift_bwd.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < channels_ * 2; n++) {
      fprintf(fp, "%f, ", scaleShift_buffer_[n]);
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print scaleshift diff
   sprintf(dump_name, "./%s_mkl_scaleshift_diff.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < channels_ * 2; n++) {
      fprintf(fp, "%f, ", scaleShift_diff_[n]);
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print top diff
   sprintf(dump_name, "./%s_mkl_top_diff.txt", this->layer_param_.name().c_str());
   fp = fopen(dump_name, "ab+");
   for (int n = 0; n < 1; n++) {
     for (int c = 0; c < 1; c++) {
       for (int h = 0; h < 1; h++) {
         for (int w = 0; w < 1; w++) {
            fprintf(fp, "%f, ", top[0]->diff_at(n, c, h, w));
         }
       }
     }
   }
   fprintf(fp, "\n");
   fclose(fp);
   fp = NULL;
#endif

#if 1
   // print bottom diff
   sprintf(dump_name, "./%s_mkl_bottom_diff.txt", this->layer_param_.name().c_str());
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
#endif
  }
#endif

}


#ifdef CPU_ONLY
STUB_GPU(MKLBatchNormLayer);
#else
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLBatchNormLayer);
// REGISTER_LAYER_CLASS(MKLBatchNorm);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
