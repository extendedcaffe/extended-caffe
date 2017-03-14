# Add-on Notes
This branch of Caffe extend [Microsoft Caffe](http://github.com/Microsoft/caffe) by integrating [MKL DNN API accelerations](http://github.com/intel/caffe) and doing many extra OpenMP/Zero-Copy optimizations to make py-RFCN fasterin Intel platform. We also implemented the CPU version of some layers(like PSROI, box annotator etc.). We can get 20x acceleration compared with Vanilla CPU Caffe in Pascal-VOC RFCN end2end case in Xeon E5 2699-v4.

## How to use MKL's DNN API
1. go to ./external/mkl and run prepare_mkl.sh. Once it's done, you can see a mklml_lnx_2017.0.2.20161122 folder in ./external/mkl directory
2. Update BLAS_INCLUDE and BLAS_LIB's absolute path part in Makefile.config

## Build
1. make -j8
2. make pycaffe

*This has been tested in CentOS 7.2, suppose no issues in Linux based OS.*

**Contact**: Matrix YAO(yaoweifeng0301@126.com)

---

# Caffe

|  **`Linux (CPU)`**   |  **`Windows (CPU)`** |
|-------------------|----------------------|
| [![Travis Build Status](https://api.travis-ci.org/Microsoft/caffe.svg?branch=master)](https://travis-ci.org/Microsoft/caffe) | [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/58wvckt0rcqtwnr5/branch/master?svg=true)](https://ci.appveyor.com/project/pavlejosipovic/caffe-3a30a) |

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
