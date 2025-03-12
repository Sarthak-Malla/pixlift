// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/platform/image-decoders/pixlift/super_resolution.h"

// For reading the model file
#include "base/files/file_util.h"
#include "base/logging.h"
#include "base/strings/string_util.h"

// Needed to register built-in TF ops (CONV, etc.)
#include "third_party/tflite/src/tensorflow/lite/kernels/register.h"

namespace blink {
namespace pixlift {

WTF::Vector<float> BitmapToTFLiteTensor(const SkBitmap& bitmap) {
  if (!bitmap.readyToDraw())
    return {};

  const int width = bitmap.width();
  const int height = bitmap.height();
  const int channels = 3;  // we assume 3 channels (RGB)

  WTF::Vector<float> tensor_data;
  tensor_data.resize(width * height * channels);

  SkPixmap pixmap;
  if (!bitmap.peekPixels(&pixmap)) {
    return {};
  }

  SkColorType color_type = bitmap.colorType();
  const uint8_t* src_data = static_cast<const uint8_t*>(pixmap.addr());

  if (color_type == kRGBA_8888_SkColorType ||
      color_type == kBGRA_8888_SkColorType) {
    const bool is_bgra = (color_type == kBGRA_8888_SkColorType);
    const int r_index = is_bgra ? 2 : 0;
    const int g_index = 1;
    const int b_index = is_bgra ? 0 : 2;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint8_t* pixel = src_data + (y * pixmap.rowBytes() + x * 4);
        const int offset = (y * width + x) * channels;

        tensor_data[offset + 0] = pixel[r_index] / 255.0f;
        tensor_data[offset + 1] = pixel[g_index] / 255.0f;
        tensor_data[offset + 2] = pixel[b_index] / 255.0f;
      }
    }
  } else if (color_type == kGray_8_SkColorType) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint8_t gray = src_data[y * pixmap.rowBytes() + x];
        const float normalized = gray / 255.0f;
        const int offset = (y * width + x) * channels;
        tensor_data[offset + 0] = normalized;
        tensor_data[offset + 1] = normalized;
        tensor_data[offset + 2] = normalized;
      }
    }
  } else if (color_type == kRGB_888x_SkColorType) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint8_t* pixel = src_data + (y * pixmap.rowBytes() + x * 4);
        const int offset = (y * width + x) * channels;
        tensor_data[offset + 0] = pixel[0] / 255.0f;
        tensor_data[offset + 1] = pixel[1] / 255.0f;
        tensor_data[offset + 2] = pixel[2] / 255.0f;
      }
    }
  } else {
    // Unsupported format
    return {};
  }

  return tensor_data;
}

std::vector<int> GetTensorDimensionsForBitmap(const SkBitmap& bitmap,
                                              bool include_batch_dim) {
  if (include_batch_dim) {
    return {1, bitmap.height(), bitmap.width(), 3};
  } else {
    return {bitmap.height(), bitmap.width(), 3};
  }
}

bool CopyTensorToInterpreter(const WTF::Vector<float>& tensor_data,
                             const std::vector<int>& dimensions,
                             tflite::Interpreter* interpreter,
                             int tensor_index) {
  if (!interpreter || tensor_data.size() == 0)
    return false;

  TfLiteStatus status = interpreter->ResizeInputTensor(tensor_index, dimensions);
  if (status != kTfLiteOk)
    return false;

  if (interpreter->AllocateTensors() != kTfLiteOk)
    return false;

  TfLiteTensor* tensor = interpreter->tensor(tensor_index);
  if (!tensor || !tensor->data.f)
    return false;

  std::memcpy(tensor->data.f,
              tensor_data.data(),
              tensor_data.size() * sizeof(float));
  return true;
}

// Singleton implementation
SuperResolutionModel* SuperResolutionModel::instance_ = nullptr;

SuperResolutionModel::SuperResolutionModel() {}
SuperResolutionModel::~SuperResolutionModel() {}

SuperResolutionModel* SuperResolutionModel::GetInstance() {
  if (!instance_) {
    instance_ = new SuperResolutionModel();
  }
  return instance_;
}

bool SuperResolutionModel::Initialize() {
  if (IsReady()) {
    return true;  // Already initialized
  }

  // 1) Load the TFLite model from a file path.
  
  const base::FilePath model_path("/pixlift/res/xlsr_quantized.tflite");
  std::string model_data;
  if (!base::ReadFileToString(model_path, &model_data)) {
    LOG(ERROR) << "Failed to read TFLite model from " << model_path.value();
    return false;
  }

  // 2) Build model from buffer
  model_ = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                    model_data.size());
  if (!model_) {
    LOG(ERROR) << "Failed to build TFLite model";
    return false;
  }

  // 3) Create interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    LOG(ERROR) << "Failed to create TFLite interpreter";
    return false;
  }

  // 4) Allocate Tensors
  interpreter_->SetNumThreads(4);
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate TFLite tensors";
    return false;
  }

  return true;
}

bool SuperResolutionModel::Process(const SkBitmap& input_bitmap,
                                   SkBitmap* output_bitmap) {
  if (!IsReady() && !Initialize()) {
    return false;
  }

  // Convert bitmap to tensor
  WTF::Vector<float> tensor_data = BitmapToTFLiteTensor(input_bitmap);
  if (tensor_data.size() == 0) {
    LOG(ERROR) << "Failed to convert bitmap to tensor data";
    return false;
  }

  // For TFLite, we typically do NHWC with batch dimension
  std::vector<int> dims = GetTensorDimensionsForBitmap(input_bitmap, true);

  if (!CopyTensorToInterpreter(tensor_data, dims, interpreter_.get(), 0)) {
    LOG(ERROR) << "Failed to copy data into TFLite interpreter";
    return false;
  }

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "TFLite inference error";
    return false;
  }

  // Convert output tensor to SkBitmap
  if (!TensorToBitmap(interpreter_.get(), 0, output_bitmap)) {
    LOG(ERROR) << "Failed to convert TFLite output to SkBitmap";
    return false;
  }

  return true;
}

bool TensorToBitmap(tflite::Interpreter* interpreter,
                    int output_tensor_index,
                    SkBitmap* output_bitmap) {
  if (!interpreter || !output_bitmap)
    return false;

  TfLiteTensor* output_tensor = interpreter->output_tensor(output_tensor_index);
  if (!output_tensor)
    return false;

  if (output_tensor->dims->size != 4) {
    return false;
  }

  int height = output_tensor->dims->data[1];
  int width = output_tensor->dims->data[2];
  int channels = output_tensor->dims->data[3];
  if (channels != 3) {
    return false;
  }

  SkImageInfo info = SkImageInfo::Make(width, height,
                                       kRGBA_8888_SkColorType,
                                       kPremul_SkAlphaType);
  if (!output_bitmap->tryAllocPixels(info)) {
    return false;
  }

  SkPixmap pixmap;
  if (!output_bitmap->peekPixels(&pixmap)) {
    return false;
  }

  uint8_t* dest_data = static_cast<uint8_t*>(pixmap.writable_addr());
  const float* src_data = output_tensor->data.f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int tensor_offset = (y * width + x) * channels;
      int bitmap_offset = y * pixmap.rowBytes() + x * 4;
      // Float [0,1] -> uint8 [0,255]
      dest_data[bitmap_offset + 0] =
          static_cast<uint8_t>(src_data[tensor_offset + 0] * 255.0f); // R
      dest_data[bitmap_offset + 1] =
          static_cast<uint8_t>(src_data[tensor_offset + 1] * 255.0f); // G
      dest_data[bitmap_offset + 2] =
          static_cast<uint8_t>(src_data[tensor_offset + 2] * 255.0f); // B
      dest_data[bitmap_offset + 3] = 255; // A
    }
  }

  return true;
}

bool ApplySuperResolution(const SkBitmap& input_bitmap,
                          SkBitmap* output_bitmap) {
  // Example policy: skip huge images
  if (input_bitmap.width() > 1024 || input_bitmap.height() > 1024) {
    *output_bitmap = input_bitmap;
    return true;
  }

  auto* model = SuperResolutionModel::GetInstance();
  return model->Process(input_bitmap, output_bitmap);
}

}  // namespace pixlift
}  // namespace blink
