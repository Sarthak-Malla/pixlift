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
  LOG(INFO) << "PixLift: Converting bitmap to TFLite tensor.";

  if (!bitmap.readyToDraw()) {
    LOG(WARNING) << "PixLift: Bitmap not ready to draw, returning empty tensor.";
    return {};
  }

  const int width = bitmap.width();
  const int height = bitmap.height();
  const int channels = 3;  // We assume 3 channels (RGB).

  WTF::Vector<float> tensor_data;
  tensor_data.resize(width * height * channels);

  SkPixmap pixmap;
  if (!bitmap.peekPixels(&pixmap)) {
    LOG(ERROR) << "PixLift: Failed to peek pixels from bitmap.";
    return {};
  }

  SkColorType color_type = bitmap.colorType();
  const uint8_t* src_data = static_cast<const uint8_t*>(pixmap.addr());
  if (!src_data) {
    LOG(ERROR) << "PixLift: src_data is null.";
    return {};
  }

  // Convert from SkBitmap to float RGB buffer
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

        tensor_data[offset + 0] = pixel[r_index] / 255.0f;  // R
        tensor_data[offset + 1] = pixel[g_index] / 255.0f;  // G
        tensor_data[offset + 2] = pixel[b_index] / 255.0f;  // B
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
        tensor_data[offset + 0] = pixel[0] / 255.0f;  // R
        tensor_data[offset + 1] = pixel[1] / 255.0f;  // G
        tensor_data[offset + 2] = pixel[2] / 255.0f;  // B
      }
    }
  } else {
    LOG(WARNING) << "PixLift: Unsupported SkColorType, returning empty tensor.";
    return {};
  }

  LOG(INFO) << "PixLift: Successfully converted bitmap to float tensor.";
  return tensor_data;
}

std::vector<int> GetTensorDimensionsForBitmap(const SkBitmap& bitmap,
                                              bool include_batch_dim) {
  LOG(INFO) << "PixLift: Getting tensor dimensions for bitmap. "
            << "Include batch dim: " << include_batch_dim;

  if (include_batch_dim) {
    // [1, height, width, channels]
    return {1, bitmap.height(), bitmap.width(), 3};
  } else {
    // [height, width, channels]
    return {bitmap.height(), bitmap.width(), 3};
  }
}

bool CopyTensorToInterpreter(const WTF::Vector<float>& tensor_data,
                             const std::vector<int>& dimensions,
                             tflite::Interpreter* interpreter,
                             int tensor_index) {
  LOG(INFO) << "PixLift: Copying tensor data to TFLite interpreter (tensor_index="
            << tensor_index << ").";

  if (!interpreter || tensor_data.size() == 0) {
    LOG(ERROR) << "PixLift: Interpreter is null or tensor_data is empty.";
    return false;
  }

  TfLiteStatus status = interpreter->ResizeInputTensor(tensor_index, dimensions);
  if (status != kTfLiteOk) {
    LOG(ERROR) << "PixLift: Failed to resize input tensor.";
    return false;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "PixLift: Failed to allocate tensors.";
    return false;
  }

  TfLiteTensor* tensor = interpreter->tensor(tensor_index);
  if (!tensor || !tensor->data.f) {
    LOG(ERROR) << "PixLift: Output tensor is null or has no float data.";
    return false;
  }

  // Copy data into the TFLite tensor
  std::memcpy(tensor->data.f,
              tensor_data.data(),
              tensor_data.size() * sizeof(float));

  LOG(INFO) << "PixLift: Copied " << tensor_data.size()
            << " floats into TFLite tensor.";
  return true;
}

// Static instance for the singleton
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
    LOG(INFO) << "PixLift: SuperResolutionModel already initialized.";
    return true;  // Already initialized
  }

  LOG(INFO) << "PixLift: Initializing SuperResolutionModel...";

  // 1) Load the TFLite model from a file path.
  const base::FilePath model_path("/pixlift/res/xlsr_quantized.tflite");
  std::string model_data;
  if (!base::ReadFileToString(model_path, &model_data)) {
    LOG(ERROR) << "PixLift: Failed to read TFLite model from "
               << model_path.value();
    return false;
  }
  LOG(INFO) << "PixLift: Model file read, size=" << model_data.size() << " bytes.";

  // 2) Build model from buffer
  model_ = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                    model_data.size());
  if (!model_) {
    LOG(ERROR) << "PixLift: Failed to build TFLite model.";
    return false;
  }
  LOG(INFO) << "PixLift: TFLite model built successfully.";

  // 3) Create interpreter
  LOG(INFO) << "PixLift: Creating TFLite interpreter with BuiltinOpResolver.";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    LOG(ERROR) << "PixLift: Failed to create TFLite interpreter.";
    return false;
  }

  // 4) Allocate Tensors
  interpreter_->SetNumThreads(4);
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "PixLift: Failed to allocate TFLite tensors.";
    return false;
  }

  LOG(INFO) << "PixLift: SuperResolutionModel initialization complete.";
  return true;
}

bool SuperResolutionModel::Process(const SkBitmap& input_bitmap,
                                   SkBitmap* output_bitmap) {
  LOG(INFO) << "PixLift: SuperResolutionModel::Process called.";

  if (!IsReady() && !Initialize()) {
    LOG(ERROR) << "PixLift: Model not ready; initialization failed.";
    return false;
  }

  // Convert bitmap to tensor
  LOG(INFO) << "PixLift: Converting bitmap to float tensor for TFLite.";
  WTF::Vector<float> tensor_data = BitmapToTFLiteTensor(input_bitmap);
  if (tensor_data.size() == 0) {
    LOG(ERROR) << "PixLift: Tensor conversion failed or empty.";
    return false;
  }

  // For TFLite, we typically do NHWC with batch dimension
  LOG(INFO) << "PixLift: Setting up TFLite dimensions.";
  std::vector<int> dims = GetTensorDimensionsForBitmap(input_bitmap, true);

  LOG(INFO) << "PixLift: Copying input tensor to TFLite interpreter.";
  if (!CopyTensorToInterpreter(tensor_data, dims, interpreter_.get(), 0)) {
    LOG(ERROR) << "PixLift: Failed to copy data into TFLite interpreter.";
    return false;
  }

  // Run inference
  LOG(INFO) << "PixLift: Invoking TFLite interpreter for super-resolution.";
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "PixLift: TFLite inference error.";
    return false;
  }
  LOG(INFO) << "PixLift: Inference completed successfully.";

  // Convert output tensor to SkBitmap
  LOG(INFO) << "PixLift: Converting TFLite output tensor to SkBitmap.";
  if (!TensorToBitmap(interpreter_.get(), 0, output_bitmap)) {
    LOG(ERROR) << "PixLift: Failed to convert TFLite output to SkBitmap.";
    return false;
  }

  LOG(INFO) << "PixLift: Super-resolution successfully applied.";
  return true;
}

bool TensorToBitmap(tflite::Interpreter* interpreter,
                    int output_tensor_index,
                    SkBitmap* output_bitmap) {
  LOG(INFO) << "PixLift: TensorToBitmap - converting TFLite output to SkBitmap.";
  if (!interpreter || !output_bitmap) {
    LOG(ERROR) << "PixLift: Invalid interpreter or output_bitmap.";
    return false;
  }

  TfLiteTensor* output_tensor = interpreter->output_tensor(output_tensor_index);
  if (!output_tensor) {
    LOG(ERROR) << "PixLift: output_tensor is null.";
    return false;
  }

  if (output_tensor->dims->size != 4) {
    LOG(ERROR) << "PixLift: output_tensor->dims is not 4D, cannot proceed.";
    return false;
  }

  int height = output_tensor->dims->data[1];
  int width = output_tensor->dims->data[2];
  int channels = output_tensor->dims->data[3];
  if (channels != 3) {
    LOG(ERROR) << "PixLift: Unexpected number of channels, expected 3, got "
               << channels;
    return false;
  }

  SkImageInfo info = SkImageInfo::Make(width, height,
                                       kRGBA_8888_SkColorType,
                                       kPremul_SkAlphaType);
  if (!output_bitmap->tryAllocPixels(info)) {
    LOG(ERROR) << "PixLift: Failed to allocate pixels in output_bitmap.";
    return false;
  }

  SkPixmap pixmap;
  if (!output_bitmap->peekPixels(&pixmap)) {
    LOG(ERROR) << "PixLift: Could not peek pixels for output_bitmap.";
    return false;
  }

  uint8_t* dest_data = static_cast<uint8_t*>(pixmap.writable_addr());
  const float* src_data = output_tensor->data.f;
  if (!src_data) {
    LOG(ERROR) << "PixLift: src_data is null in output tensor.";
    return false;
  }

  // Float [0,1] -> uint8 [0,255]
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int tensor_offset = (y * width + x) * channels;
      int bitmap_offset = y * pixmap.rowBytes() + x * 4;

      dest_data[bitmap_offset + 0] =
          static_cast<uint8_t>(src_data[tensor_offset + 0] * 255.0f); // R
      dest_data[bitmap_offset + 1] =
          static_cast<uint8_t>(src_data[tensor_offset + 1] * 255.0f); // G
      dest_data[bitmap_offset + 2] =
          static_cast<uint8_t>(src_data[tensor_offset + 2] * 255.0f); // B
      dest_data[bitmap_offset + 3] = 255;                             // A
    }
  }

  LOG(INFO) << "PixLift: Output tensor successfully converted to SkBitmap.";
  return true;
}

bool ApplySuperResolution(const SkBitmap& input_bitmap,
                          SkBitmap* output_bitmap) {
  LOG(INFO) << "PixLift: ApplySuperResolution called.";

  // Example policy: skip huge images
  if (input_bitmap.width() > 1024 || input_bitmap.height() > 1024) {
    LOG(WARNING) << "PixLift: Skipping super-resolution for large image "
                 << input_bitmap.width() << "x" << input_bitmap.height() << ".";
    *output_bitmap = input_bitmap;
    return true;
  }

  LOG(INFO) << "PixLift: Attempting super-resolution. "
            << "Input size=" << input_bitmap.width()
            << "x" << input_bitmap.height();

  auto* model = SuperResolutionModel::GetInstance();
  bool success = model->Process(input_bitmap, output_bitmap);
  if (success) {
    LOG(INFO) << "PixLift: Super-resolution succeeded. "
              << "Output size=" << output_bitmap->width() << "x"
              << output_bitmap->height();
  } else {
    LOG(ERROR) << "PixLift: Super-resolution failed.";
  }

  return success;
}

}  // namespace pixlift
}  // namespace blink
