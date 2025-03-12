#ifndef THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_PIXLIFT_SUPER_RESOLUTION_H_
#define THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_PIXLIFT_SUPER_RESOLUTION_H_

#include "third_party/skia/include/core/SkBitmap.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

// TFLite headers
#include "third_party/tflite/src/tensorflow/lite/c/common.h"
#include "third_party/tflite/src/tensorflow/lite/interpreter.h"
#include "third_party/tflite/src/tensorflow/lite/model.h"

namespace blink {
namespace pixlift {

WTF::Vector<float> BitmapToTFLiteTensor(const SkBitmap& bitmap);
std::vector<int> GetTensorDimensionsForBitmap(const SkBitmap& bitmap, bool include_batch_dim);
bool CopyTensorToInterpreter(const WTF::Vector<float>& tensor_data,
                             const std::vector<int>& dimensions,
                             tflite::Interpreter* interpreter,
                             int tensor_index);

bool TensorToBitmap(tflite::Interpreter* interpreter, 
                    int output_tensor_index,
                    SkBitmap* output_bitmap);

// Singleton managing the TFLite model for super-resolution.
class SuperResolutionModel {
 public:
  static SuperResolutionModel* GetInstance();

  bool Initialize(); // Load TFLite model from file, build interpreter
  bool Process(const SkBitmap& input_bitmap, SkBitmap* output_bitmap);

  bool IsReady() const { return model_ && interpreter_; }

 private:
  SuperResolutionModel();
  ~SuperResolutionModel();

  static SuperResolutionModel* instance_;

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

// High-level function to run super-resolution on an SkBitmap
bool ApplySuperResolution(const SkBitmap& input_bitmap, SkBitmap* output_bitmap);

}  // namespace pixlift
}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_PIXLIFT_SUPER_RESOLUTION_H_
