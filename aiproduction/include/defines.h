#ifdef TENSORRT

#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

#endif

#ifdef DIRECTML
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/dml/dml_provider_factory.h"

#endif

#ifdef AICPU
#include "../../deps/onnxruntime/cpu/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/cpu/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h"

#endif