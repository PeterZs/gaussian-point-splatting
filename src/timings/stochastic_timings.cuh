#ifndef STOCHASTIC_TIMINGS_CUH
#define STOCHASTIC_TIMINGS_CUH

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// Define a namespace to avoid name conflicts
namespace TimingSystem {

    // This is our "stochastic timing" struct.
    // For CUDA compatibility, we add __host__ __device__ to the default constructor.
    struct StochasticTiming {
        float PreprocessTime;
        float AliasTime;
        float PointcloudTime;
        float PostProcessTime;
        float FrameTime;
        float PointCount;

        StochasticTiming()
            : PreprocessTime(0.0f),
            AliasTime(0.0f),
            PointcloudTime(0.0f),
            PostProcessTime(0.0f),
            FrameTime(0.0f),
            PointCount(0.0f)
        {}
        
    };

    class StochasticTimings {
    public:
        static void addTimingBatch();
        static void addTimingEntry();

        static StochasticTiming& getCurrentTiming();

        static void writeToFile(const std::string& filePath);

    private:
        //batches of frames. idea is that you have e.g. N frames per camera viewpoint, and you do M camera viewpoints in a row
        static std::vector<std::vector<StochasticTiming>> timings;
    };

} // namespace TimingSystem

#endif // STOCHASTIC_TIMINGS_CUH
