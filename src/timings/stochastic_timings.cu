#include "stochastic_timings.cuh"
#include <utils/io.h>

namespace TimingSystem {

    // Define the static member variable.
    std::vector<std::vector<StochasticTiming>> StochasticTimings::timings;

    void StochasticTimings::addTimingBatch() {
        // Push a new default (zero initialized) timing entry.
        timings.push_back(std::vector<StochasticTiming>());
    }

    void StochasticTimings::addTimingEntry() {
        if (timings.empty()) {
            timings.push_back(std::vector<StochasticTiming>());
        }
        timings.back().push_back(StochasticTiming());
    }

    StochasticTiming& StochasticTimings::getCurrentTiming() {
        // If the timings list is empty, add a new entry before returning.
        if (timings.empty()) {
            timings.push_back(std::vector<StochasticTiming>());
        }
        if (timings.back().empty()) {
            timings.back().push_back(StochasticTiming());
        }
        return timings.back().back();
    }

    void StochasticTimings::writeToFile(const std::string& filePath) {
        // Create a JSON array to hold our timings.
        nlohmann::json jsonArray = nlohmann::json::array();

        // Populate the JSON array with each timing entry.
        for (const auto& batch : timings) {
            nlohmann::json b;
            for (const auto& timing : batch) {
                nlohmann::json j;
                j["PreprocessTime"] = timing.PreprocessTime;
                j["AliasTime"] = timing.AliasTime;
                j["PointcloudTime"] = timing.PointcloudTime;
                j["PostProcessTime"] = timing.PostProcessTime;
                j["FrameTime"] = timing.FrameTime;
                j["PointCount"] = timing.PointCount;
                b.push_back(j);
            }
            jsonArray.push_back(b);
        }

        // Open the file and write the JSON data
        std::ofstream outFile(resolveProjectPath(filePath));
        if (outFile.is_open()) {
            outFile << jsonArray.dump(4); // Dump with an indent of 4 spaces.
            outFile.close();
        }
        else {
            std::cerr << "Error opening file: " << filePath << std::endl;
        }
    }

} // namespace TimingSystem
