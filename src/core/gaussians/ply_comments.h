#pragma once
#include <string>
#include <algorithm>
#include <vector>

struct PlyComment {
    std::string text;
};

struct PlyCommentUtils {
    static PlyComment makeBVHSorted() {
        return { "bvh_sorted_order" };
    }
    static bool hasBVHSorted(const std::vector<PlyComment>& comments) {
        return std::any_of(comments.begin(), comments.end(),
            [](const PlyComment& c) {
            return c.text == "bvh_sorted_order";
        });
    }
};
