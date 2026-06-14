#include <iostream>
#include <fstream>
#include <string>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include "onnx_proto/onnx.pb.h"
#include <google/protobuf/repeated_field.h>

// ── GraphOptimizer ────────────────────────────────────────────────────────────
class GraphOptimizer {
public:
    explicit GraphOptimizer(onnx::GraphProto* graph) : graph_(graph) {}

    void run() {
        std::cout << "Before: " << graph_->node_size() << " nodes\n";
        int cf  = constantFolding();
        int dne = deadNodeElimination();
        std::cout << "Constant folding eliminated : " << cf  << " nodes\n";
        std::cout << "Dead node elimination removed: " << dne << " nodes\n";
        std::cout << "After : " << graph_->node_size() << " nodes\n";
    }

private:
    // ── Pass 1: Constant Folding ──────────────────────────────────────────────
    // Finds nodes where every input is a known constant (initializer).
    // For ResNet-18 this is 0 — all "constant-looking" nodes depend on the
    // runtime input shape via Shape → Concat → Reshape.
    int constantFolding() {
        // Build constant set from initializers
        std::unordered_set<std::string> constants;
        for (const auto& init : graph_->initializer())
            constants.insert(init.name());

        int folded = 0;
        for (const auto& node : graph_->node()) {
            bool all_const = true;
            for (const auto& inp : node.input()) {
                if (!inp.empty() && !constants.count(inp)) {
                    all_const = false;
                    break;
                }
            }
            if (all_const) {
                std::cout << "  [CF] Foldable: " << node.op_type() << "\n";
                ++folded;
                // In a full implementation: evaluate node on CPU and add
                // result as a new initializer, then remove the node.
                // For now we just count and report.
            }
        }
        return folded;
    }

    // ── Pass 2: Dead Node Elimination ────────────────────────────────────────
    // Reverse DFS from graph outputs. Any node not reachable is dead.
    int deadNodeElimination() {
        // Build producer map: output tensor name → node index
        std::unordered_map<std::string, int> producer;
        for (int i = 0; i < graph_->node_size(); ++i)
            for (const auto& out : graph_->node(i).output())
                producer[out] = i;

        // DFS backwards from graph outputs
        std::unordered_set<int> visited;
        std::stack<int> stk;
        for (const auto& out : graph_->output())
            if (producer.count(out.name()))
                stk.push(producer[out.name()]);

        while (!stk.empty()) {
            int idx = stk.top(); stk.pop();
            if (visited.count(idx)) continue;
            visited.insert(idx);
            for (const auto& inp : graph_->node(idx).input())
                if (producer.count(inp))
                    stk.push(producer[inp]);
        }

        // Report dead nodes
        int dead = 0;
        google::protobuf::RepeatedPtrField<onnx::NodeProto> live_nodes;
        for (int i = 0; i < graph_->node_size(); ++i) {
           if (visited.count(i)) {
                *live_nodes.Add() = graph_->node(i);
            } else {
                std::cout << "  [DNE] Dead: [" << i << "] "
                          << graph_->node(i).op_type() << "\n";
                ++dead;
            }
        }
        graph_->mutable_node()->Swap(&live_nodes);
        return dead;
    }

    onnx::GraphProto* graph_;
};

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const std::string path = (argc > 1) ? argv[1] : "resnet18.onnx";

    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return 1; }
    std::string data((std::istreambuf_iterator<char>(f)), {});

    onnx::ModelProto model;
    if (!model.ParseFromString(data)) {
        std::cerr << "Failed to parse ONNX\n"; return 1;
    }

    std::cout << "Model : " << path << "\n";
    std::cout << "Graph : " << model.graph().name() << "\n\n";

    GraphOptimizer optimizer(model.mutable_graph());
    optimizer.run();

    return 0;
}
