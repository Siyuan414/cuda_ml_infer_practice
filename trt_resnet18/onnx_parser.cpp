#include "onnx_proto/onnx.pb.h"
#include<fstream>
#include<iostream>
#include<string>

using namespace std;
int main(int argc, char* argv[])
{
    const std::string path = (argc > 1) ? argv[1] : "resnet18.onnx";

    // 1. Read file into string
    ifstream f(path,ios::in | ios::binary);
    if (!f.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return -1;
    }
    string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    //2.parse into modelproto
    onnx::ModelProto model;
    if(!model.ParseFromString(data)){
        cerr << "Failed to parse ONNX model." << endl;
        return -1;
    }
    cout << "Model parsed successfully." << endl;
    const auto& graph = model.graph();
    cout << "Graph name: " << graph.name() << endl;
    cout << "Number of nodes: " << graph.node_size() << endl;
    cout << "Initializer count: " << graph.initializer_size() << endl;

    //3. iterate nodes - print op type and input/output names
    for(int i = 0; i < graph.node_size(); ++i){
        const auto& node = graph.node(i);
        std::cout << "[" << i << "] " << node.op_type() << "\n";
        std::cout << "  inputs : ";
        for (const auto& s : node.input())  std::cout << s << " ";
        std::cout << "\n  outputs: ";
        for (const auto& s : node.output()) std::cout << s << " ";
        std::cout << "\n";

        //print attributes
        if (node.op_type() == "Conv") {
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape" || attr.name() == "group") {
                    std::cout << "  " << attr.name() << ": ";
                    if (attr.type() == onnx::AttributeProto::INTS) {
                        for (int v : attr.ints()) std::cout << v << " ";
                    } else if (attr.type() == onnx::AttributeProto::INT) {
                        std::cout << attr.i();
                    }
                    std::cout << "\n";
                }
            }
        }
    }

}