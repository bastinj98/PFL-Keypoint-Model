from ultralytics import YOLO
import os

import onnx



def main(model_dir, model_name):

    pt_model_name = model_name + ".pt"
    
    model = YOLO(os.path.join(model_dir, pt_model_name))
    #model.fuse()
    model.export(format="onnx", dynamic = False, simplify=True, imgsz = (1280,1280), opset=12)

    # Load your ONNX model
    onnx_model_name = model_name + '.onnx'
    model = onnx.load(os.path.join(model_dir, onnx_model_name))

    # Rename input tensor
    old_input_name = model.graph.input[0].name
    model.graph.input[0].name = "input"

    # Rename output tensor
    old_output_name = model.graph.output[0].name
    model.graph.output[0].name = "output"

    # Update any node references to input/output
    for node in model.graph.node:
        node.input[:] = ["input" if i == old_input_name else i for i in node.input]
        node.output[:] = ["output" if o == old_output_name else o for o in node.output]

    # Save the updated model
    onnx.save(model, "/home/bb/Dev/EPL-Center-Circle/" + onnx_model_name)
    print(f"Renamed input '{old_input_name}' → 'input', output '{old_output_name}' → 'output'")



if __name__ == '__main__':
    model_dir = "/home/bb/Dev/EPL-Center-Circle/runs/pose/yolo-pose-center-circle-aug_1000eps/weights"
    model_name = "last"
   
   

    main(model_dir = model_dir, model_name = model_name)

