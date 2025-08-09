
import onnx
from pathlib import Path

def export_onnx_diagram(onnx_path, output_path):
    # Load ONNX model
    model = onnx.load(str(onnx_path))
    
    # Create simple text diagram
    with open(output_path, 'w') as f:
        f.write(f"ONNX Model Architecture\n")
        f.write("="*30 + "\n\n")
        
        # List all layers
        for i, node in enumerate(model.graph.node):
            f.write(f"{i+1:3d}. {node.op_type:15s} -> {node.name}\n")
        
        f.write(f"\nTotal layers: {len(model.graph.node)}")

# Use it
onnx_file = Path("../../models/convnext_classifier.onnx")
output_file = Path("../assets/figs/convnext_architecture.txt")

export_onnx_diagram(onnx_file, output_file)
print(f"Architecture exported to {output_file}")








