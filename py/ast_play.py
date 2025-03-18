import ast

code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.relu(input)
        return output


x = torch.randn(128, device='cuda')
m = Model()
y = m(x)
print(y)
"""

exec(code)

tree = ast.parse(code)
print(ast.dump(tree, indent=4))


# replace relu with gelu
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        forward_def = node.body[1]
        assert forward_def.name == "forward"
        forward_def.body[0] = ast.parse("output = F.gelu(input)")

print("===== gelu code =====")
gelu_code = ast.unparse(tree)
print(gelu_code)
print()
exec(gelu_code)
