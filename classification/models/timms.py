import torch
import torchvision

from torch import nn
import timm


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class timm_Model(nn.Module):
    def __init__(self,
                 model_name,
                 drop_rate=0.0,
                 n_output_layers=2,
                 output_layers_shapes={
                    5: "erosion",
                    4: "jsn"
                 },
                 pretrained=True):
        super().__init__()

        self.output_mapping = output_layers_shapes
        self.backbone = timm.create_model(model_name, pretrained=pretrained, drop_rate=drop_rate)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone.head.fc.in_features, self.backbone.head.fc.in_features // 2),
                nn.GELU(),
                nn.Linear(self.backbone.head.fc.in_features // 2, key)
            )
            for key in self.output_mapping.keys()
        ])
        self.backbone.head.fc = nn.Identity()
            
        self.heads.apply(init_weights)
    
        
    def forward(self, x):
        feats = self.backbone(x)
        head_outputs = {
            self.output_mapping[self.heads[i][-1].out_features]: self.heads[i](feats)
            for i in range(len(self.heads))
        }
        return head_outputs


if __name__ == "__main__":
    model = timm_Model()
    print(model(torch.randn((1, 3, 224, 224))))