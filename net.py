import torch
import torch.nn as nn
import torch.nn.functional as F


# define the SAM model
class SAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder


    def forward(self, image, point_coords, point_labels):
        image_embedding, lower_feature = self.image_encoder(image)  # (B, 256, 64, 64)
        point_coords = torch.as_tensor(point_coords, dtype=torch.float32, device=image.device)
        point_labels = torch.as_tensor(point_labels, dtype=torch.float32, device=image.device)
        points = (point_coords, point_labels)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )   
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  
            image_pe=self.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=False,
            lower_feature = lower_feature
        )
        mask = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return mask

# define the Refinement Net
class RefinementNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(RefinementNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # encoder
        self.encoder1 = self.conv_block(self.in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        
        # decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, self.out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))
        e5 = self.encoder5(nn.MaxPool2d(2)(e4))
        
        d4 = self.upconv4(e5)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.decoder1(d1)
        
        out = self.out_conv(d1)
        return out
