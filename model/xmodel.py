import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Iterate through the sequence
        outputs = []
        for i in range(sequence_length):
            h = self.gru(x[:, i, :], h)
            outputs.append(h.unsqueeze(1))

        # Concatenate along the sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs

class XMT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024):
        super(XMT, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.patch_size = patch_size

        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1 + num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.gru_hidden_size = gru_hidden_size
        self.gru = GRUBlock(dim, self.gru_hidden_size)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(gru_hidden_size, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, num_classes)
        )


    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.features(img)
        expected_size = (p, p)
        if x.size(2) != expected_size[0] or x.size(3) != expected_size[1]:
            raise ValueError(f"Expected feature map size {expected_size}, but got {x.size(2)}x{x.size(3)}")
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        y = self.patch_to_embedding(y)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.transformer(x, mask)

        # Add GRU layer
        x = self.gru(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
