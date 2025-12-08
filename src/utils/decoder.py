import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

class PaddedVideoDecoder:
    """
    Wraps torchcodec.decoders.VideoDecoder and pads frames (H, W, 3) to 16:9.
    Output: torch.Tensor(H, W, 3) uint8
    """
    def __init__(self, filepath, pad_value: int = 0, device_id: int = 3):
        self.decoder = VideoDecoder(filepath)
        self.pad_value = pad_value
        self.target_aspect = 16 / 9

    def __len__(self):
        return len(self.decoder)

    def _pad_to_16_9(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Input frame: torch.Tensor(3, H, W), uint8
        Output frame: padded to 16:9
        """
        _, H, W = frame.shape
        current_aspect = W / H

        # If frame too tall → extend width
        if current_aspect < self.target_aspect:
            new_W = int(H * self.target_aspect)
            new_H = H
        # If frame too wide → extend height
        else:
            new_H = int(W / self.target_aspect)
            new_W = W

        pad_H = new_H - H
        pad_W = new_W - W

        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        padded = F.pad(
            frame,
            pad=(pad_left, pad_right, 
                 pad_top, pad_bottom),
            mode="constant",
            value=self.pad_value
        )
        return padded

    def __getitem__(self, index: int) -> torch.Tensor:
        frame = self.decoder[index]
        return self._pad_to_16_9(frame)
