# gRestorer/core/scene.py
# Scene - Temporal grouping of mosaic detections
#
# A Scene represents a contiguous sequence of frames where mosaic detections
# overlap spatially. Scenes are converted to Clips for restoration.

from typing import List, Tuple, Optional
import torch


class Scene:
    """
    Temporal grouping of frames with overlapping mosaic detections.
    
    Stores frames, masks, and boxes for a sequence of detections that belong
    together spatially and temporally. Used to create Clips for restoration.
    
    Attributes:
        frames: List of BGR frames [H, W, 3] float32 [0,1] on GPU
        masks: List of masks [H, W] uint8 [0,255] on GPU
        boxes: List of bounding boxes (x1, y1, x2, y2)
        frame_start: Global frame number where scene starts
        frame_end: Global frame number where scene ends (inclusive)
    """
    
    def __init__(self):
        """Initialize empty scene."""
        self.frames: List[torch.Tensor] = []      # [H, W, 3] BGR float32 [0,1] GPU
        self.masks: List[torch.Tensor] = []       # [H, W] uint8 [0,255] GPU
        self.boxes: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)
        self.frame_start: Optional[int] = None
        self.frame_end: Optional[int] = None
    
    def __len__(self) -> int:
        """Number of frames in scene."""
        return len(self.frames)
    
    def add_frame(
        self,
        frame_num: int,
        frame: torch.Tensor,  # [H, W, 3] BGR float32 [0,1] GPU
        mask: torch.Tensor,   # [H, W] uint8 GPU
        box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    ) -> None:
        """
        Add frame to scene.
        
        Args:
            frame_num: Global frame number
            frame: BGR frame [H, W, 3] float32 [0,1] on GPU
            mask: Mask [H, W] uint8 on GPU
            box: Bounding box (x1, y1, x2, y2)
        """
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
        else:
            assert frame_num == self.frame_end + 1, "Frames must be consecutive"
            self.frame_end = frame_num
        
        self.frames.append(frame)
        self.masks.append(mask)
        self.boxes.append(box)
    
    def merge_mask_box(
        self,
        mask: torch.Tensor,
        box: Tuple[int, int, int, int]
    ) -> None:
        """
        Merge additional detection into current frame (multiple detections/frame).
        
        Takes the union of boxes and max of masks for the current frame.
        
        Args:
            mask: Additional mask to merge
            box: Additional box to merge
        """
        assert len(self) > 0, "Cannot merge into empty scene"
        
        # Union of boxes
        x1 = min(self.boxes[-1][0], box[0])
        y1 = min(self.boxes[-1][1], box[1])
        x2 = max(self.boxes[-1][2], box[2])
        y2 = max(self.boxes[-1][3], box[3])
        self.boxes[-1] = (x1, y1, x2, y2)
        
        # Union of masks (element-wise max)
        self.masks[-1] = torch.maximum(self.masks[-1], mask)
    
    def belongs(self, box: Tuple[int, int, int, int]) -> bool:
        """
        Check if box belongs to this scene (overlaps with last box).
        
        Args:
            box: Box to check (x1, y1, x2, y2)
            
        Returns:
            True if box overlaps with scene's last box
        """
        if len(self.boxes) == 0:
            return False
        
        last_box = self.boxes[-1]
        return self._box_overlap(last_box, box)
    
    @staticmethod
    def _box_overlap(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if two boxes overlap (IOU-based).
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            True if boxes overlap
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Check for no overlap
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        return True
    
    def __repr__(self) -> str:
        return (f"Scene(frames={len(self)}, "
                f"start={self.frame_start}, end={self.frame_end})")
