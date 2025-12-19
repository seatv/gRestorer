# gRestorer/core/scene_tracker.py
# SceneTracker - Manages scenes and creates clips from detections
#
# Tracks temporal scenes by grouping overlapping detections across frames.
# Creates Clips when scenes complete (reach max length, end, or EOF).

from typing import List, Tuple
import torch

from .scene import Scene


class SceneTracker:
    """
    Tracks temporal scenes and creates clips for restoration.
    
    Maintains active scenes, groups detections spatially/temporally,
    and creates clips when scenes complete.
    
    Attributes:
        max_clip_length: Maximum frames per clip (e.g., 30)
        clip_size: Target clip size for preprocessing (e.g., 256)
        border_ratio: Border expansion for ROI crops (e.g., 0.06)
        pad_mode: Padding mode for clips ('reflect', 'replicate', 'constant')
        active_scenes: List of currently active scenes
        clip_counter: Total clips created (for tracking)
    """
    
    def __init__(
        self,
        max_clip_length: int = 30,
        clip_size: int = 256,
        border_ratio: float = 0.06,
        pad_mode: str = 'reflect'
    ):
        """
        Initialize scene tracker.
        
        Args:
            max_clip_length: Maximum frames per clip
            clip_size: Target clip size (e.g., 256x256)
            border_ratio: Border expansion for crops (0.06 = 6%)
            pad_mode: Padding mode ('reflect', 'replicate', 'constant')
        """
        self.max_clip_length = max_clip_length
        self.clip_size = clip_size
        self.border_ratio = border_ratio
        self.pad_mode = pad_mode
        
        self.active_scenes: List[Scene] = []
        self.clip_counter = 0
    
    def update(
        self,
        frame_num: int,
        bgr_frames: List[torch.Tensor],  # [H, W, 3] float32 [0,1] GPU
        detections: List,  # List[Detection]
        eof: bool = False
    ) -> List[Scene]:
        """
        Update scenes with new detections and return completed scenes.
        
        Args:
            frame_num: Starting frame number for this batch
            bgr_frames: List of BGR frames [H, W, 3] float32 [0,1] on GPU
            detections: List of Detection objects (one per frame)
            eof: End of file flag
            
        Returns:
            List of completed Scenes ready for restoration
        """
        completed_scenes = []
        
        for i, (frame, det) in enumerate(zip(bgr_frames, detections)):
            current_frame_num = frame_num + i
            
            # Process each detection in this frame
            if det.boxes is not None and det.boxes.numel() > 0:
                # Convert detections to list
                boxes = det.boxes.cpu().numpy()  # [N, 4]
                masks_tensor = det.masks  # [N, H, W] uint8 on CPU
                
                for j in range(len(boxes)):
                    box_xyxy = boxes[j]
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    box = (x1, y1, x2, y2)
                    
                    # Get mask and move to GPU
                    mask = masks_tensor[j].to(frame.device)  # [H, W] uint8 GPU
                    
                    # Find or create scene for this detection
                    scene = self._find_or_create_scene(box)
                    
                    # Add to scene or merge if same frame
                    if scene.frame_end == current_frame_num:
                        # Multiple detections in same frame → merge
                        scene.merge_mask_box(mask, box)
                    else:
                        # New frame for this scene
                        scene.add_frame(current_frame_num, frame, mask, box)
            
            # Check for completed scenes
            completed_scenes.extend(
                self._finalize_completed_scenes(current_frame_num, eof)
            )
        
        # EOF: finalize all remaining scenes
        if eof:
            completed_scenes.extend(self._finalize_all_scenes())
        
        return completed_scenes
    
    def _find_or_create_scene(self, box: Tuple[int, int, int, int]) -> Scene:
        """
        Find existing scene that box belongs to, or create new one.
        
        Args:
            box: (x1, y1, x2, y2)
            
        Returns:
            Scene object
        """
        # Try to find existing scene with overlapping box
        for scene in self.active_scenes:
            if scene.belongs(box):
                return scene
        
        # No match → create new scene
        new_scene = Scene()
        self.active_scenes.append(new_scene)
        return new_scene
    
    def _finalize_completed_scenes(
        self,
        current_frame_num: int,
        eof: bool
    ) -> List[Scene]:
        """
        Finalize scenes that are complete.
        
        A scene is complete if:
        - It's too long (>= max_clip_length)
        - Current frame is past scene end (gap in detections)
        - EOF
        
        Args:
            current_frame_num: Current frame number
            eof: End of file flag
            
        Returns:
            List of completed Scenes
        """
        scenes = []
        scenes_to_remove = []
        
        for scene in self.active_scenes:
            # Check if scene should be finalized
            is_too_long = len(scene) >= self.max_clip_length
            is_past_end = scene.frame_end < current_frame_num
            should_finalize = is_too_long or is_past_end or eof
            
            if should_finalize and len(scene) > 0:
                # Return the completed scene
                scenes.append(scene)
                scenes_to_remove.append(scene)
                self.clip_counter += 1
        
        # Remove finalized scenes
        for scene in scenes_to_remove:
            self.active_scenes.remove(scene)
        
        return scenes
    
    def _finalize_all_scenes(self) -> List[Scene]:
        """
        Finalize all remaining active scenes (called at EOF).
        
        Returns:
            List of completed Scenes
        """
        scenes = []
        
        for scene in self.active_scenes:
            if len(scene) > 0:
                scenes.append(scene)
                self.clip_counter += 1
        
        self.active_scenes.clear()
        return scenes
    
    def __repr__(self) -> str:
        return (f"SceneTracker(active_scenes={len(self.active_scenes)}, "
                f"clips_created={self.clip_counter})")