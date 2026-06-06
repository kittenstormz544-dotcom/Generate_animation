"""
Kaggle Animation Engine
Watches Supabase for pending generations and creates animations
Forces GPU usage when available
"""

import os
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import uuid

# ============================================
# FORCE GPU
# ============================================
print("=" * 60)
print("ANIMATION ENGINE STARTING")
print("=" * 60)

# Check and force GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print(f"✓ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("⚠ GPU NOT AVAILABLE - Using CPU (slower)")

print(f"Using device: {device}")
print("=" * 60)

# ============================================
# SUPABASE CONFIG
# ============================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("❌ ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
    print("Make sure these are set in Kaggle Secrets")
    exit(1)

print(f"✓ Connected to Supabase: {SUPABASE_URL}")

# ============================================
# ACTION MAPPING
# ============================================
ACTION_PROMPTS = {
    "walk left": "character walking to the left",
    "walk right": "character walking to the right",
    "walk forward": "character walking towards camera",
    "walk backward": "character walking away",
    "jump": "character jumping high in the air",
    "jump up": "character jumping up",
    "wave": "character waving hand friendly",
    "wave hand": "character waving",
    "sit down": "character sitting down smoothly",
    "sit": "character sitting down",
    "stand up": "character standing up",
    "point": "character pointing forward",
    "spin": "character spinning around",
    "nod": "character nodding yes",
    "shake": "character shaking head no",
    "shrug": "character shrugging shoulders",
    "sad": "character looking sad",
    "happy": "character smiling happily",
    "angry": "character looking angry",
    "surprised": "character looking surprised",
    "confused": "character looking confused",
    "excited": "character excited and energetic",
    "tired": "character tired",
    "dance": "character dancing",
    "clap": "character clapping hands",
    "thumbs up": "character giving thumbs up",
    "thumbs down": "character giving thumbs down",
    "laugh": "character laughing",
    "cry": "character crying",
    "run": "character running",
    "sit upset": "character sitting upset",
    "sits down upsettedly": "character sitting down upset",
}

# ============================================
# SUPABASE FUNCTIONS
# ============================================

def get_pending_generations():
    """Fetch pending generations from Supabase"""
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    
    url = f"{SUPABASE_URL}/rest/v1/generations?status=eq.pending&limit=1"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            print(f"✓ Found pending generation")
        return data
    except Exception as e:
        print(f"❌ Error fetching pending generations: {e}")
        return []

def update_generation_status(generation_id, status, video_url=None, error=None):
    """Update generation status in Supabase"""
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    if video_url:
        data["video_url"] = video_url
    
    if error:
        data["error_message"] = error
    
    url = f"{SUPABASE_URL}/rest/v1/generations?id=eq.{generation_id}"
    
    try:
        response = requests.patch(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        if status == "complete":
            print(f"✓ Status updated to COMPLETE")
        elif status == "processing":
            print(f"✓ Status updated to PROCESSING")
        else:
            print(f"✓ Status updated to {status}")
            
    except Exception as e:
        print(f"❌ Error updating status: {e}")

def download_image_from_url(image_url):
    """Download character image from URL"""
    try:
        print(f"  Downloading character image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        temp_path = f"/tmp/character_{uuid.uuid4()}.png"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        print(f"  ✓ Image downloaded")
        return temp_path
    except Exception as e:
        print(f"  ❌ Error downloading image: {e}")
        return None

def upload_video_to_supabase(video_path, generation_id):
    """Upload generated video to Supabase Storage"""
    try:
        print(f"  Uploading video to Supabase...")
        
        with open(video_path, "rb") as f:
            video_data = f.read()
        
        filename = f"generation_{generation_id}.mp4"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        
        url = f"{SUPABASE_URL}/storage/v1/object/generated-videos/{filename}"
        
        response = requests.post(url, headers=headers, data=video_data, timeout=60)
        response.raise_for_status()
        
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/generated-videos/{filename}"
        print(f"  ✓ Video uploaded successfully")
        
        return public_url
    except Exception as e:
        print(f"  ❌ Error uploading video: {e}")
        return None

# ============================================
# ANIMATION GENERATION
# ============================================

def generate_animation(character_image_path, actions, character_name, output_path):
    """Generate animation using PIL and OpenCV"""
    try:
        print(f"  Generating animation...")
        
        # Load character image
        character = Image.open(character_image_path).convert("RGBA")
        print(f"  ✓ Character image loaded: {character.size}")
        
        # Generate frames
        frames = []
        frame_count = 0
        total_frames = len(actions) * 72  # 72 frames per action (3 seconds at 24fps)
        
        for action_idx, action in enumerate(actions):
            print(f"    Action {action_idx + 1}/{len(actions)}: {action}")
            
            # Generate 72 frames per action (3 seconds at 24fps)
            for frame_idx in range(72):
                # Create frame from character image
                frame = apply_action_effect(character, action, frame_idx, 72)
                frames.append(frame)
                frame_count += 1
                
                # Progress
                if frame_idx % 24 == 0:
                    print(f"      Frame {frame_idx + 1}/72", end='\r')
        
        print(f"  ✓ Generated {frame_count} frames")
        
        # Create video from frames
        print(f"  Creating video file...")
        success = create_video_from_frames(frames, output_path, fps=24)
        
        if success:
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ Video created: {file_size:.2f} MB")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"  ❌ Error generating animation: {e}")
        return False

def apply_action_effect(image, action, frame_index, total_frames):
    """Apply animation effects based on action"""
    try:
        frame = image.copy()
        action_lower = action.lower()
        progress = frame_index / total_frames  # 0 to 1
        
        # Horizontal movement (walk, run)
        if "walk" in action_lower or "run" in action_lower:
            if "left" in action_lower:
                offset = int(progress * 80 - 40)  # -40 to +40
            elif "right" in action_lower:
                offset = int(-progress * 80 + 40)  # +40 to -40
            elif "forward" in action_lower or "towards" in action_lower:
                # Scale up slightly
                scale_factor = 1.0 + (progress * 0.1)
                new_size = (int(frame.width * scale_factor), int(frame.height * scale_factor))
                frame = frame.resize(new_size, Image.LANCZOS)
                # Center it
                offset_x = (new_size[0] - frame.width) // 2
                offset_y = (new_size[1] - frame.height) // 2
                return frame
            else:
                # backward
                scale_factor = 1.0 - (progress * 0.1)
                new_size = (int(frame.width * scale_factor), int(frame.height * scale_factor))
                frame = frame.resize(new_size, Image.LANCZOS)
                return frame
            
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, offset, 0, 1, 0),
                Image.BILINEAR
            )
        
        # Vertical movement (jump)
        elif "jump" in action_lower:
            offset = int(abs(np.sin(progress * np.pi)) * 50)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, -offset),
                Image.BILINEAR
            )
        
        # Rotation (spin)
        elif "spin" in action_lower:
            angle = progress * 360
            frame = frame.rotate(angle, expand=False, resample=Image.BILINEAR)
        
        # Wave (subtle oscillation)
        elif "wave" in action_lower:
            offset = int(np.sin(progress * np.pi * 6) * 8)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, offset, 0, 1, 0),
                Image.BILINEAR
            )
        
        # Sit down
        elif "sit" in action_lower:
            # Move down
            offset = int(progress * 40)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, offset),
                Image.BILINEAR
            )
        
        # Point
        elif "point" in action_lower:
            offset = int(np.sin(progress * np.pi * 4) * 5)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, offset, 0, 1, 0),
                Image.BILINEAR
            )
        
        # Nod
        elif "nod" in action_lower:
            offset = int(np.sin(progress * np.pi * 3) * 5)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, offset),
                Image.BILINEAR
            )
        
        # Shake (head no)
        elif "shake" in action_lower:
            offset = int(np.sin(progress * np.pi * 6) * 10)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, offset, 0, 1, 0),
                Image.BILINEAR
            )
        
        # Default: slight wobble
        else:
            offset = int(np.sin(progress * np.pi * 2) * 3)
            frame = frame.transform(
                frame.size,
                Image.AFFINE,
                (1, 0, offset, 0, 1, 0),
                Image.BILINEAR
            )
        
        return frame
    
    except Exception as e:
        print(f"  ❌ Error applying effect: {e}")
        return image

def create_video_from_frames(frames, output_path, fps=24):
    """Create MP4 video from frames"""
    if not frames:
        print("  ❌ No frames to create video")
        return False
    
    try:
        frame_width, frame_height = frames[0].size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            print("  ❌ Failed to open video writer")
            return False
        
        for i, frame in enumerate(frames):
            frame_np = np.array(frame.convert("RGB"))
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return True
    
    except Exception as e:
        print(f"  ❌ Error creating video: {e}")
        return False

# ============================================
# MAIN ENGINE LOOP
# ============================================

def main():
    """Main engine loop - watches for pending generations"""
    print("\n🚀 ANIMATION ENGINE RUNNING")
    print("Watching for pending generations...\n")
    
    check_interval = 5  # Check every 5 seconds
    
    while True:
        try:
            # Check for pending generations
            pending = get_pending_generations()
            
            if pending:
                generation = pending[0]
                generation_id = generation["id"]
                character_name = generation["character_name"]
                character_image_url = generation["character_image_url"]
                parsed_actions = generation.get("parsed_actions", [])
                
                print(f"\n{'='*60}")
                print(f"📹 NEW GENERATION REQUEST")
                print(f"{'='*60}")
                print(f"ID: {generation_id}")
                print(f"Character: {character_name}")
                print(f"Actions: {parsed_actions}")
                print(f"{'='*60}\n")
                
                # Update status to processing
                update_generation_status(generation_id, "processing")
                
                # Download character image
                image_path = download_image_from_url(character_image_url)
                if not image_path:
                    update_generation_status(
                        generation_id,
                        "failed",
                        error="Failed to download character image"
                    )
                    continue
                
                # Generate animation
                temp_video_path = f"/tmp/animation_{generation_id}.mp4"
                print(f"Generating animation...")
                success = generate_animation(
                    image_path,
                    parsed_actions,
                    character_name,
                    temp_video_path
                )
                
                if success and os.path.exists(temp_video_path):
                    # Upload to Supabase
                    video_url = upload_video_to_supabase(temp_video_path, generation_id)
                    
                    if video_url:
                        update_generation_status(
                            generation_id,
                            "complete",
                            video_url=video_url
                        )
                        print(f"\n✅ GENERATION COMPLETE!\n")
                    else:
                        update_generation_status(
                            generation_id,
                            "failed",
                            error="Failed to upload video"
                        )
                    
                    # Cleanup
                    try:
                        os.remove(temp_video_path)
                        os.remove(image_path)
                    except:
                        pass
                else:
                    update_generation_status(
                        generation_id,
                        "failed",
                        error="Failed to generate animation"
                    )
                    print(f"\n❌ GENERATION FAILED\n")
            
            else:
                # No pending generations - wait and check again
                now = datetime.now().strftime('%H:%M:%S')
                print(f"[{now}] Waiting... (checking every {check_interval}s)", end='\r')
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Engine stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Engine error: {e}")
            print("Retrying in 10 seconds...\n")
            time.sleep(10)

if __name__ == "__main__":
    main()
