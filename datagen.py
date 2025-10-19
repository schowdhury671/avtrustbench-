import json
import os
import subprocess
import codecs
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

# Configuration
DATA_DIR = "./"
OUTPUT_DIR = "output/"

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))

def safe_ffmpeg_run(command):
    """Safely run ffmpeg command with error handling"""
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return False

def extract_clip_moviepy(video_path, output_path, start_time=0, duration=1):
    """Extract a clip from video using MoviePy"""
    try:
        clip = VideoFileClip(video_path)
        end_time = min(start_time + duration, clip.duration)
        extracted_clip = clip.subclip(start_time, end_time)
        extracted_clip.write_videofile(output_path, verbose=False, logger=None)
        clip.close()
        extracted_clip.close()
        return True
    except Exception as e:
        print(f"MoviePy error extracting clip: {e}")
        return False

def swap_audio_ffmpeg(video1_path, video2_path, output_path):
    """Swap audio between two videos using ffmpeg"""
    command = f'ffmpeg -y -i "{video1_path}" -i "{video2_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    return safe_ffmpeg_run(command)

def swap_video_ffmpeg(video1_path, video2_path, output_path):
    """Swap video between two videos (keep audio from first) using ffmpeg"""
    command = f'ffmpeg -y -i "{video1_path}" -i "{video2_path}" -map 0:a -map 1:v -c:v copy -c:a aac "{output_path}"'
    return safe_ffmpeg_run(command)

def remove_audio_ffmpeg(video_path, output_path):
    """Remove audio from video using ffmpeg"""
    command = f'ffmpeg -y -i "{video_path}" -c:v copy -an "{output_path}"'
    return safe_ffmpeg_run(command)

def extract_audio_only_ffmpeg(video_path, output_path):
    """Extract only audio from video using ffmpeg"""
    command = f'ffmpeg -y -i "{video_path}" -vn -c:a aac "{output_path}"'
    return safe_ffmpeg_run(command)

def concatenate_videos_moviepy(video1_path, video2_path, output_path):
    """Concatenate two videos using MoviePy"""
    try:
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)
        
        final_clip = concatenate_videoclips([clip1, clip2], method='compose')
        final_clip.write_videofile(output_path, verbose=False, logger=None)
        
        # Clean up
        clip1.close()
        clip2.close()
        final_clip.close()
        return True
    except Exception as e:
        print(f"MoviePy error concatenating videos: {e}")
        return False

def process_mait(json_path):
    """Process MAIT (Misaligned Audio-Visual) dataset"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing MAIT dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="MAIT Processing"):
        try:
            video1, video2 = js['video'].split('#')
            video1_path = os.path.join(DATA_DIR, f"{video1}.mp4")
            video2_path = os.path.join(DATA_DIR, f"{video2}.mp4")
            output_path = os.path.join(OUTPUT_DIR, 'mait', f"{js['video']}.mp4")
            
            if os.path.exists(video1_path) and os.path.exists(video2_path):
                success = swap_audio_ffmpeg(video1_path, video2_path, output_path)
                if success:
                    processed_count += 1
                    print(f"MAIT: Generated {output_path}")
                else:
                    print(f"MAIT: Failed {output_path}")
            else:
                print(f"Missing files: {video1_path} or {video2_path}")
        except Exception as e:
            print(f"Error processing MAIT {js['video']}: {e}")
    
    print(f"MAIT completed: {processed_count}/{len(js_data)} videos processed")

def process_mvit(json_path):
    """Process MVIT (Misaligned Visual) dataset"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing MVIT dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="MVIT Processing"):
        try:
            video1, video2 = js['video'].split('#')
            video1_path = os.path.join(DATA_DIR, f"{video1}.mp4")
            video2_path = os.path.join(DATA_DIR, f"{video2}.mp4")
            output_path = os.path.join(OUTPUT_DIR, 'mvit', f"{js['video']}.mp4")
            
            if os.path.exists(video1_path) and os.path.exists(video2_path):
                success = swap_video_ffmpeg(video1_path, video2_path, output_path)
                if success:
                    processed_count += 1
                    print(f"MVIT: Generated {output_path}")
                else:
                    print(f"MVIT: Failed {output_path}")
            else:
                print(f"Missing files: {video1_path} or {video2_path}")
        except Exception as e:
            print(f"Error processing MVIT {js['video']}: {e}")
    
    print(f"MVIT completed: {processed_count}/{len(js_data)} videos processed")

def process_stitch(json_path):
    """Process COT-Stitch dataset"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing COT-Stitch dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="Stitch Processing"):
        try:
            video1, video2 = js['video'].split('#')
            video1_path = os.path.join(DATA_DIR, f"{video1}.mp4")
            video2_path = os.path.join(DATA_DIR, f"{video2}.mp4")
            
            # Create temporary clips
            temp_clip1 = f"temp_{video1}_1s.mp4"
            temp_clip2 = f"temp_{video2}_1s.mp4"
            output_path = os.path.join(OUTPUT_DIR, 'stitch', f"{js['video']}.mp4")
            
            if os.path.exists(video1_path) and os.path.exists(video2_path):
                # Concatenate clips using MoviePy
                success = concatenate_videos_moviepy(video1_path, video2_path, output_path)
                
                # Cleanup
                for temp_file in [temp_clip1, temp_clip2]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                if success:
                    processed_count += 1
                    print(f"STITCH: Generated {output_path}")
                else:
                    print(f"STITCH: Failed {output_path}")
            else:
                print(f"Missing files: {video1_path} or {video2_path}")
        except Exception as e:
            print(f"Error processing STITCH {js['video']}: {e}")
    
    print(f"STITCH completed: {processed_count}/{len(js_data)} videos processed")

def process_swap(json_path):
    """Process COT-Swap dataset (full audio-visual swap)"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing COT-Swap dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="Swap Processing"):
        try:
            video1, video2 = js['video'].split('#')
            video1_path = os.path.join(DATA_DIR, f"{video1}.mp4")
            video2_path = os.path.join(DATA_DIR, f"{video2}.mp4")
            
            # Create swapped versions
            temp_swap1 = f"temp_swap1_{video1}#{video2}.mp4"
            temp_swap2 = f"temp_swap2_{video2}#{video1}.mp4"
            output_path = os.path.join(OUTPUT_DIR, 'swap', f"{js['video']}.mp4")
            
            if os.path.exists(video1_path) and os.path.exists(video2_path):
                # Create swapped videos: video1 visual + video2 audio, video2 visual + video1 audio
                if (swap_audio_ffmpeg(video1_path, video2_path, temp_swap1) and
                    swap_audio_ffmpeg(video2_path, video1_path, temp_swap2)):
                    
                    # Concatenate the swapped videos using MoviePy
                    success = concatenate_videos_moviepy(temp_swap1, temp_swap2, output_path)
                    
                    # Cleanup
                    for temp_file in [temp_swap1, temp_swap2]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    if success:
                        processed_count += 1
                        print(f"SWAP: Generated {output_path}")
                    else:
                        print(f"SWAP: Failed {output_path}")
            else:
                print(f"Missing files: {video1_path} or {video2_path}")
        except Exception as e:
            print(f"Error processing SWAP {js['video']}: {e}")
    
    print(f"SWAP completed: {processed_count}/{len(js_data)} videos processed")

def process_mat(json_path):
    """Process MAT (Missing Audio) dataset"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing MAT dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="MAT Processing"):
        try:
            video_path = os.path.join(DATA_DIR, f"{js['video']}.mp4")
            output_path = os.path.join(OUTPUT_DIR, 'mat', f"{js['video']}.mp4")
            
            if os.path.exists(video_path):
                success = remove_audio_ffmpeg(video_path, output_path)
                if success:
                    processed_count += 1
                    print(f"MAT: Generated {output_path}")
                else:
                    print(f"MAT: Failed {output_path}")
            else:
                print(f"Missing file: {video_path}")
        except Exception as e:
            print(f"Error processing MAT {js['video']}: {e}")
    
    print(f"MAT completed: {processed_count}/{len(js_data)} videos processed")

def process_mvt(json_path):
    """Process MVT (Missing Visual) dataset"""
    js_data = json_load(json_path)
    processed_count = 0
    
    print(f"Processing MVT dataset: {len(js_data)} items")
    
    for js in tqdm(js_data, desc="MVT Processing"):
        try:
            video_path = os.path.join(DATA_DIR, f"{js['video']}.mp4")
            output_path = os.path.join(OUTPUT_DIR, 'mvt', f"{js['video']}.mp4")
            
            if os.path.exists(video_path):
                success = extract_audio_only_ffmpeg(video_path, output_path)
                if success:
                    processed_count += 1
                    print(f"MVT: Generated {output_path}")
                else:
                    print(f"MVT: Failed {output_path}")
            else:
                print(f"Missing file: {video_path}")
        except Exception as e:
            print(f"Error processing MVT {js['video']}: {e}")
    
    print(f"MVT completed: {processed_count}/{len(js_data)} videos processed")

def setup_directories():
    """Create necessary output directories"""
    dirs = ['mait', 'mvit', 'stitch', 'swap', 'mat', 'mvt']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for dir_name in dirs:
        os.makedirs(os.path.join(OUTPUT_DIR, dir_name), exist_ok=True)

def main():
    """Main function to run all data generation tasks"""
    setup_directories()
    
    # Example Usage
    ## Configuration
    base_path = "AVTrustBench-10K/"
    
    tasks = {
        "MAIT": ("MAIT/avqa/instruct_test_MAIT.json", process_mait),
        "MVIT": ("MVIT/avqa/instruct_test_MVIT.json", process_mvit),
        "COT-Stitch": ("COT-Stitch/instruct_test_COT_Stitch.json", process_stitch),
        "COT-Swap": ("COT-Swap/instruct_test_COT_Swap.json", process_swap),
        "MAT": ("MAT/music_avqa/instruct_test_MAT.json", process_mat),
        "MVT": ("MVT/music_avqa/instruct_test_MVT.json", process_mvt),
    }
    
    for task_name, (json_path, process_func) in tasks.items():
        print(f"\n{'='*50}")
        print(f"Starting {task_name} processing...")
        print(f"{'='*50}")
        
        full_json_path = os.path.join(base_path, json_path)
        if os.path.exists(full_json_path):
            process_func(full_json_path)
        else:
            print(f"JSON file not found: {full_json_path}")
    
    print("\nAll data generation tasks completed!")

if __name__ == "__main__":
    main()
