"""
YouTube Content Automation System
Complete pipeline for automated video creation from script/voice to final render
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core processing
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# Audio processing
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
import soundfile as sf

# Video editing
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, TextClip, 
    CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
)
from moviepy.video.fx import fadein, fadeout, resize
from moviepy.audio.fx.audio_normalize import audio_normalize

# TTS and Speech Recognition
import pyttsx3
try:
    import speech_recognition as sr
except ImportError:
    sr = None


class Config:
    """Configuration manager for the automation system"""
    
    DEFAULT_STYLE_BANK = {
        "tech": {
            "visual_prompt": "futuristic neon visuals, circuits, robots, digital cities, cyberpunk aesthetic",
            "color_scheme": ["#00ffff", "#ff00ff", "#0080ff"],
            "music_mood": "electronic"
        },
        "motivation": {
            "visual_prompt": "cinematic sunrise, mountains, running athlete, inspiring landscapes",
            "color_scheme": ["#ff6b35", "#f7931e", "#fdc830"],
            "music_mood": "uplifting"
        },
        "history": {
            "visual_prompt": "ancient architecture, sepia tones, slow pan, historical artifacts",
            "color_scheme": ["#8b7355", "#d4a574", "#f4e4c1"],
            "music_mood": "orchestral"
        },
        "education": {
            "visual_prompt": "clean diagrams, whiteboard animations, colorful infographics",
            "color_scheme": ["#4a90e2", "#50c878", "#f39c12"],
            "music_mood": "ambient"
        },
        "nature": {
            "visual_prompt": "wildlife, forests, oceans, aerial landscapes, documentary style",
            "color_scheme": ["#2ecc71", "#3498db", "#e67e22"],
            "music_mood": "calm"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.style_bank = self._load_style_bank()
        
    def _load_style_bank(self) -> Dict:
        """Load style bank from config file or use defaults"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f).get('style_bank', self.DEFAULT_STYLE_BANK)
        return self.DEFAULT_STYLE_BANK
    
    def save_default_config(self):
        """Save default configuration to file"""
        config = {
            'style_bank': self.DEFAULT_STYLE_BANK,
            'video_settings': {
                'resolution': [1920, 1080],
                'fps': 30,
                'bitrate': '8000k'
            },
            'audio_settings': {
                'sample_rate': 44100,
                'music_volume': 0.2,
                'voice_volume': 0.8
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)


class ScriptProcessor:
    """Process text scripts and segment into scenes"""
    
    def __init__(self):
        self.scene_delimiter = r'\n\n+|\. {2,}|---'
        
    def load_script(self, script_path: str) -> str:
        """Load script from file"""
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_metadata(self, script: str) -> Dict:
        """Extract title, tags, and description from script headers"""
        metadata = {
            'title': 'Untitled Video',
            'description': '',
            'tags': []
        }
        
        lines = script.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('# ') or line.startswith('Title:'):
                metadata['title'] = line.replace('#', '').replace('Title:', '').strip()
            elif line.startswith('Tags:'):
                tags = line.replace('Tags:', '').strip()
                metadata['tags'] = [t.strip() for t in tags.split(',')]
            elif line.startswith('Description:'):
                metadata['description'] = line.replace('Description:', '').strip()
        
        return metadata
    
    def segment_script(self, script: str, min_words: int = 15) -> List[Dict]:
        """Segment script into scenes based on punctuation and pauses"""
        # Remove metadata lines
        lines = [l for l in script.split('\n') if not l.startswith(('Title:', 'Tags:', 'Description:', '#'))]
        text = '\n'.join(lines).strip()
        
        # Split by paragraph breaks or multiple periods
        raw_segments = re.split(self.scene_delimiter, text)
        
        scenes = []
        current_scene = ""
        
        for segment in raw_segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Combine short segments
            current_scene += " " + segment
            word_count = len(current_scene.split())
            
            if word_count >= min_words or segment.endswith(('.', '!', '?')):
                scenes.append({
                    'text': current_scene.strip(),
                    'duration': self._estimate_duration(current_scene),
                    'word_count': len(current_scene.split())
                })
                current_scene = ""
        
        # Add remaining text
        if current_scene.strip():
            scenes.append({
                'text': current_scene.strip(),
                'duration': self._estimate_duration(current_scene),
                'word_count': len(current_scene.split())
            })
        
        return scenes
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate speaking duration (average 150 words per minute)"""
        words = len(text.split())
        return (words / 150) * 60  # Convert to seconds


class AudioProcessor:
    """Process audio files and generate voiceovers"""
    
    def __init__(self):
        self.tts_engine = None
        self.recognizer = sr.Recognizer() if sr else None
        
    def transcribe_audio(self, audio_path: str) -> Tuple[str, List[Dict]]:
        """Transcribe audio file to text with timing information"""
        if not sr:
            raise ImportError("speech_recognition not installed")
        
        audio = AudioSegment.from_file(audio_path)
        audio.export("temp_audio.wav", format="wav")
        
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
            except:
                text = "Transcription failed"
        
        # Get timing information using librosa
        y, sr_rate = librosa.load(audio_path)
        
        # Detect speech segments
        intervals = librosa.effects.split(y, top_db=20)
        
        segments = []
        words = text.split()
        words_per_segment = max(1, len(words) // len(intervals))
        
        for i, (start, end) in enumerate(intervals):
            start_time = start / sr_rate
            end_time = end / sr_rate
            
            word_start = i * words_per_segment
            word_end = min((i + 1) * words_per_segment, len(words))
            segment_text = ' '.join(words[word_start:word_end])
            
            segments.append({
                'text': segment_text,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
        
        os.remove("temp_audio.wav")
        return text, segments
    
    def generate_voiceover(self, text: str, output_path: str, rate: int = 150):
        """Generate TTS voiceover from text"""
        if not self.tts_engine:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', rate)
        
        self.tts_engine.save_to_file(text, output_path)
        self.tts_engine.runAndWait()
        
        # Normalize audio
        audio = AudioSegment.from_file(output_path)
        normalized = normalize(audio)
        normalized.export(output_path, format='wav')
        
        return output_path
    
    def analyze_audio_timing(self, audio_path: str) -> List[Dict]:
        """Analyze audio file for speech timing and pauses"""
        y, sr_rate = librosa.load(audio_path)
        
        # Detect speech/silence
        intervals = librosa.effects.split(y, top_db=20)
        
        segments = []
        for start, end in intervals:
            segments.append({
                'start': start / sr_rate,
                'end': end / sr_rate,
                'duration': (end - start) / sr_rate
            })
        
        return segments


class VisualGenerator:
    """Generate visuals and fetch stock footage"""
    
    def __init__(self, style_bank: Dict, output_dir: str = "assets/visuals"):
        self.style_bank = style_bank
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_placeholder_visual(self, style: str, scene_text: str, 
                                   duration: float, index: int) -> str:
        """Generate a placeholder visual (gradient + text overlay)"""
        width, height = 1920, 1080
        
        # Get color scheme
        colors = self.style_bank.get(style, {}).get('color_scheme', ['#3498db', '#2ecc71'])
        
        # Create gradient image
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Parse colors
        color1 = tuple(int(colors[0][i:i+2], 16) for i in (1, 3, 5))
        color2 = tuple(int(colors[1][i:i+2], 16) for i in (1, 3, 5))
        
        # Draw gradient
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Apply blur for aesthetic
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Add text overlay with scene number
        draw = ImageDraw.Draw(img)
        text = f"Scene {index + 1}"
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
        except:
            font = ImageFont.load_default()
        
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Draw text with shadow
        shadow_offset = 5
        draw.text((position[0] + shadow_offset, position[1] + shadow_offset), 
                 text, font=font, fill=(0, 0, 0, 128))
        draw.text(position, text, font=font, fill='white')
        
        # Save
        output_path = self.output_dir / f"scene_{index:03d}.png"
        img.save(output_path)
        
        return str(output_path)
    
    def create_kinetic_text_clip(self, text: str, duration: float, 
                                 fontsize: int = 60) -> TextClip:
        """Create animated text clip for captions"""
        # Split long text into multiple lines
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 40:  # Max chars per line
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        text = '\n'.join(lines)
        
        clip = TextClip(
            text, 
            fontsize=fontsize, 
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(1600, None),
            align='center'
        )
        
        return clip.set_duration(duration)


class VideoEditor:
    """Main video editing and composition engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.visual_gen = VisualGenerator(config.style_bank)
        
    def create_subtitle_clip(self, text: str, duration: float, 
                            position: Tuple[int, int] = (960, 900)) -> TextClip:
        """Create subtitle clip with styling"""
        clip = TextClip(
            text,
            fontsize=50,
            color='white',
            bg_color='rgba(0,0,0,0.6)',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(1600, None),
            align='center'
        ).set_duration(duration).set_position(('center', position[1]))
        
        return clip
    
    def add_transition(self, clip: VideoFileClip, transition_type: str = 'fade',
                      duration: float = 0.5) -> VideoFileClip:
        """Add transition effects to clip"""
        if transition_type == 'fade':
            clip = fadein(clip, duration)
            clip = fadeout(clip, duration)
        elif transition_type == 'zoom':
            # Zoom in effect
            clip = clip.resize(lambda t: 1 + 0.1 * (t / clip.duration))
        
        return clip
    
    def compose_scene(self, visual_path: str, audio_path: Optional[str],
                     subtitle_text: str, duration: float, 
                     scene_index: int) -> VideoFileClip:
        """Compose a single scene with visual, audio, and subtitles"""
        # Create video from image
        img_clip = ImageClip(visual_path).set_duration(duration)
        
        # Add ken burns effect (slow zoom and pan)
        zoom_factor = 1.2
        img_clip = img_clip.resize(lambda t: 1 + (zoom_factor - 1) * (t / duration))
        img_clip = img_clip.set_position(lambda t: (
            -100 * (t / duration), 
            -50 * (t / duration)
        ))
        
        # Add fade transitions
        img_clip = self.add_transition(img_clip, 'fade', 0.5)
        
        # Create subtitle
        subtitle = self.create_subtitle_clip(subtitle_text, duration)
        
        # Composite video and subtitle
        video = CompositeVideoClip([img_clip, subtitle], size=(1920, 1080))
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            audio = AudioFileClip(audio_path)
            video = video.set_audio(audio)
        
        return video
    
    def create_intro(self, title: str, duration: float = 3) -> VideoFileClip:
        """Create intro sequence"""
        # Create black background
        img = Image.new('RGB', (1920, 1080), color='black')
        img.save('temp_intro.png')
        
        bg = ImageClip('temp_intro.png').set_duration(duration)
        
        # Add title text
        title_clip = TextClip(
            title,
            fontsize=100,
            color='white',
            stroke_color='#3498db',
            stroke_width=3,
            method='caption',
            size=(1600, None)
        ).set_duration(duration).set_position('center')
        
        # Fade in/out
        title_clip = fadein(fadeout(title_clip, 0.5), 0.5)
        
        intro = CompositeVideoClip([bg, title_clip], size=(1920, 1080))
        
        os.remove('temp_intro.png')
        return intro
    
    def create_outro(self, text: str = "Thanks for watching!", 
                    duration: float = 3) -> VideoFileClip:
        """Create outro sequence"""
        # Create gradient background
        img = Image.new('RGB', (1920, 1080))
        draw = ImageDraw.Draw(img)
        
        for y in range(1080):
            ratio = y / 1080
            color = (
                int(52 * (1 - ratio) + 155 * ratio),
                int(152 * (1 - ratio) + 89 * ratio),
                int(219 * (1 - ratio) + 182 * ratio)
            )
            draw.line([(0, y), (1920, y)], fill=color)
        
        img.save('temp_outro.png')
        
        bg = ImageClip('temp_outro.png').set_duration(duration)
        
        # Add text
        text_clip = TextClip(
            text,
            fontsize=80,
            color='white',
            stroke_color='black',
            stroke_width=2
        ).set_duration(duration).set_position('center')
        
        text_clip = fadein(fadeout(text_clip, 0.5), 0.5)
        
        outro = CompositeVideoClip([bg, text_clip], size=(1920, 1080))
        
        os.remove('temp_outro.png')
        return outro
    
    def add_background_music(self, video: VideoFileClip, 
                            music_path: Optional[str],
                            music_volume: float = 0.15) -> VideoFileClip:
        """Add background music with volume ducking"""
        if not music_path or not os.path.exists(music_path):
            return video
        
        music = AudioFileClip(music_path)
        
        # Loop music if shorter than video
        if music.duration < video.duration:
            n_loops = int(video.duration / music.duration) + 1
            music = concatenate_videoclips([music] * n_loops)
        
        # Trim to video length
        music = music.subclip(0, video.duration)
        
        # Reduce volume
        music = music.volumex(music_volume)
        
        # Combine with existing audio
        if video.audio:
            final_audio = CompositeAudioClip([video.audio, music])
            video = video.set_audio(final_audio)
        else:
            video = video.set_audio(music)
        
        return video
    
    def generate_thumbnail(self, video_path: str, title: str, 
                          output_path: str) -> str:
        """Generate thumbnail from video mid-frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Create blank thumbnail
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Convert to PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Add title overlay
        draw = ImageDraw.Draw(img)
        
        # Add semi-transparent overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 128))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Add title text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        except:
            font = ImageFont.load_default()
        
        # Word wrap title
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > 1700:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        title = '\n'.join(lines[:2])  # Max 2 lines
        
        # Center text
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((1920 - text_width) // 2, (1080 - text_height) // 2)
        
        # Draw with shadow
        draw.text((position[0] + 5, position[1] + 5), title, font=font, fill='black')
        draw.text(position, title, font=font, fill='white')
        
        img.save(output_path)
        return output_path


class YouTubeAutomation:
    """Main orchestrator for YouTube video automation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.script_processor = ScriptProcessor()
        self.audio_processor = AudioProcessor()
        self.video_editor = VideoEditor(self.config)
        
        # Create output directories
        self.dirs = {
            'assets': Path('assets'),
            'audio': Path('assets/audio'),
            'visuals': Path('assets/visuals'),
            'output': Path('output')
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process(self, script_path: Optional[str] = None,
                audio_path: Optional[str] = None,
                style: str = 'tech',
                voiceover: bool = True,
                music_path: Optional[str] = None,
                output_path: str = 'output/final_video.mp4') -> Dict:
        """
        Main processing pipeline
        
        Args:
            script_path: Path to text script
            audio_path: Path to audio narration
            style: Visual style from style bank
            voiceover: Generate TTS if True and no audio provided
            music_path: Path to background music
            output_path: Output video path
            
        Returns:
            Dictionary with output paths and metadata
        """
        print("🎬 Starting YouTube Automation Pipeline...")
        
        # Step 1: Process script or transcribe audio
        if script_path:
            print("\n📝 Processing script...")
            script = self.script_processor.load_script(script_path)
            metadata = self.script_processor.extract_metadata(script)
            scenes = self.script_processor.segment_script(script)
            print(f"   ✓ Extracted {len(scenes)} scenes")
            
        elif audio_path:
            print("\n🎤 Transcribing audio...")
            script, audio_segments = self.audio_processor.transcribe_audio(audio_path)
            metadata = {'title': 'Transcribed Video', 'description': '', 'tags': []}
            scenes = [{
                'text': seg['text'],
                'duration': seg['duration'],
                'start': seg['start'],
                'end': seg['end']
            } for seg in audio_segments]
            print(f"   ✓ Transcribed {len(scenes)} segments")
        else:
            raise ValueError("Either script_path or audio_path must be provided")
        
        # Step 2: Generate or process audio
        scene_audio_paths = []
        
        if audio_path and os.path.exists(audio_path):
            print("\n🔊 Using provided audio...")
            # Use existing audio - split by scenes
            audio = AudioSegment.from_file(audio_path)
            
            for i, scene in enumerate(scenes):
                start_ms = int(scene.get('start', i * 3) * 1000)
                end_ms = int(scene.get('end', (i + 1) * 3) * 1000)
                
                scene_audio = audio[start_ms:end_ms]
                audio_out = self.dirs['audio'] / f"scene_{i:03d}.wav"
                scene_audio.export(audio_out, format='wav')
                scene_audio_paths.append(str(audio_out))
                
        elif voiceover:
            print("\n🗣️  Generating TTS voiceover...")
            for i, scene in enumerate(scenes):
                audio_out = self.dirs['audio'] / f"scene_{i:03d}.wav"
                self.audio_processor.generate_voiceover(
                    scene['text'], 
                    str(audio_out)
                )
                scene_audio_paths.append(str(audio_out))
                
                # Update duration based on actual audio
                audio = AudioSegment.from_file(str(audio_out))
                scene['duration'] = len(audio) / 1000.0
            
            print(f"   ✓ Generated voiceover for {len(scenes)} scenes")
        
        # Step 3: Generate visuals
        print("\n🎨 Generating visuals...")
        visual_paths = []
        
        for i, scene in enumerate(scenes):
            visual_path = self.video_editor.visual_gen.generate_placeholder_visual(
                style, 
                scene['text'], 
                scene['duration'],
                i
            )
            visual_paths.append(visual_path)
        
        print(f"   ✓ Generated {len(visual_paths)} visual scenes")
        
        # Step 4: Compose scenes
        print("\n🎞️  Composing video scenes...")
        scene_clips = []
        
        for i, scene in enumerate(scenes):
            audio = scene_audio_paths[i] if i < len(scene_audio_paths) else None
            
            clip = self.video_editor.compose_scene(
                visual_paths[i],
                audio,
                scene['text'],
                scene['duration'],
                i
            )
            scene_clips.append(clip)
        
        print(f"   ✓ Composed {len(scene_clips)} scenes")
        
        # Step 5: Add intro/outro
        print("\n🎬 Adding intro and outro...")
        intro = self.video_editor.create_intro(metadata['title'], 3)
        outro = self.video_editor.create_outro("Thanks for watching! Subscribe for more.", 3)
        
        # Step 6: Concatenate all clips
        print("\n🔗 Concatenating clips...")
        final_clips = [intro] + scene_clips + [outro]
        final_video = concatenate_videoclips(final_clips, method='compose')
        
        # Step 7: Add background music
        if music_path:
            print("\n🎵 Adding background music...")
            final_video = self.video_editor.add_background_music(
                final_video, 
                music_path,
                music_volume=0.15
            )
        
        # Step 8: Render final video
        print(f"\n📹 Rendering final video to {output_path}...")
        final_video.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            preset='medium',
            threads=4
        )
        
        print("   ✓ Video rendered successfully!")
        
        # Step 9: Generate thumbnail
        print("\n🖼️  Generating thumbnail...")
        thumbnail_path = output_path.replace('.mp4', '_thumbnail.jpg')
        self.video_editor.generate_thumbnail(
            output_path,
            metadata['title'],
            thumbnail_path
        )
        
        # Step 10: Save metadata
        print("\n📄 Saving metadata...")
        metadata_path = output_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✅ Pipeline complete!")
        
        results = {
            'video_path': output_path,
            'thumbnail_path': thumbnail_path,
            'metadata_path': metadata_path,
            'metadata': metadata,
            'scene_count': len(scenes),
            'duration': final_video.duration
        }
        
        # Cleanup
        for clip in scene_clips:
            clip.close()
        final_video.close()
        intro.close()
        outro.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Content Automation System'
    )
    parser.add_argument('--script', type=str, help='Path to script file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--style', type=str, default='tech', 
                       help='Visual style (tech, motivation, history, education, nature)')
    parser.add_argument('--voiceover', action='store_true',
                       help='Generate TTS voiceover')
    parser.add_argument('--music', type=str, help='Path to background music')
    parser.add_argument('--output', type=str, default='output/final_video.mp4',
                       help='Output video path')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default config file')
    
    args = parser.parse_args()
    
    # Create config if requested
    if args.create_config:
        config = Config()
        config.save_default_config()
        print("✓ Default config.json created")
        return
    
    # Validate inputs
    if not args.script and not args.audio:
        parser.error("Either --script or --audio must be provided")
    
    # Initialize automation system
    automation = YouTubeAutomation(args.config)
    
    # Process video
    results = automation.process(
        script_path=args.script,
        audio_path=args.audio,
        style=args.style,
        voiceover=args.voiceover,
        music_path=args.music,
        output_path=args.output
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 VIDEO GENERATION SUMMARY")
    print("="*60)
    print(f"Title: {results['metadata']['title']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Scenes: {results['scene_count']}")
    print(f"\n📁 Output Files:")
    print(f"   Video: {results['video_path']}")
    print(f"   Thumbnail: {results['thumbnail_path']}")
    print(f"   Metadata: {results['metadata_path']}")
    print("="*60)


if __name__ == '__main__':
    main()