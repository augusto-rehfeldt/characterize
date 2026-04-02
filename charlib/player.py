import argparse
import time
import sys
import os
import cv2
import pygame
import subprocess
from PIL import Image
from charlib.image_utils import get_chars, sample_colors
import rank
import characters
import tempfile
import numpy as np

class VideoPlayer:
    def __init__(self, args):
        self.args = args
        self.fonts = {
            "ascii": "arial.ttf", "arabic": "arial.ttf", "braille": "seguisym.ttf",
            "emoji": "seguiemj.ttf", "chinese": "msyh.ttc", "simple": "arial.ttf",
            "numbers+": "arial.ttf", "roman": "times.ttf", "numbers": "arial.ttf",
            "latin": "arial.ttf", "hiragana": "msyh.ttc", "katakana": "msyh.ttc",
            "kanji": "msyh.ttc", "cyrillic": "arial.ttf", "hangul": "malgunbd.ttf",
        }
        self.detail_map = {
            "braille": 16, "hiragana": 15, "katakana": 15, "kanji": 15,
            "chinese": 15, "hangul": 15, "arabic": 15
        }
        self.CONTROLS_AREA_HEIGHT = 100
        self.char_list = self._build_char_list()
        self.cap = cv2.VideoCapture(self.args.input)
        if not self.cap.isOpened():
            print(f"Error: cannot open video {self.args.input}")
            sys.exit(1)
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cv_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.args.framerate or self.cv_fps or 24.0
        self.delay = 1.0 / self.fps
        self.target_w_chars = self.args.width
        self.target_h_chars = self._calculate_target_height()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps if self.total_frames > 0 and self.fps > 0 else 0
        self.screen = None
        self.font = None
        self.char_width, self.char_height = 1, 1
        self.font_size = 15
        self.playback_paused = False
        self.last_frame = None
        self.window_sizes = {
            'small': (640, 480 + self.CONTROLS_AREA_HEIGHT),
            'medium': (800, 600 + self.CONTROLS_AREA_HEIGHT),
            'large': (1024, 768 + self.CONTROLS_AREA_HEIGHT)
        }
        self.current_size_key = 'medium'
        self.original_window_size = self.window_sizes[self.current_size_key]
        self.volume_icon = None
        self.fullscreen = False
        self.volume = 1.0
        self.slider_width, self.slider_height = 100, 10
        self.slider_knob_radius = 7
        self.timeline_height = 15
        self.timeline_knob_radius = 10
        self.current_time = 0.0
        self.frame_count = 0
        self.audio_loaded = False
        self.temp_audio_path = None
        self.button_hover = False
        self.dragging_volume = False
        self.dragging_timeline = False
        self.play_button_text = "Pause"
        self.button_rect = None
        self.timeline_rect = None
        self.extended_timeline_rect = None
        self.slider_rect = None
        self.button_font = None
        self.pygame_font_path = self._find_pygame_font()

    def _find_pygame_font(self):
        pygame.init()
        pygame.mixer.init()
        mono_fonts = ['Consolas', 'Courier New', 'monospace', 'DejaVu Sans Mono']
        for font_name in mono_fonts:
            try:
                path = pygame.font.match_font(font_name)
                if path:
                    return path
            except Exception:
                continue
        print("Warning: Could not find a suitable monospace font. Using Pygame default.")
        return pygame.font.get_default_font()

    def _build_char_list(self):
        detail = self.detail_map.get(self.args.language, 12)
        font_file = self.fonts.get(self.args.language, self.fonts["ascii"])
        allowed = characters.dict_caracteres.get(self.args.language, characters.dict_caracteres["ascii"])
        char_rank = rank.create_ranking(
            detail, font=font_file,
            list_size=self.args.complexity,
            allowed_characters=allowed
        )
        if self.args.empty:
            char_rank.append((" ", 0))
            char_rank = sorted(char_rank, key=lambda x: x[1])[:self.args.complexity]
        return [c for c, _ in char_rank]

    def _calculate_target_height(self):
        if self.args.height:
            return self.args.height
        aspect_ratio_correction = 0.5
        if self.orig_w > 0:
            return max(1, int(self.orig_h / self.orig_w * self.target_w_chars * aspect_ratio_correction))
        return max(1, int(self.target_w_chars * aspect_ratio_correction))

    def _calculate_font_for_area(self, video_area_width, video_area_height):
        if self.target_w_chars <= 0 or self.target_h_chars <= 0:
            self.target_w_chars = max(1, self.target_w_chars)
            self.target_h_chars = max(1, self.target_h_chars)

        min_font_size = 5
        best_fs = min_font_size
        try:
            best_font = pygame.font.Font(self.pygame_font_path, best_fs)
        except pygame.error as e:
            print(f"Error loading font {self.pygame_font_path} at min_size {min_font_size}: {e}")
            self.pygame_font_path = pygame.font.get_default_font()
            best_font = pygame.font.Font(self.pygame_font_path, best_fs)

        final_char_w, final_char_h = best_font.size(" ")
        final_char_w = max(1, final_char_w)
        final_char_h = max(1, final_char_h)

        if video_area_width < self.target_w_chars or video_area_height < self.target_h_chars:
            print(f"Warning: Video area too small for grid. Using min font size {best_fs}.")
            return best_font, final_char_w, final_char_h, best_fs

        max_test_fs = 200
        for fs_test in range(min_font_size + 1, max_test_fs + 1):
            try:
                current_font = pygame.font.Font(self.pygame_font_path, fs_test)
                test_w, test_h = current_font.size("W")
                if test_w == 0 or test_h == 0: continue
                if (test_w * self.target_w_chars <= video_area_width and
                    test_h * self.target_h_chars <= video_area_height):
                    best_fs = fs_test
                    best_font = current_font
                    final_char_w, final_char_h = current_font.size(" ")
                    final_char_w = max(1, final_char_w)
                    final_char_h = max(1, final_char_h)
                else:
                    break
            except pygame.error:
                break
        return best_font, final_char_w, final_char_h, best_fs

    def _init_pygame(self):
        try:
            initial_win_w, initial_win_h_controls = self.original_window_size
            video_area_w = initial_win_w
            video_area_h = initial_win_h_controls - self.CONTROLS_AREA_HEIGHT
            self.font, self.char_width, self.char_height, self.font_size = self._calculate_font_for_area(
                video_area_w, video_area_h
            )
            self.screen = pygame.display.set_mode(self.original_window_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("ASCII Video Player")
            print(f"Initial Window: {self.original_window_size[0]}x{self.original_window_size[1]} | Font: {self.font_size}pt | Char Cell: {self.char_width}x{self.char_height} | Grid: {self.target_w_chars}x{self.target_h_chars}")
            self._load_assets()
            self.button_font = pygame.font.Font(self.pygame_font_path, 20)
            _initial_play_surface = self.button_font.render(self.play_button_text, True, (0,0,0))
            self.button_rect = _initial_play_surface.get_rect()
        except Exception as e:
            print(f"Error setting up Pygame window: {e}. Falling back to terminal.")
            self.args.terminal = True

    def _load_assets(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, "volume_icon.png")
            if os.path.exists(icon_path):
                volume_icon_loaded = pygame.image.load(icon_path)
                self.volume_icon = pygame.transform.scale(volume_icon_loaded, (20, 20))
            else:
                print(f"Info: volume_icon.png not found at {icon_path}. Volume slider will appear without icon.")
        except pygame.error as e:
            self.volume_icon = None
            print(f"Warning: Failed to load volume_icon.png: {e}. Volume slider will appear without icon.")

    def _extract_audio(self):
        ffmpeg_available = False
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        if ffmpeg_available:
            try:
                temp_dir = tempfile.gettempdir()
                self.temp_audio_path = os.path.join(temp_dir, f"ascii_video_audio_{os.getpid()}.wav")
                subprocess.run([
                    'ffmpeg', '-i', self.args.input, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', str(pygame.mixer.get_init()[0]),
                    '-ac', str(pygame.mixer.get_init()[1]),
                    '-y', self.temp_audio_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pygame.mixer.music.load(self.temp_audio_path)
                pygame.mixer.music.play()
                self.audio_loaded = True
                import atexit
                atexit.register(self._cleanup_audio)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Audio extraction failed (ffmpeg error). Video will play without sound.\nError: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
                self._cleanup_audio()
            except Exception as e:
                print(f"Warning: Could not process audio: {str(e)}. Video will play without sound.")
                self._cleanup_audio()
        else:
            print("Warning: ffmpeg not found. Please install ffmpeg for audio support. Video will play without sound.")

    def _cleanup_audio(self):
        try:
            if self.temp_audio_path and os.path.exists(self.temp_audio_path):
                os.unlink(self.temp_audio_path)
        except Exception as e:
            print(f"Warning: Could not delete temp audio file {self.temp_audio_path}: {e}")

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self._handle_resize(event)
            self._handle_keydown(event)
            self._handle_mouse(event)
        return True

    def _handle_resize(self, event):
        if event.type == pygame.VIDEORESIZE and not self.fullscreen:
            new_w, new_h = event.w, event.h
            self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.original_window_size = (new_w, new_h)
            self.font, self.char_width, self.char_height, self.font_size = self._calculate_font_for_area(
                new_w, new_h - self.CONTROLS_AREA_HEIGHT
            )
            print(f"Resized Window: {new_w}x{new_h} | Font: {self.font_size}pt | Grid: {self.target_w_chars}x{self.target_h_chars}")

    def _handle_keydown(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.playback_paused = not self.playback_paused
                if self.audio_loaded:
                    if self.playback_paused: pygame.mixer.music.pause()
                    else: pygame.mixer.music.unpause()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.volume = min(1.0, self.volume + 0.1)
                if self.audio_loaded: pygame.mixer.music.set_volume(self.volume)
            elif event.key == pygame.K_MINUS:
                self.volume = max(0.0, self.volume - 0.1)
                if self.audio_loaded: pygame.mixer.music.set_volume(self.volume)
            elif event.key == pygame.K_LEFT:
                new_time = max(0, self.current_time - 5)
                self._seek_to_time(new_time)
            elif event.key == pygame.K_RIGHT:
                new_time = min(self.total_duration if self.total_duration > 0 else self.current_time + 5, self.current_time + 5)
                self._seek_to_time(new_time)
            elif event.key == pygame.K_f:
                self._toggle_fullscreen()
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self._set_window_size(event.key)
            elif event.key == pygame.K_c:
                self.args.color = not self.args.color
                print(f"Color mode: {'on' if self.args.color else 'off'}")
            elif event.key == pygame.K_t:
                self.args.true_color = not self.args.true_color
                print(f"True color mode: {'on' if self.args.true_color else 'off'}")

    def _handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.button_rect and self.button_rect.collidepoint(event.pos):
                self.playback_paused = not self.playback_paused
                if self.audio_loaded:
                    if self.playback_paused: pygame.mixer.music.pause()
                    else: pygame.mixer.music.unpause()
            elif self.slider_rect and self.slider_rect.collidepoint(event.pos):
                self.dragging_volume = True
                self.volume = max(0.0, min(1.0, (event.pos[0] - self.slider_rect.left) / self.slider_rect.width))
                if self.audio_loaded: pygame.mixer.music.set_volume(self.volume)
            elif self.extended_timeline_rect and self.extended_timeline_rect.collidepoint(event.pos):
                self.dragging_timeline = True
                if self.total_duration > 0:
                    progress = max(0, min(1, (event.pos[0] - self.timeline_rect.left) / self.timeline_rect.width))
                    new_time = progress * self.total_duration
                    self._seek_to_time(new_time)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_volume = False
            self.dragging_timeline = False
        elif event.type == pygame.MOUSEMOTION:
            if self.button_rect:
                self.button_hover = self.button_rect.collidepoint(event.pos)
            if self.dragging_volume and self.slider_rect:
                self.volume = max(0.0, min(1.0, (event.pos[0] - self.slider_rect.left) / self.slider_rect.width))
                if self.audio_loaded: pygame.mixer.music.set_volume(self.volume)
            if self.dragging_timeline and self.timeline_rect and self.total_duration > 0:
                progress = max(0, min(1, (event.pos[0] - self.timeline_rect.left) / self.timeline_rect.width))
                new_time = progress * self.total_duration
                self._seek_to_time(new_time)

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.original_window_size = self.screen.get_size()
            display_info = pygame.display.Info()
            fs_w, fs_h = display_info.current_w, display_info.current_h
            self.screen = pygame.display.set_mode((fs_w, fs_h), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font, self.char_width, self.char_height, self.font_size = self._calculate_font_for_area(
                fs_w, fs_h - self.CONTROLS_AREA_HEIGHT
            )
            print(f"Fullscreen: {fs_w}x{fs_h} | Font: {self.font_size}pt")
        else:
            self.screen = pygame.display.set_mode(self.original_window_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font, self.char_width, self.char_height, self.font_size = self._calculate_font_for_area(
                self.original_window_size[0], self.original_window_size[1] - self.CONTROLS_AREA_HEIGHT
            )
            print(f"Windowed: {self.original_window_size[0]}x{self.original_window_size[1]} | Font: {self.font_size}pt")

    def _set_window_size(self, key):
        if self.fullscreen:
            self.fullscreen = False
            self.screen = pygame.display.set_mode(self.original_window_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
        if key == pygame.K_1: self.current_size_key = 'small'
        elif key == pygame.K_2: self.current_size_key = 'medium'
        elif key == pygame.K_3: self.current_size_key = 'large'
        new_size = self.window_sizes[self.current_size_key]
        self.screen = pygame.display.set_mode(new_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.original_window_size = new_size
        self.font, self.char_width, self.char_height, self.font_size = self._calculate_font_for_area(
            new_size[0], new_size[1] - self.CONTROLS_AREA_HEIGHT
        )
        print(f"Set size '{self.current_size_key}': {new_size[0]}x{new_size[1]} | Font: {self.font_size}pt")

    def _seek_to_time(self, target_time):
        if self.fps <= 0: return
        self.current_time = target_time
        self.frame_count = int(self.current_time * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        if self.audio_loaded:
            try:
                pygame.mixer.music.play(start=self.current_time)
                if self.playback_paused:
                    pygame.mixer.music.pause()
            except pygame.error as e:
                print(f"Error seeking audio: {e}. Audio might be out of sync.")
                pygame.mixer.music.stop()
                pygame.mixer.music.play()
                pygame.mixer.music.set_pos(self.current_time)
                if self.playback_paused:
                    pygame.mixer.music.pause()

    def _update_ui_layout(self):
        active_screen_w, active_screen_h = self.screen.get_size()
        controls_base_y = active_screen_h - self.CONTROLS_AREA_HEIGHT
        slot1_h = self.CONTROLS_AREA_HEIGHT * 0.5
        slot2_h = self.CONTROLS_AREA_HEIGHT * 0.5
        slot1_mid_y = controls_base_y + slot1_h / 2
        slot2_mid_y = (controls_base_y + slot1_h) + slot2_h / 2
        self.play_button_text = "Play" if self.playback_paused else "Pause"
        if self.button_rect:
            self.button_rect.right = active_screen_w - 10
            self.button_rect.centery = slot2_mid_y
        current_slider_x_pos = 10
        if self.volume_icon:
            current_slider_x_pos = 10 + self.volume_icon.get_width() + 5
        self.slider_rect = pygame.Rect(current_slider_x_pos, 0, self.slider_width, self.slider_height)
        self.slider_rect.centery = slot2_mid_y
        _time_curr_surf_calc = self.button_font.render(self.format_time(self.current_time), True, (0,0,0))
        _time_total_surf_calc = self.button_font.render(self.format_time(self.total_duration), True, (0,0,0))
        _timeline_x_calc = 10 + _time_curr_surf_calc.get_width() + 5
        _timeline_width_calc = active_screen_w - _timeline_x_calc - (_time_total_surf_calc.get_width() + 5 + 10)
        _timeline_width_calc = max(2 * self.timeline_knob_radius + 2, _timeline_width_calc)
        self.timeline_rect = pygame.Rect(_timeline_x_calc, 0, _timeline_width_calc, self.timeline_height)
        self.timeline_rect.centery = slot1_mid_y
        self.extended_timeline_rect = pygame.Rect(
            self.timeline_rect.left - self.timeline_knob_radius,
            self.timeline_rect.top,
            self.timeline_rect.width + 2 * self.timeline_knob_radius,
            self.timeline_rect.height
        )

    def _render_frame(self, frame_to_render):
        pil_img = Image.fromarray(cv2.cvtColor(frame_to_render, cv2.COLOR_BGR2RGB))
        img_resized_for_ascii = pil_img.resize((self.target_w_chars, self.target_h_chars))
        color_data_grid = None
        if self.args.color or self.args.true_color:
            source_for_color = cv2.cvtColor(frame_to_render, cv2.COLOR_BGR2RGB) if self.args.true_color else img_resized_for_ascii
            color_data_grid = sample_colors(source_for_color, self.target_w_chars, self.target_h_chars)
        _, out_txt_list = get_chars(img_resized_for_ascii, self.char_list, None, fmt="txt", color=False)
        if self.args.terminal:
            self._render_terminal(out_txt_list, color_data_grid)
        else:
            self._render_pygame(out_txt_list, color_data_grid)

    def _render_terminal(self, out_txt_list, color_data_grid):
        self.clear_screen()
        for y, line_str in enumerate(out_txt_list):
            output_line = []
            for x, char_val in enumerate(line_str):
                if color_data_grid is not None:
                    r, g, b = color_data_grid[y, x]
                    output_line.append(self.rgb_to_ansi(r, g, b) + char_val)
                else:
                    output_line.append(char_val)
            sys.stdout.write("".join(output_line) + self.reset_ansi_color() + "\n")
        sys.stdout.flush()

    def _render_pygame(self, out_txt_list, color_data_grid):
        if self.screen and self.font:
            self.screen.fill((0, 0, 0))
            for y, line_str in enumerate(out_txt_list):
                if color_data_grid is not None:
                    # Render character by character for color
                    for x, char_val in enumerate(line_str):
                        char_color = tuple(color_data_grid[y, x])
                        text_surface = self.font.render(char_val, True, char_color)
                        self.screen.blit(text_surface, (x * self.char_width, y * self.char_height))
                else:
                    # Render line by line for monochrome
                    text_surface = self.font.render("".join(line_str), True, (255, 255, 255))
                    self.screen.blit(text_surface, (0, y * self.char_height))
            self._render_controls()
            pygame.display.flip()

    def _render_controls(self):
        self._update_ui_layout()
        time_curr_surf = self.button_font.render(self.format_time(self.current_time), True, (255,255,255))
        time_total_surf = self.button_font.render(self.format_time(self.total_duration), True, (255,255,255))
        time_curr_label_x_pos = self.timeline_rect.left - time_curr_surf.get_width() - 5
        time_curr_label_y_pos = self.timeline_rect.centery - time_curr_surf.get_height() // 2
        self.screen.blit(time_curr_surf, (time_curr_label_x_pos, time_curr_label_y_pos))
        time_total_label_x_pos = self.timeline_rect.right + 5
        time_total_label_y_pos = self.timeline_rect.centery - time_total_surf.get_height() // 2
        self.screen.blit(time_total_surf, (time_total_label_x_pos, time_total_label_y_pos))
        pygame.draw.rect(self.screen, (80, 80, 80), self.timeline_rect)
        if self.total_duration > 0:
            progress_ratio = self.current_time / self.total_duration
            progress_pixel_width = int(progress_ratio * self.timeline_rect.width)
            progress_pixel_width = min(progress_pixel_width, self.timeline_rect.width)
            progress_rect = pygame.Rect(self.timeline_rect.left, self.timeline_rect.top, progress_pixel_width, self.timeline_rect.height)
            pygame.draw.rect(self.screen, (150, 150, 150), progress_rect)
            knob_timeline_center_x = self.timeline_rect.left + progress_pixel_width
            knob_timeline_center_x = max(self.timeline_rect.left, min(knob_timeline_center_x, self.timeline_rect.right))
            pygame.draw.circle(self.screen, (255,255,255), (knob_timeline_center_x, self.timeline_rect.centery), self.timeline_knob_radius)
        if self.volume_icon:
            icon_render_x = 10
            icon_render_y = self.slider_rect.centery - self.volume_icon.get_height() // 2
            self.screen.blit(self.volume_icon, (icon_render_x, icon_render_y))
        pygame.draw.rect(self.screen, (100, 100, 100), self.slider_rect)
        knob_vol_x = self.slider_rect.left + int(self.volume * self.slider_rect.width)
        knob_vol_x = max(self.slider_rect.left, min(knob_vol_x, self.slider_rect.right))
        pygame.draw.circle(self.screen, (255, 255, 255), (knob_vol_x, self.slider_rect.centery), self.slider_knob_radius)
        vol_text_surf = self.button_font.render(f"{int(self.volume*100)}%", True, (255,255,255))
        vol_text_render_x = self.slider_rect.right + 8
        vol_text_render_y = self.slider_rect.centery - vol_text_surf.get_height() // 2
        self.screen.blit(vol_text_surf, (vol_text_render_x, vol_text_render_y))
        play_button_surface_render = self.button_font.render(self.play_button_text, True, (200,200,200) if self.button_hover else (255,255,255))
        self.screen.blit(play_button_surface_render, self.button_rect.topleft)

    def run(self):
        if not self.args.terminal:
            self._init_pygame()
        self._extract_audio()
        running = True
        try:
            while running:
                loop_start_time = time.time()
                if not self.args.terminal and self.screen:
                    running = self._handle_events()
                if not running: break
                frame_to_render = None
                if not self.playback_paused:
                    ret, cv_frame = self.cap.read()
                    if ret:
                        self.last_frame = cv_frame
                        frame_to_render = cv_frame
                        if self.fps > 0: self.current_time = self.frame_count / self.fps
                    else:
                        if self.frame_count > 0: print("End of video.")
                        else: print("Error reading first frame or video empty.")
                        running = False; continue
                else:
                    if self.last_frame is not None: frame_to_render = self.last_frame
                    else: running = False; continue
                if frame_to_render is not None:
                    self._render_frame(frame_to_render)
                if not self.playback_paused:
                    self.frame_count += 1
                elapsed_since_loop_start = time.time() - loop_start_time
                sleep_duration = self.delay - elapsed_since_loop_start
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        except KeyboardInterrupt:
            print("\nPlayback interrupted by user.")
        finally:
            self.cap.release()
            if self.audio_loaded:
                pygame.mixer.music.stop()
            pygame.quit()

    @staticmethod
    def clear_screen():
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    @staticmethod
    def format_time(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def rgb_to_ansi(r, g, b):
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def reset_ansi_color():
        return "\033[0m"
