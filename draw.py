"""
AutoDraw Studio - High Performance Screen & Game Canvas Drawing Tool
Designed for MS Paint, browser drawing canvases, and video games (Roblox Free Draw, Starving Artists).
Supports vector contour tracing, serpentine raster scanlines, Canny edge detection,
Win32 DirectInput-compatible game execution, instant F8/ESC abort recovery, and a modern minimalist UI.
"""

import os
import sys
import time
import queue
import threading
import argparse
import ctypes
from ctypes import wintypes
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyautogui


# ==============================================================================
# 1. HARDWARE WIN32 INPUT ENGINE (GAME & ROBLOX COMPATIBILITY)
# ==============================================================================

class GameInputEngine:
    """
    Direct Win32 API input dispatcher.
    Provides hardware-compatible mouse events, absolute cursor movements,
    frame-synced hold delays for video games (Roblox), and exception-free cleanup.
    """
    user32 = ctypes.windll.user32

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_VIRTUALDESK = 0x4000

    VK_F8 = 0x77
    VK_ESCAPE = 0x1B

    @classmethod
    def get_screen_size(cls):
        w = cls.user32.GetSystemMetrics(0)
        h = cls.user32.GetSystemMetrics(1)
        return max(1, w), max(1, h)

    @classmethod
    def is_abort_key_pressed(cls):
        """Checks if F8 or ESC is pressed globally anywhere on Windows."""
        return bool(
            (cls.user32.GetAsyncKeyState(cls.VK_F8) & 0x8000) or
            (cls.user32.GetAsyncKeyState(cls.VK_ESCAPE) & 0x8000)
        )

    @classmethod
    def safe_release_mouse(cls):
        """Releases the left mouse button directly via Win32 without throwing failsafe exceptions."""
        try:
            cls.user32.mouse_event(cls.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except Exception:
            pass

    @classmethod
    def move_cursor(cls, x, y):
        """Moves cursor and dispatches absolute mouse motion events for DirectX/RawInput games."""
        sw, sh = cls.get_screen_size()
        norm_x = int(x * 65535 / (sw - 1))
        norm_y = int(y * 65535 / (sh - 1))
        cls.user32.SetCursorPos(int(x), int(y))
        cls.user32.mouse_event(
            cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE,
            norm_x, norm_y, 0, 0
        )

    @classmethod
    def mouse_down(cls):
        cls.user32.mouse_event(cls.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    @classmethod
    def mouse_up(cls):
        cls.user32.mouse_event(cls.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    @classmethod
    def click_pixel(cls, x, y, hold_time=0.018, delay_after=0.006):
        """
        Executes a frame-synced discrete mouse click.
        Crucial for Roblox games like 'Starving Artists' where canvas pixels are UI buttons.
        """
        cls.move_cursor(x, y)
        cls.mouse_down()
        time.sleep(hold_time)
        cls.mouse_up()
        if delay_after > 0:
            time.sleep(delay_after)


# ==============================================================================
# 2. IMAGE PROCESSING & STROKE GENERATION ENGINE
# ==============================================================================

class ImageProcessor:
    """Handles image loading, alpha blending, contour extraction, and path optimization."""

    @staticmethod
    def load_image(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Blend transparent PNGs onto white background
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            bgr = img[:, :, :3]
            white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
            blended = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
            gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
            return blended, gray
        elif len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img, gray
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return bgr, img

    @staticmethod
    def compute_scaled_dims(orig_w, orig_h, target_w, target_h):
        if orig_w == 0 or orig_h == 0 or target_w == 0 or target_h == 0:
            return 1, 1, 0, 0
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        return new_w, new_h, offset_x, offset_y

    @staticmethod
    def get_binary_mask(gray_img, threshold=128, invert=False):
        if invert:
            _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def optimize_path_order(strokes):
        """Greedy nearest-neighbor TSP to minimize pen-up air travel."""
        if not strokes:
            return []

        remaining = [np.array(s, dtype=np.int32) for s in strokes if len(s) > 0]
        if not remaining:
            return []

        ordered = [remaining.pop(0)]
        curr_pos = ordered[0][-1]

        while remaining:
            starts = np.array([s[0] for s in remaining])
            ends = np.array([s[-1] for s in remaining])

            d_starts = np.hypot(starts[:, 0] - curr_pos[0], starts[:, 1] - curr_pos[1])
            d_ends = np.hypot(ends[:, 0] - curr_pos[0], ends[:, 1] - curr_pos[1])

            min_s_idx = int(np.argmin(d_starts))
            min_e_idx = int(np.argmin(d_ends))

            if d_starts[min_s_idx] <= d_ends[min_e_idx]:
                chosen = remaining.pop(min_s_idx)
            else:
                chosen = remaining.pop(min_e_idx)[::-1]

            ordered.append(chosen)
            curr_pos = chosen[-1]

        return ordered

    @classmethod
    def generate_contour_strokes(cls, mask, epsilon=1.2, min_length=4):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        raw_strokes = []

        for cnt in contours:
            length = cv2.arcLength(cnt, True)
            if length < min_length:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = approx.reshape(-1, 2)
            if len(pts) >= 2:
                pts = np.vstack([pts, pts[0]])
                raw_strokes.append(pts)

        return cls.optimize_path_order(raw_strokes)

    @classmethod
    def generate_canny_strokes(cls, gray_img, low_thresh=50, high_thresh=150, epsilon=1.0):
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return cls.generate_contour_strokes(edges, epsilon=epsilon, min_length=4)

    @staticmethod
    def generate_serpentine_raster_strokes(mask, step=3):
        h, w = mask.shape
        strokes = []
        ltr = True

        for y in range(0, h, step):
            row = mask[y]
            runs = []
            in_run = False
            start_x = 0

            for x in range(w):
                is_draw = row[x] > 0
                if is_draw and not in_run:
                    in_run = True
                    start_x = x
                elif not is_draw and in_run:
                    in_run = False
                    if x - 1 - start_x >= 1:
                        runs.append((start_x, x - 1))

            if in_run and (w - 1 - start_x >= 1):
                runs.append((start_x, w - 1))

            if not runs:
                continue

            if not ltr:
                runs.reverse()
                for x1, x2 in runs:
                    strokes.append(np.array([(x2, y), (x1, y)], dtype=np.int32))
            else:
                for x1, x2 in runs:
                    strokes.append(np.array([(x1, y), (x2, y)], dtype=np.int32))

            ltr = not ltr

        return strokes

    @classmethod
    def generate_hybrid_strokes(cls, mask, epsilon=1.2, step=4):
        contour_strokes = cls.generate_contour_strokes(mask, epsilon=epsilon, min_length=5)
        kernel_size = max(2, step // 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        shading_mask = cv2.erode(mask, kernel)
        fill_strokes = cls.generate_serpentine_raster_strokes(shading_mask, step=step)
        return contour_strokes + fill_strokes

    @staticmethod
    def interpolate_points_for_game(pts, max_step=6):
        """Interpolates sparse line segments so video game canvases capture continuous strokes."""
        if len(pts) < 2:
            return pts
        interp = [pts[0]]
        for p2 in pts[1:]:
            p1 = interp[-1]
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist > max_step:
                steps = int(np.ceil(dist / max_step))
                for i in range(1, steps):
                    t = i / steps
                    ix = int(round(p1[0] + (p2[0] - p1[0]) * t))
                    iy = int(round(p1[1] + (p2[1] - p1[1]) * t))
                    interp.append([ix, iy])
            interp.append([int(p2[0]), int(p2[1])])
        return np.array(interp, dtype=np.int32)

    @staticmethod
    def render_preview(strokes, canvas_w, canvas_h, offset_x=0, offset_y=0, bg_color=(255, 255, 255)):
        preview = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)
        preview[:] = bg_color

        # Subtle canvas boundary line
        cv2.rectangle(preview, (0, 0), (canvas_w - 1, canvas_h - 1), (225, 225, 230), 1)

        for stroke in strokes:
            if len(stroke) < 2:
                continue
            shifted = stroke + np.array([offset_x, offset_y], dtype=np.int32)
            cv2.polylines(preview, [shifted], False, (30, 30, 35), 1, cv2.LINE_AA)

        return preview


# ==============================================================================
# 3. INTERACTIVE CANVAS SNIPPER
# ==============================================================================

class CanvasSnipper:
    """Semi-transparent fullscreen overlay for selecting canvas area."""

    def __init__(self, parent, on_selection_complete):
        self.parent = parent
        self.callback = on_selection_complete

        self.top = tk.Toplevel(parent)
        self.top.attributes("-alpha", 0.3)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.top.config(cursor="cross")

        self.canvas = tk.Canvas(self.top, cursor="cross", bg="#0a0a0c", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        screen_w = self.top.winfo_screenwidth()
        self.canvas.create_text(
            screen_w // 2, 45,
            text="DRAG A RECTANGLE OVER YOUR DRAWING CANVAS  |  PRESS ESC TO CANCEL",
            fill="#818cf8", font=("Segoe UI", 13, "bold")
        )

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.text_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.top.bind("<Escape>", lambda e: self.top.destroy())

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.text_id:
            self.canvas.delete(self.text_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#6366f1", width=2, fill="#1e1e2d"
        )
        self.text_id = self.canvas.create_text(
            self.start_x + 50, max(20, self.start_y - 15),
            text="0 x 0", fill="#a5b4fc", font=("Segoe UI", 10, "bold")
        )

    def on_drag(self, event):
        if self.start_x is None:
            return
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
        w = abs(cur_x - self.start_x)
        h = abs(cur_y - self.start_y)
        top_left_x = min(self.start_x, cur_x)
        top_left_y = min(self.start_y, cur_y)
        self.canvas.coords(self.text_id, top_left_x + 50, max(25, top_left_y - 15))
        self.canvas.itemconfig(self.text_id, text=f"{w} x {h} px")

    def on_release(self, event):
        if self.start_x is None:
            return
        end_x, end_y = event.x, event.y
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        w = abs(end_x - self.start_x)
        h = abs(end_y - self.start_y)

        self.top.destroy()
        if w >= 15 and h >= 15:
            self.callback(x, y, w, h)


# ==============================================================================
# 4. ROBUST DRAWING WORKER (THREAD-SAFE VIA QUEUE & GAME-COMPATIBLE)
# ==============================================================================

class DrawingWorker(threading.Thread):
    """
    Executes drawing automation with dedicated game profiles,
    bulletproof exception handling, and global F8/ESC abort detection.
    Communicates via a thread-safe Queue to avoid any Tkinter threading crashes.
    """

    def __init__(self, strokes, start_x, start_y, offset_x, offset_y,
                 profile="Standard (Paint, Photoshop, Web)",
                 speed_mode="Fast (2ms)",
                 countdown_secs=3,
                 event_queue=None):
        super().__init__(daemon=True)
        self.strokes = strokes
        self.base_x = start_x + offset_x
        self.base_y = start_y + offset_y
        self.profile = profile
        self.speed_mode = speed_mode
        self.countdown_secs = countdown_secs
        self.event_queue = event_queue
        self.is_aborted = False

    def abort(self):
        self.is_aborted = True

    def _should_stop(self):
        if self.is_aborted:
            return True
        if GameInputEngine.is_abort_key_pressed():
            self.is_aborted = True
            return True
        return False

    def run(self):
        # Countdown with frequent abort polling
        for c in range(self.countdown_secs, 0, -1):
            if self._should_stop():
                self._dispatch_finish(False, "Drawing cancelled before start.")
                return
            self._dispatch_countdown(c)
            for _ in range(10):
                if self._should_stop():
                    self._dispatch_finish(False, "Drawing cancelled before start.")
                    return
                time.sleep(0.1)

        self._dispatch_countdown(0)

        total_strokes = len(self.strokes)
        status_msg = "Drawing complete!"
        success = True

        is_roblox_drag = "Roblox (Smooth Drag" in self.profile
        is_roblox_tap = "Roblox (Pixel Tap" in self.profile

        if is_roblox_drag:
            stroke_delay = 0.012
            point_delay = 0.008
            down_hold = 0.020
            up_hold = 0.012
        elif is_roblox_tap:
            stroke_delay = 0.006
            down_hold = 0.018
            up_hold = 0.006
        else:
            if "Turbo" in self.speed_mode:
                point_delay = 0.001
                stroke_delay = 0.001
            elif "Fast" in self.speed_mode:
                point_delay = 0.002
                stroke_delay = 0.003
            elif "Normal" in self.speed_mode:
                point_delay = 0.004
                stroke_delay = 0.006
            else:
                point_delay = 0.008
                stroke_delay = 0.012
            down_hold = 0.005
            up_hold = 0.005

        try:
            for idx, stroke in enumerate(self.strokes):
                if self._should_stop():
                    status_msg = "Drawing stopped by user (F8/ESC)."
                    success = False
                    break

                if len(stroke) == 0:
                    continue

                if is_roblox_tap:
                    # Discrete point-by-point clicking for Roblox button grid games (Starving Artists)
                    for pt in stroke:
                        if self._should_stop():
                            break
                        px = int(self.base_x + pt[0])
                        py = int(self.base_y + pt[1])
                        GameInputEngine.click_pixel(px, py, hold_time=down_hold, delay_after=up_hold)
                elif is_roblox_drag:
                    # Interpolated smooth dragging with hardware events for Roblox canvas
                    interpolated = ImageProcessor.interpolate_points_for_game(stroke, max_step=6)
                    x0 = int(self.base_x + interpolated[0][0])
                    y0 = int(self.base_y + interpolated[0][1])

                    GameInputEngine.move_cursor(x0, y0)
                    GameInputEngine.mouse_down()
                    time.sleep(down_hold)

                    for pt in interpolated[1:]:
                        if self._should_stop():
                            break
                        px = int(self.base_x + pt[0])
                        py = int(self.base_y + pt[1])
                        GameInputEngine.move_cursor(px, py)
                        if point_delay > 0:
                            time.sleep(point_delay)

                    time.sleep(up_hold)
                    GameInputEngine.mouse_up()
                    if stroke_delay > 0:
                        time.sleep(stroke_delay)
                else:
                    # Standard fast desktop canvas dragging
                    x0 = int(self.base_x + stroke[0][0])
                    y0 = int(self.base_y + stroke[0][1])

                    GameInputEngine.move_cursor(x0, y0)
                    if len(stroke) == 1:
                        GameInputEngine.mouse_down()
                        time.sleep(down_hold)
                        GameInputEngine.mouse_up()
                        continue

                    GameInputEngine.mouse_down()
                    time.sleep(down_hold)

                    for pt in stroke[1:]:
                        if self._should_stop():
                            break
                        px = int(self.base_x + pt[0])
                        py = int(self.base_y + pt[1])
                        GameInputEngine.move_cursor(px, py)
                        if point_delay > 0:
                            time.sleep(point_delay)

                    GameInputEngine.mouse_up()
                    if stroke_delay > 0:
                        time.sleep(stroke_delay)

                if idx % 4 == 0 or idx == total_strokes - 1:
                    self._dispatch_progress(idx + 1, total_strokes)

        except pyautogui.FailSafeException:
            status_msg = "Drawing stopped by mouse corner failsafe."
            success = False
        except Exception as ex:
            status_msg = f"Drawing stopped: {ex}"
            success = False
        finally:
            # Bulletproof cleanup: guarantee mouse release without re-throwing failsafe
            GameInputEngine.safe_release_mouse()

        if self.is_aborted and "stopped" not in status_msg:
            status_msg = "Drawing stopped by user."
            success = False

        self._dispatch_finish(success, status_msg)

    def _dispatch_countdown(self, count):
        if self.event_queue:
            self.event_queue.put(("countdown", count))

    def _dispatch_progress(self, current, total):
        if self.event_queue:
            self.event_queue.put(("progress", (current, total)))

    def _dispatch_finish(self, success, msg):
        if self.event_queue:
            self.event_queue.put(("finish", (success, msg)))


# ==============================================================================
# 5. MODERN MINIMALIST STUDIO GUI
# ==============================================================================

class ModernAutoDrawStudio(tk.Tk):
    """Sleek, minimalist dark-mode desktop interface for AutoDraw."""

    def __init__(self, initial_image=None):
        super().__init__()
        self.title("AutoDraw Studio")
        self.geometry("1100x740")
        self.minsize(960, 640)

        # Palette
        self.c_bg = "#0d0d11"          # Deep obsidian background
        self.c_surface = "#16161c"     # Dark zinc surface
        self.c_card = "#1c1c24"        # Card background
        self.c_border = "#262632"      # Border line
        self.c_accent = "#6366f1"      # Electric indigo
        self.c_accent_hover = "#4f46e5"
        self.c_text_primary = "#f4f4f6"
        self.c_text_secondary = "#888899"
        self.c_text_muted = "#5d5d70"
        self.c_emerald = "#10b981"
        self.c_rose = "#ef4444"

        self.configure(bg=self.c_bg)

        # Thread-safe event queue
        self.event_queue = queue.Queue()

        # State Variables
        self.image_path = initial_image or ""
        self.bgr_img = None
        self.gray_img = None
        self.canvas_x = 100
        self.canvas_y = 100
        self.canvas_w = 600
        self.canvas_h = 450
        self.generated_strokes = []
        self.offset_x = 0
        self.offset_y = 0
        self.scaled_w = 0
        self.scaled_h = 0
        self.worker = None

        # Configuration Variables
        self.profile_var = tk.StringVar(value="Standard (Paint, Photoshop, Web)")
        self.mode_var = tk.StringVar(value="Contour / Vector Sketch")
        self.threshold_var = tk.IntVar(value=128)
        self.invert_var = tk.BooleanVar(value=False)
        self.epsilon_var = tk.DoubleVar(value=1.2)
        self.step_var = tk.IntVar(value=3)
        self.speed_var = tk.StringVar(value="Fast (2ms)")
        self.countdown_var = tk.IntVar(value=3)

        self._debounce_timer = None
        self._preview_photo = None

        self._setup_ttk_styles()
        self._build_ui()

        # Keyboard listener: pressing Escape or F8 in the window halts drawing
        self.bind("<F8>", lambda e: self.abort_drawing())
        self.bind("<Escape>", lambda e: self.abort_drawing())

        # Start queue poller
        self.after(25, self._poll_event_queue)

        if self.image_path:
            self.load_image_file(self.image_path)

    def _setup_ttk_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=self.c_bg, foreground=self.c_text_primary, font=("Segoe UI", 9))
        style.configure("TCombobox", fieldbackground="#22222c", background="#2a2a38", foreground="white", arrowcolor="white")
        style.map("TCombobox", fieldbackground=[("readonly", "#22222c")], foreground=[("readonly", "white")])
        style.configure("TCheckbutton", background=self.c_card, foreground=self.c_text_primary)
        style.map("TCheckbutton", background=[("active", self.c_card)])
        style.configure("TProgressbar", troughcolor="#1b1b24", background="#6366f1", thickness=4)

    def _build_ui(self):
        # 1. Sleek Minimalist Navigation Bar
        nav_bar = tk.Frame(self, bg=self.c_surface, height=48, highlightthickness=1, highlightbackground=self.c_border)
        nav_bar.pack(fill=tk.X, side=tk.TOP)
        nav_bar.pack_propagate(False)

        brand_frame = tk.Frame(nav_bar, bg=self.c_surface)
        brand_frame.pack(side=tk.LEFT, padx=18)

        tk.Label(brand_frame, text="AutoDraw", font=("Segoe UI", 12, "bold"), fg="#ffffff", bg=self.c_surface).pack(side=tk.LEFT)
        tk.Label(brand_frame, text="PRO", font=("Segoe UI", 7, "bold"), fg="#818cf8", bg="#222232", padx=5, pady=1).pack(side=tk.LEFT, padx=8)

        # Status Pill
        self.pill_status = tk.Label(
            nav_bar, text="Ready", font=("Segoe UI", 8, "bold"),
            fg="#10b981", bg="#15261f", padx=10, pady=2
        )
        self.pill_status.pack(side=tk.LEFT, padx=10)

        # Emergency Stop Pill Hint
        pill_stop = tk.Label(
            nav_bar, text="Emergency Stop: Press F8 or ESC",
            font=("Segoe UI", 8), fg="#f59e0b", bg="#261e12", padx=10, pady=2
        )
        pill_stop.pack(side=tk.RIGHT, padx=18)

        # 2. Workspace Body
        body = tk.Frame(self, bg=self.c_bg)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        # Left Column: Controls (Fixed 380px)
        left_panel = tk.Frame(body, bg=self.c_surface, width=390, highlightthickness=1, highlightbackground=self.c_border)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        left_panel.pack_propagate(False)

        # Right Column: Viewport & Stats
        right_panel = tk.Frame(body, bg=self.c_bg)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_sidebar(left_panel)
        self._build_viewport(right_panel)

        # 3. Slim Bottom Status Bar
        self.footer = tk.Frame(self, bg=self.c_surface, height=28, highlightthickness=1, highlightbackground=self.c_border)
        self.footer.pack(fill=tk.X, side=tk.BOTTOM)
        self.footer.pack_propagate(False)

        self.lbl_footer = tk.Label(
            self.footer, text="Load an image and snip your canvas area to begin.",
            font=("Segoe UI", 8), fg=self.c_text_secondary, bg=self.c_surface, padx=16
        )
        self.lbl_footer.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(self.footer, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side=tk.RIGHT, padx=16)

    def _build_sidebar(self, parent):
        container = tk.Frame(parent, bg=self.c_surface, padx=16, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        # Section 1: Inputs & Canvas
        self._build_section_header(container, "TARGET & CANVAS")

        # Image picker row
        row_img = tk.Frame(container, bg=self.c_card, padx=8, pady=6, highlightthickness=1, highlightbackground=self.c_border)
        row_img.pack(fill=tk.X, pady=(4, 6))

        btn_browse = tk.Button(
            row_img, text="Open Image", bg="#2a2a38", fg=self.c_text_primary,
            activebackground="#363646", activeforeground="white",
            relief="flat", font=("Segoe UI", 8, "bold"), padx=10, pady=3,
            command=self.browse_image
        )
        btn_browse.pack(side=tk.LEFT)

        self.lbl_filename = tk.Label(
            row_img, text="No image selected", fg=self.c_text_secondary,
            bg=self.c_card, font=("Segoe UI", 8), anchor="w", padx=8
        )
        self.lbl_filename.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Canvas snip row
        row_canvas = tk.Frame(container, bg=self.c_card, padx=8, pady=6, highlightthickness=1, highlightbackground=self.c_border)
        row_canvas.pack(fill=tk.X, pady=(0, 12))

        btn_snip = tk.Button(
            row_canvas, text="Snip Canvas", bg="#1e2d27", fg="#34d399",
            activebackground="#253a32", activeforeground="#34d399",
            relief="flat", font=("Segoe UI", 8, "bold"), padx=10, pady=3,
            command=self.start_canvas_snipper
        )
        btn_snip.pack(side=tk.LEFT)

        self.lbl_canvas_info = tk.Label(
            row_canvas, text=f"{self.canvas_w} x {self.canvas_h} px", fg="#a7f3d0",
            bg=self.c_card, font=("Segoe UI", 8, "bold"), anchor="w", padx=8
        )
        self.lbl_canvas_info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Section 2: Drawing Profile & Mode
        self._build_section_header(container, "ENVIRONMENT & MODE")

        # Target Profile (Roblox vs Desktop)
        tk.Label(container, text="Target Environment:", fg=self.c_text_secondary, bg=self.c_surface, font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(2, 2))
        profiles = [
            "Standard (Paint, Photoshop, Web)",
            "Roblox (Smooth Drag - Free Draw)",
            "Roblox (Pixel Tap - Starving Artists)"
        ]
        self.cmb_profile = ttk.Combobox(container, textvariable=self.profile_var, values=profiles, state="readonly")
        self.cmb_profile.pack(fill=tk.X, pady=(0, 8))
        self.cmb_profile.bind("<<ComboboxSelected>>", lambda e: self.trigger_preview_update())

        # Drawing Strategy
        tk.Label(container, text="Drawing Strategy:", fg=self.c_text_secondary, bg=self.c_surface, font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(0, 2))
        modes = [
            "Contour / Vector Sketch",
            "Canny Edge Sketch",
            "Serpentine Raster Fill",
            "Hybrid (Outline + Fill)"
        ]
        self.cmb_mode = ttk.Combobox(container, textvariable=self.mode_var, values=modes, state="readonly")
        self.cmb_mode.pack(fill=tk.X, pady=(0, 12))
        self.cmb_mode.bind("<<ComboboxSelected>>", lambda e: self.trigger_preview_update())

        # Section 3: Fine Tuning
        self._build_section_header(container, "FINE TUNING")

        # Sliders Card
        card_sliders = tk.Frame(container, bg=self.c_card, padx=10, pady=8, highlightthickness=1, highlightbackground=self.c_border)
        card_sliders.pack(fill=tk.X, pady=(4, 12))

        # Threshold Slider
        row_th = tk.Frame(card_sliders, bg=self.c_card)
        row_th.pack(fill=tk.X)
        tk.Label(row_th, text="Threshold", fg=self.c_text_secondary, bg=self.c_card, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.badge_th = tk.Label(row_th, text="128", fg=self.c_accent, bg=self.c_card, font=("Segoe UI", 8, "bold"))
        self.badge_th.pack(side=tk.RIGHT)

        tk.Scale(
            card_sliders, from_=5, to=250, variable=self.threshold_var, orient=tk.HORIZONTAL,
            showvalue=0, bg=self.c_card, fg=self.c_accent, highlightthickness=0,
            activebackground=self.c_accent, troughcolor="#2a2a38",
            command=self._on_thresh_slide
        ).pack(fill=tk.X, pady=(0, 4))

        # Smoothness Slider
        row_eps = tk.Frame(card_sliders, bg=self.c_card)
        row_eps.pack(fill=tk.X)
        tk.Label(row_eps, text="Smoothness (Epsilon)", fg=self.c_text_secondary, bg=self.c_card, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.badge_eps = tk.Label(row_eps, text="1.2", fg=self.c_accent, bg=self.c_card, font=("Segoe UI", 8, "bold"))
        self.badge_eps.pack(side=tk.RIGHT)

        tk.Scale(
            card_sliders, from_=0.5, to=5.0, resolution=0.1, variable=self.epsilon_var, orient=tk.HORIZONTAL,
            showvalue=0, bg=self.c_card, fg=self.c_accent, highlightthickness=0,
            activebackground=self.c_accent, troughcolor="#2a2a38",
            command=self._on_eps_slide
        ).pack(fill=tk.X, pady=(0, 4))

        # Brush Step Slider
        row_step = tk.Frame(card_sliders, bg=self.c_card)
        row_step.pack(fill=tk.X)
        tk.Label(row_step, text="Brush Step (px)", fg=self.c_text_secondary, bg=self.c_card, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.badge_step = tk.Label(row_step, text="3", fg=self.c_accent, bg=self.c_card, font=("Segoe UI", 8, "bold"))
        self.badge_step.pack(side=tk.RIGHT)

        tk.Scale(
            card_sliders, from_=1, to=10, variable=self.step_var, orient=tk.HORIZONTAL,
            showvalue=0, bg=self.c_card, fg=self.c_accent, highlightthickness=0,
            activebackground=self.c_accent, troughcolor="#2a2a38",
            command=self._on_step_slide
        ).pack(fill=tk.X, pady=(0, 6))

        # Invert Checkbox
        ttk.Checkbutton(
            card_sliders, text="Invert Colors (Light on Dark)",
            variable=self.invert_var, command=self.trigger_preview_update
        ).pack(anchor=tk.W)

        # Section 4: Speed & Execution
        self._build_section_header(container, "EXECUTION")

        row_exec_cfg = tk.Frame(container, bg=self.c_surface)
        row_exec_cfg.pack(fill=tk.X, pady=(2, 8))

        tk.Label(row_exec_cfg, text="Speed:", fg=self.c_text_secondary, bg=self.c_surface, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.cmb_speed = ttk.Combobox(
            row_exec_cfg, textvariable=self.speed_var,
            values=["Turbo (0ms)", "Fast (2ms)", "Normal (5ms)", "Safe / Web (10ms)"],
            state="readonly", width=13
        )
        self.cmb_speed.pack(side=tk.LEFT, padx=(6, 12))

        tk.Label(row_exec_cfg, text="Delay (s):", fg=self.c_text_secondary, bg=self.c_surface, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Spinbox(row_exec_cfg, from_=1, to=10, textvariable=self.countdown_var, width=3).pack(side=tk.LEFT, padx=4)

        # Start Button
        self.btn_start = tk.Button(
            container, text="Start Auto-Draw", bg="#4f46e5", fg="white",
            activebackground="#4338ca", activeforeground="white",
            relief="flat", font=("Segoe UI", 10, "bold"), pady=7,
            command=self.start_drawing
        )
        self.btn_start.pack(fill=tk.X, pady=(2, 6))

        # Abort Button
        self.btn_abort = tk.Button(
            container, text="Stop Drawing (F8)", bg="#261a1d", fg="#f87171",
            activebackground="#3b1d22", activeforeground="#f87171",
            relief="flat", font=("Segoe UI", 9, "bold"), pady=4, state=tk.DISABLED,
            command=self.abort_drawing
        )
        self.btn_abort.pack(fill=tk.X)

    def _build_section_header(self, parent, title):
        tk.Label(
            parent, text=title, font=("Segoe UI", 8, "bold"),
            fg=self.c_text_muted, bg=self.c_surface
        ).pack(anchor=tk.W, pady=(4, 2))

    def _build_viewport(self, parent):
        # Floating Stats Pill Bar
        stats_bar = tk.Frame(parent, bg=self.c_surface, padx=14, pady=8, highlightthickness=1, highlightbackground=self.c_border)
        stats_bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 12))

        self.lbl_stats = tk.Label(
            stats_bar,
            text="0 strokes  |  0 points  |  Canvas: 600x450  |  Est. Time: --",
            fg="#c7d2fe", bg=self.c_surface, font=("Segoe UI", 9, "bold")
        )
        self.lbl_stats.pack(side=tk.LEFT)

        # Main Canvas Viewport Frame
        viewport_frame = tk.Frame(parent, bg="#0a0a0e", highlightthickness=1, highlightbackground=self.c_border)
        viewport_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(viewport_frame, bg="#0a0a0e", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------------------
    # Thread-Safe Queue Poller
    # --------------------------------------------------------------------------

    def _poll_event_queue(self):
        """Polls events posted by the background DrawingWorker."""
        try:
            while True:
                msg_type, data = self.event_queue.get_nowait()
                if msg_type == "countdown":
                    self.on_worker_countdown(data)
                elif msg_type == "progress":
                    self.on_worker_progress(data[0], data[1])
                elif msg_type == "finish":
                    self.on_worker_finish(data[0], data[1])
        except queue.Empty:
            pass

        # Reschedule next poll
        self.after(25, self._poll_event_queue)

    # --------------------------------------------------------------------------
    # Slider & Control Event Handlers
    # --------------------------------------------------------------------------

    def _on_thresh_slide(self, val):
        self.badge_th.config(text=str(val))
        self.trigger_preview_update()

    def _on_eps_slide(self, val):
        self.badge_eps.config(text=f"{float(val):.1f}")
        self.trigger_preview_update()

    def _on_step_slide(self, val):
        self.badge_step.config(text=str(val))
        self.trigger_preview_update()

    def trigger_preview_update(self):
        if self._debounce_timer:
            self.after_cancel(self._debounce_timer)
        self._debounce_timer = self.after(75, self.update_preview)

    # --------------------------------------------------------------------------
    # Image Loading & Canvas Snip Calibration
    # --------------------------------------------------------------------------

    def browse_image(self):
        chosen = filedialog.askopenfilename(
            title="Select Image to Draw",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if chosen:
            self.load_image_file(chosen)

    def load_image_file(self, path):
        try:
            self.image_path = path
            self.bgr_img, self.gray_img = ImageProcessor.load_image(path)
            short_name = os.path.basename(path)
            if len(short_name) > 24:
                short_name = short_name[:21] + "..."
            self.lbl_filename.config(text=f"{short_name} ({self.bgr_img.shape[1]}x{self.bgr_img.shape[0]})")
            self.set_pill("Loaded", "#6366f1", "#1e1e38")
            self.set_status(f"Loaded {os.path.basename(path)}")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def start_canvas_snipper(self):
        self.withdraw()
        CanvasSnipper(self, self._on_canvas_selected)

    def _on_canvas_selected(self, x, y, w, h):
        self.deiconify()
        self.canvas_x = x
        self.canvas_y = y
        self.canvas_w = w
        self.canvas_h = h
        self.lbl_canvas_info.config(text=f"{w} x {h} at ({x}, {y})")
        self.set_pill("Calibrated", "#10b981", "#15261f")
        self.set_status(f"Canvas set to {w}x{h} px at ({x}, {y})")
        self.update_preview()

    # --------------------------------------------------------------------------
    # Live Preview Generation
    # --------------------------------------------------------------------------

    def update_preview(self):
        if self.gray_img is None:
            return

        mode = self.mode_var.get()
        thresh = self.threshold_var.get()
        invert = self.invert_var.get()
        epsilon = self.epsilon_var.get()
        step = self.step_var.get()

        orig_h, orig_w = self.gray_img.shape
        new_w, new_h, off_x, off_y = ImageProcessor.compute_scaled_dims(
            orig_w, orig_h, self.canvas_w, self.canvas_h
        )
        self.scaled_w, self.scaled_h = new_w, new_h
        self.offset_x, self.offset_y = off_x, off_y

        resized_gray = cv2.resize(self.gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if mode == "Contour / Vector Sketch":
            mask = ImageProcessor.get_binary_mask(resized_gray, thresh, invert)
            strokes = ImageProcessor.generate_contour_strokes(mask, epsilon=epsilon)
        elif mode == "Canny Edge Sketch":
            strokes = ImageProcessor.generate_canny_strokes(resized_gray, low_thresh=thresh//2, high_thresh=thresh, epsilon=epsilon)
        elif mode == "Serpentine Raster Fill":
            mask = ImageProcessor.get_binary_mask(resized_gray, thresh, invert)
            strokes = ImageProcessor.generate_serpentine_raster_strokes(mask, step=step)
        elif mode == "Hybrid (Outline + Fill)":
            mask = ImageProcessor.get_binary_mask(resized_gray, thresh, invert)
            strokes = ImageProcessor.generate_hybrid_strokes(mask, epsilon=epsilon, step=step)
        else:
            strokes = []

        self.generated_strokes = strokes

        total_strokes = len(strokes)
        total_points = sum(len(s) for s in strokes)

        profile = self.profile_var.get()
        if "Pixel Tap" in profile:
            est_seconds = int(total_points * 0.024)
        elif "Roblox (Smooth" in profile:
            est_seconds = int(total_strokes * 0.015 + total_points * 0.012)
        else:
            speed_mode = self.speed_var.get()
            delay = 0.001 if "Turbo" in speed_mode else 0.003 if "Fast" in speed_mode else 0.006 if "Normal" in speed_mode else 0.012
            est_seconds = int(total_strokes * delay + total_points * 0.0025)

        m, s = divmod(est_seconds, 60)
        time_str = f"{m}m {s}s" if m > 0 else f"{s}s"

        self.lbl_stats.config(
            text=f"{total_strokes:,} strokes  |  {total_points:,} points  |  Size: {new_w}x{new_h} px  |  Est. Time: ~{time_str}"
        )

        # Render preview onto virtual white canvas
        preview_rgb = ImageProcessor.render_preview(strokes, self.canvas_w, self.canvas_h, off_x, off_y)

        # Scale preview cleanly inside Tkinter preview canvas view
        vw = max(50, self.preview_canvas.winfo_width())
        vh = max(50, self.preview_canvas.winfo_height())
        scale = min(vw / self.canvas_w, vh / self.canvas_h, 1.0)
        dw = max(10, int(self.canvas_w * scale))
        dh = max(10, int(self.canvas_h * scale))

        disp_img = cv2.resize(preview_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
        self._preview_photo = ImageTk.PhotoImage(pil_img)

        self.preview_canvas.delete("all")
        cx = vw // 2
        cy = vh // 2
        self.preview_canvas.create_image(cx, cy, image=self._preview_photo, anchor=tk.CENTER)

    # --------------------------------------------------------------------------
    # Drawing Execution & Bulletproof State Management
    # --------------------------------------------------------------------------

    def start_drawing(self):
        if not self.generated_strokes:
            messagebox.showwarning("No Strokes", "Please select an image and calibrate canvas first.")
            return

        # UI state transition: active drawing
        self.btn_start.config(state=tk.DISABLED, bg="#2a2a38", fg=self.c_text_muted)
        self.btn_abort.config(state=tk.NORMAL, bg="#3b1d22", fg="#f87171")
        self.set_pill("Starting...", "#f59e0b", "#261e12")
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(self.generated_strokes)

        self.worker = DrawingWorker(
            strokes=self.generated_strokes,
            start_x=self.canvas_x,
            start_y=self.canvas_y,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            profile=self.profile_var.get(),
            speed_mode=self.speed_var.get(),
            countdown_secs=self.countdown_var.get(),
            event_queue=self.event_queue
        )
        self.worker.start()

    def abort_drawing(self):
        """Immediately halts worker and resets mouse."""
        if self.worker and self.worker.is_alive():
            self.worker.abort()
            GameInputEngine.safe_release_mouse()
            self.set_status("Aborting drawing...")

    # Event handlers dispatched on the main thread
    def on_worker_countdown(self, count):
        if count > 0:
            self.set_pill(f"Starting in {count}s", "#f59e0b", "#261e12")
            self.set_status(f"Starting in {count}s... Focus your drawing canvas now! (F8 to cancel)")
        else:
            self.set_pill("Drawing", "#6366f1", "#1e1e38")
            self.set_status("Drawing in progress... Press F8 or move mouse to corner to stop.")

    def on_worker_progress(self, current, total):
        self.progress_bar["value"] = current
        pct = int((current / total) * 100)
        self.set_status(f"Drawing: stroke {current:,} / {total:,} ({pct}%)")

    def on_worker_finish(self, success, msg):
        """Cleanly re-enables Start button on main thread without requiring app restart."""
        self.btn_start.config(state=tk.NORMAL, bg="#4f46e5", fg="white")
        self.btn_abort.config(state=tk.DISABLED, bg="#261a1d", fg="#f87171")
        self.worker = None

        if success:
            self.set_pill("Complete", "#10b981", "#15261f")
        else:
            self.set_pill("Stopped", "#ef4444", "#261417")

        self.set_status(msg)

    def set_pill(self, text, fg, bg):
        self.pill_status.config(text=text, fg=fg, bg=bg)

    def set_status(self, text):
        self.lbl_footer.config(text=text)


# ==============================================================================
# 6. CLI RUNNER
# ==============================================================================

def run_cli_mode(args):
    print("=== AutoDraw CLI ===")
    img_path = args.image
    if not img_path:
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(title="Select image")
        if not img_path:
            print("No image selected.")
            return

    bgr, gray = ImageProcessor.load_image(img_path)
    h, w = gray.shape

    if args.canvas:
        cx, cy, cw, ch = [int(v.strip()) for v in args.canvas.split(",")]
    else:
        print("\nMove cursor to TOP-LEFT of canvas and press Enter...")
        input()
        tl_x, tl_y = pyautogui.position()
        print("Move cursor to BOTTOM-RIGHT of canvas and press Enter...")
        input()
        br_x, br_y = pyautogui.position()
        cx = min(tl_x, br_x)
        cy = min(tl_y, br_y)
        cw = abs(br_x - tl_x)
        ch = abs(br_y - tl_y)

    print(f"Canvas: {cw}x{ch} at ({cx}, {cy})")
    nw, nh, ox, oy = ImageProcessor.compute_scaled_dims(w, h, cw, ch)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    if args.mode == "canny":
        strokes = ImageProcessor.generate_canny_strokes(resized, args.threshold//2, args.threshold, epsilon=args.epsilon)
    elif args.mode == "raster":
        mask = ImageProcessor.get_binary_mask(resized, args.threshold, args.invert)
        strokes = ImageProcessor.generate_serpentine_raster_strokes(mask, step=args.step)
    elif args.mode == "hybrid":
        mask = ImageProcessor.get_binary_mask(resized, args.threshold, args.invert)
        strokes = ImageProcessor.generate_hybrid_strokes(mask, epsilon=args.epsilon, step=args.step)
    else:
        mask = ImageProcessor.get_binary_mask(resized, args.threshold, args.invert)
        strokes = ImageProcessor.generate_contour_strokes(mask, epsilon=args.epsilon)

    print(f"Strokes: {len(strokes)}. Press F8 or Esc anytime to abort.")
    for i in range(args.countdown, 0, -1):
        print(f"{i}...", end="", flush=True)
        time.sleep(1)
    print("\nDrawing...")

    worker = DrawingWorker(
        strokes=strokes,
        start_x=cx, start_y=cy,
        offset_x=ox, offset_y=oy,
        profile="Standard (Paint, Photoshop, Web)",
        speed_mode="Fast (2ms)",
        countdown_secs=0
    )
    worker.run()
    print("\nDone!")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AutoDraw Studio")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--cli", action="store_true", help="Run without GUI")
    parser.add_argument("--mode", choices=["contour", "raster", "canny", "hybrid"], default="contour")
    parser.add_argument("--canvas", type=str, help="Canvas box as x,y,w,h")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1.2)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

    if args.cli:
        run_cli_mode(args)
    else:
        app = ModernAutoDrawStudio(initial_image=args.image)
        app.mainloop()


if __name__ == "__main__":
    main()