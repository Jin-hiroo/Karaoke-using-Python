import os
import time
import pyaudio
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mido
import pygame
import threading
import random
import matplotlib.patches as patches
import sys
import librosa

# --- Audio Settings ---
SAMPLE_RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024

# --- Global Variables ---
current_pitch = 0
freq_history = []  # Stores real-time pitch values
MAX_HISTORY = 5  # Time window in seconds
CENTER_OFFSET = MAX_HISTORY / 2  # Center the scrolling view
note_boxes = []  # Active note boxes for MIDI notes
filled_boxes = {}  # Tracks progress for filling boxes per note
START_TIME = None  # Tracks when the MP3 starts playing
stream = None  # PyAudio stream object
p = None  # PyAudio instance
TARGET_PITCH_HISTORY = []
SCROLL_SPEED = 0.5  # Speed of the scrolling view
MP3_START_OFFSET = 0  # Offset in seconds to start the audio track

# --- File Paths ---
midi_file = "colors-of-the-Wind-Bass.mid"
mp3_file = "Colors of the Wind #Bass.mp3"

# --- Check Files ---
if not os.path.exists(midi_file):
    print(f"MIDI file not found: {midi_file}")
    exit()
if not os.path.exists(mp3_file):
    print(f"MP3 file not found: {mp3_file}")
    exit()

# --- MIDI Loading Function ---
def load_midi(midi_file_path):
    try:
        mid = mido.MidiFile(midi_file_path)
        notes = []
        current_time = 0
        ticks_per_beat = mid.ticks_per_beat
        tempo = 606060

        active_notes = {}

        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                current_time += msg.time
                seconds = mido.tick2second(current_time, ticks_per_beat, tempo)

                if msg.type == "note_on" and msg.velocity > 0:
                    frequency = librosa.midi_to_hz(msg.note)
                    active_notes[msg.note] = {
                        'start_time': seconds,
                        'frequency': frequency
                    }
                elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                    if msg.note in active_notes:
                        note_data = active_notes.pop(msg.note)
                        note_data['end_time'] = seconds
                        notes.append(note_data)

        return preprocess_midi(notes)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        exit()

# --- Preprocess MIDI Notes ---
def preprocess_midi(midi_notes):
    for note in midi_notes:
        note['visible'] = False
    return midi_notes

# --- Audio Processing ---
def get_pitch(audio_data, sample_rate):
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    n = len(y)
    windowed = y * np.hanning(n)  # Apply a window function
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    peak_idx = np.argmax(magnitude)
    pitch = frequencies[peak_idx] if peak_idx > 0 else 0
    return pitch if pitch > 0 else 0


import threading
audio_lock = threading.Lock()

def audio_callback(in_data, frame_count, time_info, status):
    global current_pitch, freq_history, START_TIME
    with audio_lock:
        if in_data:
            pitch = get_pitch(in_data, SAMPLE_RATE)
            if pitch > 0:
                current_pitch = pitch
                elapsed_time = time.time() - START_TIME
                freq_history.append((elapsed_time, pitch))
                if len(freq_history) > MAX_HISTORY * SAMPLE_RATE / CHUNK:
                    freq_history.pop(0)
            else:
                current_pitch = 0
        else:
            current_pitch = 0
    return (in_data, pyaudio.paContinue)

# --- Stop Button ---
def stop_program():
    global stream, p
    if stream:
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()
    pygame.mixer.quit()
    window.destroy()
    sys.exit()

# --- Tkinter Visualization ---
window = tk.Tk()
window.title("Karaoke Practice")

fig = Figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111)
ax.set_xlim(-MAX_HISTORY / 4, MAX_HISTORY - (MAX_HISTORY / 4))
ax.set_ylim(50, 2000)
ax.set_yscale('log')
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Frequency (Hz)")

# Add the visualization elements
user_line, = ax.plot([], [], label="User Pitch", color='b')
center_line = ax.axvline(0, color='green', linestyle='--')
canvas = FigureCanvasTkAgg(fig, master=window)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

stop_button = tk.Button(window, text="Stop", command=stop_program, bg="red", fg="white")
stop_button.pack(side=tk.BOTTOM, pady=10)

# --- Sparkle Settings ---
SPARKLE_THRESHOLD = 5  # Frequency difference to trigger a sparkle
SPARKLE_COUNT = 15  # Number of sparkles for the effect
SPARKLE_MAX_SIZE = 30  # Maximum size of sparkles
SPARKLE_MIN_SIZE = 10  # Minimum size of sparkles
SPARKLE_COLOR = 'gold'  # Color of sparkles

# --- Draw Sparkles Dynamically ---
def draw_sparkles():
    global TARGET_PITCH_HISTORY, current_pitch, filled_boxes

    for artist in ax.collections:
        artist.remove()

    if current_pitch > 0:
        elapsed_time = time.time() - START_TIME
        for note in TARGET_PITCH_HISTORY:
            note_key = (note['start_time'], note['frequency'])

            if abs(current_pitch - note['frequency']) <= SPARKLE_THRESHOLD:
                if note_key not in filled_boxes:
                    filled_boxes[note_key] = [(note['start_time'], elapsed_time)]
                else:
                    filled_boxes[note_key].append((filled_boxes[note_key][-1][1], elapsed_time))

            # Draw the progress-filled parts of the target box
            if note_key in filled_boxes:
                for fill_start, fill_end in filled_boxes[note_key]:
                    progress_start = fill_start - elapsed_time
                    progress_end = fill_end - elapsed_time
                    progress_width = progress_end - progress_start

                    if progress_width > 0:
                        rect = patches.Rectangle(
                            (progress_start, note['frequency'] - 5),
                            max(0, progress_width), 10,
                            edgecolor='none', facecolor='yellow', alpha=0.5
                        )
                        ax.add_patch(rect)

                # Generate sparkles only when matching the pitch
                if abs(current_pitch - note['frequency']) <= SPARKLE_THRESHOLD:
                    sparkle_x = [0 for _ in range(SPARKLE_COUNT)]
                    sparkle_y = [
                        note['frequency'] + random.uniform(-2, 2)
                        for _ in range(SPARKLE_COUNT)
                    ]
                    sparkle_sizes = [
                        random.uniform(SPARKLE_MIN_SIZE, SPARKLE_MAX_SIZE)
                        for _ in range(SPARKLE_COUNT)
                    ]
                    ax.scatter(
                        sparkle_x, sparkle_y,
                        s=sparkle_sizes, c=SPARKLE_COLOR, marker='*', alpha=0.8
                    )

# --- Update Visualization ---
def update_target_notes():
    global TARGET_PITCH_HISTORY
    elapsed_time = time.time() - START_TIME
    TARGET_PITCH_HISTORY = [
        note for note in midi_notes
        if note['end_time'] > elapsed_time - MAX_HISTORY and note['start_time'] < elapsed_time + MAX_HISTORY
    ]

def draw_note_boxes():
    global note_boxes
    for box in note_boxes:
        box.remove()
    note_boxes = []

    elapsed_time = time.time() - START_TIME

    for note in TARGET_PITCH_HISTORY:
        start_time = note['start_time'] - elapsed_time
        end_time = note['end_time'] - elapsed_time
        rect = patches.Rectangle(
            (start_time, note['frequency'] - 5),
            end_time - start_time, 10,
            edgecolor='red', facecolor='none', linewidth=2, linestyle='-'
        )
        rect.set_capstyle('round')
        ax.add_patch(rect)
        note_boxes.append(rect)

# --- Update Plot ---
def update_plot():
    elapsed_time = time.time() - START_TIME

    # Update user pitch line
    x_user = [t[0] - elapsed_time for t in freq_history if t[1] > 0]
    y_user = [t[1] for t in freq_history if t[1] > 0]
    user_line.set_data(x_user, y_user)

    # Update target notes and draw note boxes
    update_target_notes()
    draw_note_boxes()

    # Update filled boxes
    for note_key, fills in list(filled_boxes.items()):
        updated_fills = []
        for fill_start, fill_end in fills:
            progress_start = fill_start - elapsed_time
            progress_end = fill_end - elapsed_time
            progress_width = progress_end - progress_start
            if progress_width > 0:
                updated_fills.append((fill_start, fill_end))
        if updated_fills:
            filled_boxes[note_key] = updated_fills
        else:
            del filled_boxes[note_key]

    draw_sparkles()

    ax.set_xlim(-MAX_HISTORY / 4, MAX_HISTORY - (MAX_HISTORY / 4))
    canvas.draw()
    window.after(10, update_plot)

# --- MP3 Player ---
def play_mp3(mp3_path):
    def audio_thread():
        global START_TIME
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(mp3_path)
            START_TIME = time.time() + MP3_START_OFFSET
            pygame.mixer.music.play(start=MP3_START_OFFSET)
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Pygame mixer error: {e}")

    threading.Thread(target=audio_thread, daemon=True).start()

# --- Main Execution ---
if __name__ == "__main__":
    midi_notes = load_midi(midi_file)
    play_mp3(mp3_file)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)
    update_plot()
    window.mainloop()
    stream.stop_stream()
    stream.close()
    p.terminate()
