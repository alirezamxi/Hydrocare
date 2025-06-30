import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import csv
from tkinter import filedialog, Tk, simpledialog

# === ToF Cleaning Functions ===
def get_median(history, zone_index):
    values = [frame[zone_index] for frame in history]
    values = [v for v in values if 0 < v < 6000]
    if not values:
        return np.random.randint(10, 51)
    values.sort()
    return values[len(values) // 2]

def smooth_tof_frames(history):
    median_filtered = np.zeros(64, dtype=int)

    # Step 1: Temporal median filter
    for i in range(64):
        median_filtered[i] = get_median(history, i)

    # Step 2: Spatial smoothing
    smoothed = median_filtered.copy().reshape((8, 8))
    output = smoothed.copy()

    for row in range(1, 7):
        for col in range(1, 7):
            val = smoothed[row, col]
            neighbors = [
                smoothed[row - 1, col - 1], smoothed[row - 1, col], smoothed[row - 1, col + 1],
                smoothed[row, col - 1],                             smoothed[row, col + 1],
                smoothed[row + 1, col - 1], smoothed[row + 1, col], smoothed[row + 1, col + 1],
            ]
            valid_neighbors = [n for n in neighbors if 0 < n < 6000]
            if valid_neighbors:
                avg = sum(valid_neighbors) / len(valid_neighbors)
                if val < 0.5 * avg:
                    output[row, col] = int(avg)

    return output.flatten()

# === Main GUI App ===
def main():
    root = Tk()
    root.withdraw()
    rgb_folder = filedialog.askdirectory(title="Select RGB Image Folder")
    tof_path = filedialog.askopenfilename(title="Select ToF Text File")

    # === Load ToF raw data ===
    raw_tof = {}
    with open(tof_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            ts = int(lines[i].strip())
            values = list(map(int, lines[i + 1].strip().split(",")))
            if len(values) == 64:
                raw_tof[ts] = np.array(values)

    # === Load RGB images ===
    rgb_files = sorted([
        f for f in os.listdir(rgb_folder) if f.endswith(".jpg") or f.endswith(".png")
    ], key=lambda x: int(os.path.splitext(x)[0]))

    timestamps = [int(os.path.splitext(f)[0]) for f in rgb_files]
    rgb_paths = [os.path.join(rgb_folder, f) for f in rgb_files]

    index = 0
    drinking_active = False
    last_drinking_state = False
    labels = {}
    volumes = {}
    cleaned_tof = {}
    history_buffer = []

    # === Prompt for initial volume ===
    current_volume = simpledialog.askfloat("Initial Volume", "Enter initial volume (ml):", parent=root)

    # === Custom colormap: Blue (near) to Yellow (far) ===
    blue_yellow_cmap = LinearSegmentedColormap.from_list("blue_yellow", [(0, 0, 1), (1, 1, 0)], N=256)

    while 0 <= index < len(rgb_paths):
        timestamp = timestamps[index]
        rgb = cv2.imread(rgb_paths[index])
        rgb = cv2.resize(rgb, (640, 480))
        rgb_disp = rgb.copy()

        # === ToF Processing ===
        if timestamp in raw_tof:
            history_buffer.append(raw_tof[timestamp])
            if len(history_buffer) > 3:
                history_buffer.pop(0)

            if len(history_buffer) == 3:
                filtered = smooth_tof_frames(history_buffer)
            else:
                filtered = raw_tof[timestamp]
        else:
            filtered = np.zeros(64, dtype=int)

        # Save cleaned and flipped ToF frame
        tof_matrix = filtered.reshape((8, 8))
        tof_matrix = np.flipud(np.fliplr(tof_matrix))  # Flip vertically and horizontally
        cleaned_tof[timestamp] = tof_matrix.flatten()

        # === Generate heatmap ===
        norm = Normalize(vmin=tof_matrix.min(), vmax=tof_matrix.max())
        colormap = cm.ScalarMappable(norm=norm, cmap=blue_yellow_cmap)
        heatmap = (colormap.to_rgba(tof_matrix)[:, :, :3] * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (640, 480), interpolation=cv2.INTER_NEAREST)

        # === Label and volume assignment ===
        label = "Drinking" if drinking_active else "Not_Drinking"
        labels[timestamp] = label
        volumes[timestamp] = current_volume

        cv2.putText(rgb_disp, f"Time: {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rgb_disp, f"Status: {label}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rgb_disp, f"Volume: {current_volume:.2f} ml", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        combined = np.hstack((rgb_disp, heatmap))
        cv2.imshow("Drinking Labeling - [D] toggle | [Space] next | [B] back | [Esc] save & exit", combined)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Next
            index += 1
        elif key == ord('b') and index > 0:  # Back
            index -= 1
        elif key == ord('d'):  # Toggle gesture state
            drinking_active = not drinking_active

            # When going from Drinking → Not_Drinking, ask for new volume
            if last_drinking_state and not drinking_active:
                current_volume = simpledialog.askfloat("New Volume", "Enter volume after drinking (ml):", parent=root)

            last_drinking_state = drinking_active

    cv2.destroyAllWindows()

    # === Save to CSV ===
    out_csv = filedialog.asksaveasfilename(title="Save Labeled CSV", defaultextension=".csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time"] + [f"Zone_{i}" for i in range(64)] + ["Label", "Volume"])
        for ts in timestamps:
            values = cleaned_tof.get(ts, np.zeros(64)).astype(int).tolist()
            writer.writerow([ts] + values + [labels.get(ts, "Not_Drinking"), volumes.get(ts, "")])
    print(f"✅ CSV saved at: {out_csv}")

if __name__ == "__main__":
    main()