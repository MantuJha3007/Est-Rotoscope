# SAM 2 Unlimited Rotoscoping Studio

A dedicated, local, GPU-accelerated video rotoscoping application powered by **Meta's Segment Anything Model 2 (SAM 2.1)**, tailored specifically for **DaVinci Resolve** workflows.

---

## 🚀 Key Advantages Over Commercial Plugins

* **No 30-Second Limit:** Process clips of any duration (1 minute, 5 minutes, 15 minutes, or more).
* **Unlimited Usage:** Zero token limits, paywalls, or monthly credit caps.
* **GPU & VRAM Optimized:** Uses SAM 2.1 memory offloading (`offload_video_to_cpu` and `offload_state_to_cpu`) so long clips never exceed your **NVIDIA RTX 4060 8GB VRAM**.
* **Zero Compression Artifacts:** High-precision point prompting with real-time feedback.

---

## 📁 Project Structure

```text
D:\Mantu Jha\Project\Sam2\
│
├── run.bat                   <-- Double-click to start! (Launches server and browser)
├── setup.bat                 <-- One-time automated environment installer
├── app.py                    <-- Web UI backend (FastAPI)
├── requirements.txt          <-- Dependencies
│
├── core/
│   ├── sam2_engine.py        <-- SAM 2 tracking engine with CPU offload & memory purging
│   ├── model_manager.py      <-- Auto-downloader for SAM 2.1 models (tiny, small, base+, large)
│   ├── video_utils.py        <-- Frame extraction and timecode helpers
│   └── exporter.py           <-- Renders B&W Matte MP4, PNG sequence, Green Screen
│
├── static/
│   ├── css/style.css         <-- Studio dark-mode UI
│   └── js/
│       ├── canvas.js         <-- Interactive canvas with point prompts
│       └── app.js            <-- Player scrub, playback, and tracking logic
│
├── templates/
│   └── index.html            <-- Interface
│
├── inputs/                   <-- Put your source video clips here (MP4, MOV, MKV)
├── models/                   <-- SAM 2.1 model checkpoints (.pt)
└── outputs/                  <-- Generated matte videos & alpha sequences
```

---

## ⚡ Quick Start

1. **One-Time Setup:**
   * Double-click `setup.bat` (or let `run.bat` run it automatically). It creates a dedicated Python environment, installs PyTorch with CUDA acceleration, and gets the SAM 2.1 checkpoint.

2. **Launch:**
   * Double-click `run.bat`.
   * Your browser will automatically open `http://127.0.0.1:8000`.

---

## 🎯 How to Rotoscope Any Clip

1. **Load Video:** Drop your video file into `inputs/` or click **Upload Video** in the UI, then click **Load**.
2. **Add Keyframe Prompts:**
   * On Frame 0, **Left-Click** on your subject to place a 🟢 **Green (Include)** point.
   * **Right-Click** anywhere outside to place a 🔴 **Red (Exclude)** point to refine edges.
   * A cyan overlay will immediately preview the segmented mask.
3. **Refine (Optional):**
   * Scrub the timeline. If the subject undergoes extreme motion or occlusion at a later frame (e.g., Frame 180), click on that frame to add a corrective point. SAM 2 will automatically use both keyframes!
4. **Track:**
   * Click **⚡ Track Entire Video**.
   * Watch the live progress bar as SAM 2 propagates the mask through every frame.
5. **Export:**
   * Click **💾 Export Matte** and select **Black & White Matte Video (.mp4)**.

---

## 🎬 How to Use the Exported Matte in DaVinci Resolve

### Method 1: Timeline Edit Page (Fastest)
1. In DaVinci Resolve, place your original clip on **Video Track 1**.
2. Place the exported `_matte.mp4` directly above it on **Video Track 2**.
3. Select the matte clip on Track 2, open the **Inspector ➔ Composite**, and set **Composite Mode** to **Multiply** (or use it with an adjustment clip/luma keyer).

### Method 2: Color Page Studio Grade (Recommended for Color Grading & VFX)
1. Import both the original video and the exported `_matte.mp4` into your **Media Pool**.
2. Go to the **Color Page**.
3. In the Node Graph, right-click an empty area and select:  
   **Add Source ➔ External Matte ➔ Select your exported matte file**.
4. A matte node will appear with a blue mask output.
5. Drag a line from the **Blue Output** of the Matte node into the **Blue Input** of any Corrector node or Alpha output.
6. Now you have 100% control over background color grading, subject isolations, blurs, or glow effects with zero artifacts!
