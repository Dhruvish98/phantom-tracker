"""Generate the Phantom Tracker midterm project report as a PDF."""

from fpdf import FPDF


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Phantom Tracker  - Midterm Project Report", align="R")
        self.ln(4)
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 70, 140)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet(self, text, indent=15):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        self.cell(5, 5.5, "-" + " ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def key_value(self, key, value, indent=15):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5.5, f"{key}: ", new_x="END")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 5.5, value, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER PAGE ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 15, "Phantom Tracker", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Real-Time Multi-Object Tracking with Occlusion Handling", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "and Re-Identification", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_draw_color(0, 102, 204)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "Midterm Progress Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Computer Vision Course", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "SP Jain School of Global Management", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Team Members", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for member in [
        "Dhruvish Parekh  - Tracking Module",
        "Divyansh Gupta  - Detection Module",
        "Dharmik Naicker  - Re-Identification Module",
        "Agastya Ramakrishnan  - Visualization Module",
    ]:
        pdf.cell(0, 6, member, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Date: March 4, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Repository: github.com/Dhruvish98/phantom-tracker", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── 1. PROBLEM STATEMENT ──
    pdf.add_page()
    pdf.section_title("1", "Problem Statement")
    pdf.body_text(
        "Multi-object tracking (MOT) in video surveillance and real-time monitoring systems faces "
        "critical challenges when objects become temporarily occluded  - hidden behind other objects, "
        "passing behind pillars, or briefly leaving the camera's field of view. Existing tracking "
        "systems often lose track of objects during occlusion events, assigning new identities when "
        "the object reappears. This leads to fragmented tracking histories, inaccurate counts, and "
        "unreliable analytics."
    )
    pdf.body_text(
        "The core technical challenges include:"
    )
    pdf.bullet("Identity Persistence: Maintaining consistent object IDs across occlusion events lasting 1-30+ seconds.")
    pdf.bullet("Motion Prediction: Estimating where an occluded object is likely to be, even when it is not visible.")
    pdf.bullet("Appearance Matching: Re-identifying returning objects using visual features despite changes in pose, lighting, and partial visibility.")
    pdf.bullet("Real-Time Performance: Processing all of detection, tracking, re-identification, and visualization within acceptable latency for live video feeds.")

    # ── 2. OBJECTIVE ──
    pdf.section_title("2", "Objective")
    pdf.body_text(
        "The objective of Phantom Tracker is to build a complete, modular, real-time multi-object "
        "tracking pipeline that:"
    )
    pdf.bullet("Detects objects in each video frame using state-of-the-art YOLO object detection.")
    pdf.bullet("Tracks detected objects across frames with persistent identity assignment using BoT-SORT (Bag of Tricks for SORT).")
    pdf.bullet("Handles occlusion gracefully through a custom state machine that transitions objects through Active, Occluded, Lost, and Deleted states.")
    pdf.bullet("Re-identifies returning objects using deep appearance features (OSNet) with a temporal-decay feature bank.")
    pdf.bullet("Visualizes all tracking state in real-time with bounding boxes, trajectory trails, ghost outlines for occluded objects, predicted paths, heatmaps, and an analytics dashboard.")
    pdf.body_text(
        "The system targets indoor surveillance scenarios (e.g., subway stations, offices, retail stores) "
        "where people and objects frequently occlude each other."
    )

    # ── 3. PROJECT ARCHITECTURE ──
    pdf.section_title("3", "System Architecture and Specifications")
    pdf.sub_title("3.1 Pipeline Architecture")
    pdf.body_text(
        "Phantom Tracker uses a four-stage sequential pipeline that processes each video frame:"
    )
    pdf.body_text(
        "Stage 1  - Detection: Raw video frame is passed to YOLOv11, which outputs bounding boxes, "
        "confidence scores, and class labels for all detected objects.\n\n"
        "Stage 2  - Tracking: Detections are fed to BoT-SORT (via the boxmot library), which performs "
        "data association using the Hungarian algorithm, Kalman filtering for motion estimation, and "
        "short-term appearance matching. Our custom state machine then classifies each track as Active, "
        "Occluded, Lost, or Deleted based on visibility history.\n\n"
        "Stage 3  - Re-Identification: When lost tracks exist alongside new detections, the Re-ID module "
        "extracts OSNet embeddings and compares them against the feature galleries of lost tracks using "
        "temporally-weighted cosine similarity. Confident matches reactivate lost tracks with their "
        "original identities.\n\n"
        "Stage 4  - Visualization: The output frame is rendered with all overlays  - bounding boxes, "
        "trajectory trails, ghost outlines for occluded tracks, predicted paths, a heatmap, and an "
        "analytics dashboard showing real-time statistics."
    )

    pdf.sub_title("3.2 Technical Stack")
    pdf.key_value("Language", "Python 3.10")
    pdf.key_value("Detection", "YOLOv11s (Ultralytics)  - 72.8 FPS on RTX 2050")
    pdf.key_value("Tracking", "BoT-SORT via boxmot v16  - Kalman + Hungarian + Appearance")
    pdf.key_value("Re-ID", "OSNet x1.0 via torchreid  - 512-d L2-normalized embeddings")
    pdf.key_value("Visualization", "OpenCV with custom overlay engine")
    pdf.key_value("GPU", "NVIDIA GeForce RTX 2050 (4 GB VRAM), CUDA 13.0")
    pdf.key_value("Framework", "PyTorch 2.10.0+cu128")

    pdf.sub_title("3.3 Occlusion State Machine")
    pdf.body_text(
        "Each tracked object transitions through the following states:\n\n"
        "ACTIVE: Object is visible and being tracked by BoT-SORT. Position, velocity, and appearance "
        "are updated every frame.\n\n"
        "OCCLUDED: Object has not been seen for 3+ frames. Position is extrapolated using EMA-smoothed "
        "velocity. A ghost outline is displayed at the predicted location.\n\n"
        "LOST: Object has not been seen for 30+ frames. Position extrapolation stops. The Re-ID module "
        "actively searches for this object among new detections.\n\n"
        "DELETED: Object has not been seen for 300+ frames. Track is permanently removed from memory."
    )

    pdf.sub_title("3.4 Motion Prediction")
    pdf.body_text(
        "Velocity is computed using Exponential Moving Average (EMA) smoothing with alpha=0.3, providing "
        "a balance between responsiveness and stability. For occluded tracks, position is linearly "
        "extrapolated using the smoothed velocity vector, clamped to frame boundaries. For active tracks, "
        "a predicted trajectory is computed for the next 15 frames for visualization purposes."
    )

    # ── 4. MEMBER CONTRIBUTIONS ──
    pdf.add_page()
    pdf.section_title("4", "Member Contributions")

    pdf.sub_title("4.1 Divyansh Gupta  - Detection Module")
    pdf.body_text("File: detection/detector.py")
    pdf.bullet("Implemented YOLOv11 inference pipeline with automatic GPU detection and device management.")
    pdf.bullet("Added class filtering capability allowing selective detection (e.g., person-only mode for cleaner demos), passed directly to YOLO's native classes parameter for zero-overhead filtering.")
    pdf.bullet("Implemented FP16 half-precision inference support for GPU, providing approximately 2x speedup.")
    pdf.bullet("Built benchmark utility for FPS profiling across different YOLO model sizes (yolo11n: 70.5 FPS, yolo11s: 72.8 FPS on RTX 2050).")
    pdf.bullet("Standardized detection output format (Detection and FrameDetections dataclasses) for seamless integration with the tracking pipeline.")
    pdf.bullet("Future scope: TensorRT export for maximum inference speed, Grounding DINO integration for open-vocabulary detection.")

    pdf.sub_title("4.2 Dhruvish Parekh  - Tracking Module")
    pdf.body_text("File: tracking/tracker.py")
    pdf.bullet("Implemented BoT-SORT multi-object tracking via the boxmot library with deep appearance features for robust data association.")
    pdf.bullet("Designed and built the dual-ledger occlusion state machine (Active/Occluded/Lost/Deleted) that operates independently on top of boxmot's output, detecting disappearances by comparing output IDs against known tracks each frame.")
    pdf.bullet("Implemented EMA-smoothed velocity estimation and linear trajectory prediction for both occluded position extrapolation and future path visualization.")
    pdf.bullet("Tuned 8 BoT-SORT hyperparameters for indoor walking scenarios (track_buffer=90 frames for 3-second occlusion tolerance, lowered appearance threshold for better re-association).")
    pdf.bullet("Built automatic CUDA/CPU device fallback to ensure the tracker runs on any hardware configuration.")
    pdf.bullet("Implemented IoU-based fallback tracker for environments where boxmot is not installed.")
    pdf.bullet("Integrated Re-ID events pipeline  - tracker stores re-identification events and passes them to the analytics dashboard via AnalyticsSnapshot.")
    pdf.bullet("Wired appearance gallery updates in the main pipeline so active tracks accumulate OSNet embeddings every 5th frame for Re-ID matching.")
    pdf.bullet("Maintained shared interfaces (core/interfaces.py) and main pipeline orchestration (main.py).")
    pdf.bullet("Future scope: LSTM-based motion model for non-linear trajectory prediction.")

    pdf.sub_title("4.3 Dharmik Naicker  - Re-Identification Module")
    pdf.body_text("File: reid/reidentifier.py")
    pdf.bullet("Implemented OSNet feature extraction via torchreid, producing 512-dimensional L2-normalized embeddings from person crops.")
    pdf.bullet("Designed the FeatureBank class with a per-track appearance gallery using temporal decay weighting (decay_rate=0.95), ensuring recent appearances are weighted more heavily during matching.")
    pdf.bullet("Built the cosine similarity matching pipeline that compares new detection embeddings against all lost track galleries, with configurable confidence threshold (default 0.6).")
    pdf.bullet("Implemented on-the-fly embedding extraction  - when detections don't carry pre-computed embeddings, the Re-ID module extracts them directly from the raw frame crop.")
    pdf.bullet("Added safe cropping with boundary clamping and minimum size validation to handle edge-case detections.")
    pdf.bullet("Provided statistics API (get_stats) for ablation study metrics: total queries, matches, rejections, and match rate.")
    pdf.bullet("Future scope: Cross-camera Re-ID, appearance-based clustering for unknown identities.")

    pdf.sub_title("4.4 Agastya Ramakrishnan  - Visualization Module")
    pdf.body_text("File: visualization/visualizer.py")
    pdf.bullet("Built a comprehensive 800+ line visualization engine with all overlay types: color-coded bounding boxes, trajectory trails with fade effect, ghost outlines for occluded tracks with uncertainty indicators, and predicted path rendering with uncertainty cones.")
    pdf.bullet("Implemented a real-time analytics dashboard (toggle with 'd' key) showing current object count, total entries/exits, per-track speeds, dwell times, and recent Re-ID events.")
    pdf.bullet("Built a spatial heatmap overlay (toggle with 'h' key) showing cumulative object density across the frame.")
    pdf.bullet("Implemented speed indicators on active tracks and Re-ID notification overlays when tracks are re-identified.")
    pdf.bullet("Added interactive keyboard controls for toggling individual visualization features during playback.")
    pdf.bullet("Created comprehensive test suite and visualization showcase demo.")
    pdf.bullet("Future scope: 3D trajectory rendering, zone-based analytics, exportable reports.")

    # ── 5. CURRENT PROGRESS ──
    pdf.add_page()
    pdf.section_title("5", "Current Progress (Midterm)")
    pdf.body_text("All four modules are implemented, integrated, and running as a unified pipeline.")

    pdf.sub_title("5.1 Completed Milestones")
    pdf.bullet("Full end-to-end pipeline: Detection -> Tracking -> Re-ID -> Visualization running on GPU.")
    pdf.bullet("BoT-SORT tracking with persistent IDs across occlusion events.")
    pdf.bullet("OSNet-based Re-ID successfully re-identifying returning persons with 60-75% cosine similarity.")
    pdf.bullet("Real-time visualization with all overlays, dashboard, and heatmap.")
    pdf.bullet("GPU acceleration on RTX 2050: YOLO at 72.8 FPS, full pipeline at 6-7 FPS.")
    pdf.bullet("Tested on 6 diverse demo videos: subway pedestrians, slow walkers, busy street, dog park, Tokyo subway, and night street.")
    pdf.bullet("Class filtering for clean demos (--classes person filters at YOLO level).")
    pdf.bullet("Automatic device detection and CPU fallback for portability.")

    pdf.sub_title("5.2 Performance Metrics")
    pdf.key_value("YOLO Detection (GPU)", "72.8 FPS (yolo11s, RTX 2050)")
    pdf.key_value("Full Pipeline (GPU)", "6-7 FPS (detection + tracking + Re-ID + visualization)")
    pdf.key_value("Re-ID Match Accuracy", "60-75% cosine similarity on confident matches")
    pdf.key_value("Occlusion Recovery", "3-second buffer (90 frames @ 30fps)")
    pdf.key_value("Track Persistence", "Tracks maintained through 30-frame occlusions")
    pdf.ln(2)

    pdf.sub_title("5.3 Known Limitations")
    pdf.bullet("Full pipeline FPS is bottlenecked by Re-ID embedding extraction (~100ms per crop on CPU path). Optimization planned for final submission.")
    pdf.bullet("Re-ID similarity scores hover around 60-75% due to single-camera viewpoint changes. Cross-camera scenarios will require model fine-tuning.")
    pdf.bullet("Ghost position extrapolation uses linear prediction, which does not handle curved or non-linear motion patterns. LSTM model planned for final.")

    # ── 6. FUTURE SCOPE ──
    pdf.section_title("6", "Future Scope (Final Submission)")
    pdf.bullet("Grounding DINO Integration: Open-vocabulary detection allowing natural language queries like 'person with red backpack' to initialize specific tracks.")
    pdf.bullet("LSTM Motion Model: Replace linear extrapolation with a learned motion model for more accurate occlusion prediction, especially for non-linear trajectories.")
    pdf.bullet("TensorRT Optimization: Export YOLO and OSNet models to TensorRT for 2-4x inference speedup, targeting 20+ FPS for the full pipeline.")
    pdf.bullet("Cross-Camera Re-ID: Extend the Re-ID module to work across multiple camera feeds, enabling object handoff between views.")
    pdf.bullet("Zone-Based Analytics: Define regions of interest (entry zones, dwell zones) for targeted analytics and alerts.")
    pdf.bullet("Ablation Study: Comprehensive comparison of YOLO model sizes, Re-ID thresholds, and tracking parameters with quantitative metrics (MOTA, IDF1).")
    pdf.bullet("Web Dashboard: Real-time browser-based visualization using WebSocket streaming for remote monitoring.")

    # ── 7. CONCLUSION ──
    pdf.section_title("7", "Conclusion")
    pdf.body_text(
        "Phantom Tracker demonstrates a fully functional, modular multi-object tracking system that "
        "addresses the core challenge of identity persistence through occlusion. By combining "
        "state-of-the-art detection (YOLOv11), robust tracking (BoT-SORT), deep appearance-based "
        "re-identification (OSNet), and comprehensive real-time visualization, the system provides "
        "a practical solution for indoor surveillance scenarios.\n\n"
        "The modular architecture  - with clearly defined interfaces between all four components  - "
        "enables independent development and testing by each team member while ensuring seamless "
        "integration. The system runs on commodity GPU hardware (RTX 2050) and includes automatic "
        "fallbacks for CPU-only environments.\n\n"
        "For the final submission, the focus will be on performance optimization (TensorRT), advanced "
        "motion prediction (LSTM), and extended capabilities (Grounding DINO, cross-camera Re-ID) "
        "to push the system towards production readiness."
    )

    # Save
    pdf.output("docs/Phantom_Tracker_Midterm_Report.pdf")
    print("Report saved: docs/Phantom_Tracker_Midterm_Report.pdf")


if __name__ == "__main__":
    build()
