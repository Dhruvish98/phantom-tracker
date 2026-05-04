"""Generate the Phantom Tracker midterm project report as a detailed PDF."""

from fpdf import FPDF
import math


# ── Colors ──
BLUE = (0, 70, 140)
LIGHT_BLUE = (0, 102, 204)
DARK = (40, 40, 40)
GRAY = (100, 100, 100)
MED_GRAY = (60, 60, 60)
LIGHT_GRAY = (220, 220, 220)
WHITE = (255, 255, 255)
BOX_BLUE = (200, 220, 240)
BOX_GREEN = (210, 240, 210)
BOX_ORANGE = (255, 235, 210)
BOX_RED = (255, 215, 215)


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # no header on cover
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GRAY)
        self.cell(0, 8, "Phantom Tracker  - Midterm Project Report", align="R")
        self.ln(4)
        self.set_draw_color(*LIGHT_BLUE)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*BLUE)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*LIGHT_BLUE)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 90, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.ln(1)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*MED_GRAY)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_sub_title(self, title):
        self.set_font("Helvetica", "BI", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        self.cell(5, 5.5, "- ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def numbered(self, num, text, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        self.cell(8, 5.5, f"{num}. ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def kv(self, key, value, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        w = self.get_string_width(f"{key}: ") + 2
        self.cell(w, 5.5, f"{key}: ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, value)
        self.ln(0.5)

    def code_line(self, text, indent=12):
        self.set_font("Courier", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.set_fill_color(242, 242, 242)
        self.cell(indent, 5, "")
        self.cell(0, 5.5, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.ln(0.5)

    def note_box(self, text, fill_color=BOX_BLUE):
        self.set_fill_color(*fill_color)
        self.set_draw_color(150, 150, 150)
        x, y = self.get_x(), self.get_y()
        self.rect(12, y, 186, 14, style="DF")
        self.set_xy(16, y + 2)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*DARK)
        self.multi_cell(178, 5, text)
        self.set_y(y + 16)

    # ── Diagram drawing helpers ──

    def draw_box(self, x, y, w, h, label, fill_color, sublabel=None):
        """Draw a rounded-ish colored box with label."""
        self.set_fill_color(*fill_color)
        self.set_draw_color(100, 100, 100)
        self.set_line_width(0.4)
        self.rect(x, y, w, h, style="DF")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*DARK)
        text_y = y + h / 2 - 2 if not sublabel else y + h / 2 - 5
        self.set_xy(x, text_y)
        self.cell(w, 5, label, align="C")
        if sublabel:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.set_xy(x, text_y + 5)
            self.cell(w, 4, sublabel, align="C")

    def draw_arrow(self, x1, y1, x2, y2):
        """Draw an arrow from (x1,y1) to (x2,y2)."""
        self.set_draw_color(80, 80, 80)
        self.set_line_width(0.5)
        self.line(x1, y1, x2, y2)
        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        sz = 2.5
        self.line(x2, y2, x2 - sz * math.cos(angle - 0.4), y2 - sz * math.sin(angle - 0.4))
        self.line(x2, y2, x2 - sz * math.cos(angle + 0.4), y2 - sz * math.sin(angle + 0.4))

    def draw_label(self, x, y, text, size=7):
        """Draw small label text at position."""
        self.set_font("Helvetica", "", size)
        self.set_text_color(*GRAY)
        self.set_xy(x, y)
        self.cell(0, 4, text)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ================================================================
    # COVER PAGE
    # ================================================================
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 15, "Phantom Tracker", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Real-Time Multi-Object Tracking with Occlusion Handling,", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Re-Identification, and Multi-Camera Handoff", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_draw_color(*LIGHT_BLUE)
    pdf.set_line_width(0.8)
    pdf.line(55, pdf.get_y(), 155, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*MED_GRAY)
    pdf.cell(0, 7, "Midterm Progress Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Computer Vision Course", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "SP Jain School of Global Management", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "Team Members", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MED_GRAY)
    for name, role in [
        ("Dhruvish Shah", "Multi-Object Tracking"),
        ("Divyansh Singh Maiwar", "Object Detection"),
        ("Dharmik Kothari", "Re-Identification"),
        ("Agastya Shetty", "Visualization & Analytics"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, name, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 5, role, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*MED_GRAY)
        pdf.ln(2)

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "Date: March 4, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Repository: github.com/Dhruvish98/phantom-tracker", align="C", new_x="LMARGIN", new_y="NEXT")

    # ================================================================
    # TABLE OF CONTENTS
    # ================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    toc = [
        ("1", "Problem Statement", "3", True),
        ("2", "Objective and Project Scope", "3", True),
        ("3", "System Architecture", "4", True),
        ("3.1", "High-Level Pipeline Architecture", "4", False),
        ("3.2", "Data Flow Diagram", "5", False),
        ("3.3", "Occlusion State Machine", "5", False),
        ("3.4", "Technical Stack", "6", False),
        ("4", "Module Design (Technical Details)", "6", True),
        ("4.1", "Detection Module (YOLOv11)", "6", False),
        ("4.2", "Tracking Module (BoT-SORT)", "7", False),
        ("4.3", "Re-Identification Module (OSNet)", "8", False),
        ("4.4", "Visualization Module (OpenCV)", "9", False),
        ("5", "Integration and Pipeline Orchestration", "10", True),
        ("6", "Member Contributions", "11", True),
        ("7", "Current Progress (Midterm)", "12", True),
        ("8", "Final Product Scope", "13", True),
        ("9", "Conclusion", "14", True),
        ("", "References", "14", True),
    ]

    left_margin = 15
    right_margin = 195
    page_num_w = 10

    for num, title, pg, is_main in toc:
        indent = 0 if is_main else 10
        x_start = left_margin + indent
        bold = "B" if is_main else ""

        # Number column
        pdf.set_xy(x_start, pdf.get_y())
        pdf.set_font("Helvetica", bold, 10)
        pdf.set_text_color(*DARK)
        if num:
            num_str = f"{num}."
            num_w = pdf.get_string_width(num_str) + 3
            pdf.cell(num_w, 7, num_str)
        else:
            num_w = 0

        # Title
        title_x = x_start + num_w
        pdf.set_font("Helvetica", bold, 10)
        title_w = pdf.get_string_width(title)
        pdf.cell(title_w + 2, 7, title)

        # Dots
        dots_start = title_x + title_w + 4
        dots_end = right_margin - page_num_w
        pdf.set_font("Helvetica", "", 10)
        dot_w = pdf.get_string_width(". ")
        if dots_end > dots_start:
            n_dots = int((dots_end - dots_start) / dot_w)
            dot_str = " ." * max(3, n_dots)
            pdf.set_text_color(*GRAY)
            pdf.set_xy(dots_start, pdf.get_y())
            pdf.cell(dots_end - dots_start, 7, dot_str)

        # Page number (right-aligned)
        pdf.set_font("Helvetica", bold, 10)
        pdf.set_text_color(*DARK)
        pdf.set_xy(right_margin - page_num_w, pdf.get_y())
        pdf.cell(page_num_w, 7, pg, align="R")
        pdf.ln(7)

    # ================================================================
    # 1. PROBLEM STATEMENT
    # ================================================================
    pdf.add_page()
    pdf.section_title("1", "Problem Statement")
    pdf.body(
        "Multi-Object Tracking (MOT) is a fundamental computer vision task that involves detecting "
        "multiple objects in a video stream and maintaining their unique identities across consecutive "
        "frames. While significant progress has been made in detection accuracy (YOLO, Faster R-CNN) "
        "and single-object tracking (KCF, SiamFC), the multi-object setting introduces three "
        "compounding difficulties that remain actively researched:"
    )
    pdf.sub_sub_title("1.1  Occlusion and Identity Fragmentation")
    pdf.body(
        "When an object becomes temporarily hidden behind another object, a pillar, or leaves the "
        "camera's field of view, most tracking systems lose its identity. Upon reappearance, the "
        "system assigns a new track ID, fragmenting the object's history. In a retail analytics "
        "scenario, this means a single customer may be counted as 3-4 different people during a "
        "10-minute shopping trip, rendering dwell-time and path analytics meaningless."
    )
    pdf.sub_sub_title("1.2  Cross-Camera Identity Persistence")
    pdf.body(
        "In multi-camera surveillance deployments (malls, airports, campuses), the same person "
        "appears in different cameras with drastically different viewpoints, lighting, and scales. "
        "Maintaining a globally consistent identity across cameras requires robust appearance-based "
        "re-identification that generalizes across domains, a problem known as Cross-Camera Re-ID."
    )
    pdf.sub_sub_title("1.3  Real-Time Constraint")
    pdf.body(
        "A production tracking system must process detection, association, re-identification, and "
        "visualization within the frame budget (33ms at 30fps). Each additional module (Re-ID "
        "embedding extraction at ~100ms, Kalman prediction, Hungarian matching) adds latency. "
        "Achieving real-time performance requires careful GPU pipelining, model optimization "
        "(TensorRT, FP16), and selective computation (e.g., extracting Re-ID embeddings only "
        "every Nth frame)."
    )

    # ================================================================
    # 2. OBJECTIVE AND SCOPE
    # ================================================================
    pdf.section_title("2", "Objective and Project Scope")
    pdf.sub_title("2.1  Midterm Objective")
    pdf.body(
        "Build a complete, modular, real-time single-camera multi-object tracking pipeline that "
        "detects, tracks, re-identifies, and visualizes objects with persistent identities through "
        "occlusion events."
    )
    pdf.sub_title("2.2  Final Product Scope")
    pdf.body(
        "The final deliverable extends the midterm system into a multi-camera, production-grade "
        "tracking and analytics platform. The complete scope includes:"
    )
    pdf.numbered(1, "Single-Camera MOT: YOLOv11 detection + BoT-SORT tracking + occlusion state machine with ghost prediction. [Midterm: DONE]")
    pdf.numbered(2, "Single-Camera Re-ID: OSNet appearance matching to recover lost track identities within the same camera feed. [Midterm: DONE]")
    pdf.numbered(3, "Multi-Camera Re-ID: Extend the Re-ID system to maintain globally consistent identities across 2+ camera feeds. This involves a shared global feature gallery, camera-invariant embedding normalization, and a handoff protocol that transfers track state between camera-specific trackers. [Final]")
    pdf.numbered(4, "Open-Vocabulary Detection: Integrate Grounding DINO for natural language-based object queries (e.g., 'person with red backpack'), enabling flexible, prompt-driven tracking initialization without retraining. [Final]")
    pdf.numbered(5, "LSTM Motion Prediction: Replace linear velocity extrapolation with a learned sequence model for non-linear trajectory prediction during occlusion (curved paths, acceleration/deceleration). [Final]")
    pdf.numbered(6, "TensorRT Optimization: Export YOLO and OSNet to TensorRT FP16 engines for 2-4x inference speedup, targeting 20+ FPS for the full pipeline on a single GPU. [Final]")
    pdf.numbered(7, "Comprehensive Visualization and Analytics: Real-time dashboard with heatmaps, zone-based analytics, dwell-time histograms, trajectory clustering, and Re-ID event logs. Web dashboard for remote monitoring. [Final]")
    pdf.numbered(8, "Ablation Study: Quantitative evaluation using MOT metrics (MOTA, IDF1, HOTA) across YOLO model sizes, Re-ID thresholds, and tracker configurations on standard benchmarks (MOT17, MOT20). [Final]")

    pdf.note_box("The multi-camera Re-ID capability is the central differentiator of the final product, enabling Phantom Tracker to scale beyond single-camera deployments.")

    # ================================================================
    # 3. SYSTEM ARCHITECTURE
    # ================================================================
    pdf.add_page()
    pdf.section_title("3", "System Architecture")

    # ── 3.1 Pipeline Diagram ──
    pdf.sub_title("3.1  High-Level Pipeline Architecture")
    pdf.body("The system processes each video frame through a four-stage sequential pipeline:")

    # Draw pipeline diagram
    diag_y = pdf.get_y() + 2
    bw, bh = 35, 18  # box width, height
    gap = 8
    start_x = 15

    stages = [
        ("Stage 1", "DETECTION", "YOLOv11s", BOX_BLUE),
        ("Stage 2", "TRACKING", "BoT-SORT", BOX_GREEN),
        ("Stage 3", "RE-ID", "OSNet x1.0", BOX_ORANGE),
        ("Stage 4", "VISUALIZE", "OpenCV", (220, 210, 240)),
    ]

    # Input box
    pdf.draw_box(start_x, diag_y, 25, bh, "Video", LIGHT_GRAY, "Frame")
    x = start_x + 25
    pdf.draw_arrow(x, diag_y + bh / 2, x + gap, diag_y + bh / 2)
    x += gap

    for i, (stage, name, sub, color) in enumerate(stages):
        pdf.draw_box(x, diag_y, bw, bh, name, color, sub)
        if i < len(stages) - 1:
            pdf.draw_arrow(x + bw, diag_y + bh / 2, x + bw + gap, diag_y + bh / 2)
        x += bw + gap

    # Output
    pdf.draw_arrow(x - gap, diag_y + bh / 2, x, diag_y + bh / 2)
    pdf.draw_box(x, diag_y, 25, bh, "Output", LIGHT_GRAY, "Display")

    # Stage labels above
    x = start_x + 25 + gap
    for i, (stage, _, _, _) in enumerate(stages):
        pdf.draw_label(x + 5, diag_y - 5, stage, 7)
        x += bw + gap

    # State machine feedback arrow (Re-ID -> Tracker)
    reid_x = start_x + 25 + gap + (bw + gap) * 2  # Re-ID box x
    track_x = start_x + 25 + gap + (bw + gap)  # Tracker box x
    feedback_y = diag_y + bh + 3
    pdf.set_draw_color(180, 80, 80)
    pdf.set_line_width(0.4)
    # Draw path: down from Re-ID, left to Tracker, up to Tracker
    pdf.line(reid_x + bw / 2, diag_y + bh, reid_x + bw / 2, feedback_y)
    pdf.line(reid_x + bw / 2, feedback_y, track_x + bw / 2, feedback_y)
    pdf.line(track_x + bw / 2, feedback_y, track_x + bw / 2, diag_y + bh)
    # Arrowhead
    pdf.line(track_x + bw / 2, diag_y + bh, track_x + bw / 2 - 2, diag_y + bh + 2)
    pdf.line(track_x + bw / 2, diag_y + bh, track_x + bw / 2 + 2, diag_y + bh + 2)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(180, 80, 80)
    pdf.set_xy(track_x + bw, feedback_y - 4)
    pdf.cell(30, 4, "Re-ID match feedback")

    pdf.set_y(feedback_y + 8)

    pdf.body(
        "Stage 1 - Detection: Each raw BGR frame is passed to YOLOv11s, which performs single-shot "
        "multi-scale object detection. The model outputs bounding boxes [x1, y1, x2, y2], confidence "
        "scores, and COCO class IDs. An optional class filter restricts output to specified categories "
        "(e.g., only 'person'). Output is packaged as a FrameDetections dataclass.\n\n"
        "Stage 2 - Tracking: The FrameDetections array is converted to an (N, 6) numpy matrix "
        "[x1, y1, x2, y2, conf, cls_id] and passed to BoT-SORT via the boxmot library along with "
        "the raw frame (for internal ReID). BoT-SORT performs Kalman-based motion prediction, "
        "Hungarian assignment for IoU + appearance association, and outputs an (M, 8) result matrix "
        "with persistent track IDs. Our custom state machine then manages tracks not present in "
        "boxmot's output through OCCLUDED and LOST states.\n\n"
        "Stage 3 - Re-Identification: When LOST tracks exist alongside unmatched detections, the "
        "Re-ID module crops each detection from the frame, passes it through OSNet to extract a "
        "512-dimensional L2-normalized embedding, and computes temporally-weighted cosine similarity "
        "against each lost track's appearance gallery. Matches above the confidence threshold (0.6) "
        "are sent back to the tracker to reactivate tracks with their original IDs.\n\n"
        "Stage 4 - Visualization: The annotated frame is rendered with color-coded bounding boxes, "
        "trajectory trails, ghost outlines for occluded tracks, predicted future paths, speed "
        "indicators, a spatial heatmap, and an analytics dashboard."
    )

    # ── 3.2 Data Flow Diagram ──
    pdf.add_page()
    pdf.sub_title("3.2  Data Flow Between Modules")

    diag_y = pdf.get_y() + 2
    col_w = 42
    row_h = 16

    # Module boxes (top row)
    modules = [
        ("Detector", "detection/\ndetector.py", BOX_BLUE),
        ("Tracker", "tracking/\ntracker.py", BOX_GREEN),
        ("ReIdentifier", "reid/\nreidentifier.py", BOX_ORANGE),
        ("Visualizer", "visualization/\nvisualizer.py", (220, 210, 240)),
    ]

    for i, (name, file, color) in enumerate(modules):
        x = 12 + i * (col_w + 6)
        pdf.draw_box(x, diag_y, col_w, row_h, name, color)

    # Data flow arrows with labels
    arr_y = diag_y + row_h / 2
    flows = [
        (12 + col_w, 12 + col_w + 6, "FrameDetections\n(N,6) array"),
        (12 + (col_w + 6) + col_w, 12 + (col_w + 6) * 2, "lost_tracks\n[Track]"),
        (12 + (col_w + 6) * 2 + col_w, 12 + (col_w + 6) * 3, "FrameState\nAnalytics"),
    ]
    for x1, x2, label in flows:
        pdf.draw_arrow(x1, arr_y, x2, arr_y)

    # Central data store
    store_y = diag_y + row_h + 12
    pdf.draw_box(50, store_y, 100, 14, "core/interfaces.py", (240, 240, 240), "Track, Detection, FrameState, ReIDResult, AnalyticsSnapshot, PipelineConfig")

    # Arrows down to data store
    for i in range(4):
        cx = 12 + i * (col_w + 6) + col_w / 2
        pdf.set_draw_color(160, 160, 160)
        pdf.set_line_width(0.3)
        pdf.dashed_line(cx, diag_y + row_h, cx, store_y, dash_length=2, space_length=1.5)

    pdf.set_y(store_y + 20)

    # ── 3.3 State Machine Diagram ──
    pdf.sub_title("3.3  Occlusion State Machine")
    pdf.body(
        "Each tracked object is maintained independently from boxmot's internal state using a "
        "dual-ledger architecture. boxmot only reports currently visible tracks; our state machine "
        "detects disappearances by comparing boxmot's output IDs against our known track dictionary "
        "each frame, then transitions unmatched tracks through the following states:"
    )

    # Draw state machine diagram
    sm_y = pdf.get_y() + 2
    box_w, box_h = 32, 14

    states = [
        (25, sm_y, "ACTIVE", BOX_GREEN),
        (75, sm_y, "OCCLUDED", BOX_ORANGE),
        (128, sm_y, "LOST", BOX_RED),
        (175, sm_y, "DELETED", LIGHT_GRAY),
    ]

    for x, y, label, color in states:
        pdf.draw_box(x, y, box_w, box_h, label, color)

    # Transitions
    pdf.draw_arrow(25 + box_w, sm_y + box_h / 2, 75, sm_y + box_h / 2)
    pdf.draw_label(42, sm_y - 4, "3 frames", 6)
    pdf.draw_label(42, sm_y + 0, "unseen", 6)

    pdf.draw_arrow(75 + box_w, sm_y + box_h / 2, 128, sm_y + box_h / 2)
    pdf.draw_label(95, sm_y - 4, "30 frames", 6)
    pdf.draw_label(95, sm_y + 0, "unseen", 6)

    pdf.draw_arrow(128 + box_w, sm_y + box_h / 2, 175, sm_y + box_h / 2)
    pdf.draw_label(147, sm_y - 4, "300 frames", 6)
    pdf.draw_label(147, sm_y + 0, "unseen", 6)

    # Re-ID recovery arrow (LOST -> ACTIVE)
    recovery_y = sm_y + box_h + 5
    pdf.set_draw_color(0, 150, 0)
    pdf.set_line_width(0.5)
    pdf.line(128 + box_w / 2, sm_y + box_h, 128 + box_w / 2, recovery_y)
    pdf.line(128 + box_w / 2, recovery_y, 25 + box_w / 2, recovery_y)
    pdf.line(25 + box_w / 2, recovery_y, 25 + box_w / 2, sm_y + box_h)
    # Arrowhead
    pdf.line(25 + box_w / 2, sm_y + box_h, 25 + box_w / 2 - 2, sm_y + box_h + 2)
    pdf.line(25 + box_w / 2, sm_y + box_h, 25 + box_w / 2 + 2, sm_y + box_h + 2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(0, 130, 0)
    pdf.set_xy(60, recovery_y - 4)
    pdf.cell(60, 4, "Re-ID match (similarity > 0.6)")

    # Re-appear arrow (OCCLUDED -> ACTIVE)
    reappear_y = sm_y - 6
    pdf.set_draw_color(0, 100, 200)
    pdf.set_line_width(0.4)
    pdf.line(75 + box_w / 2, sm_y, 75 + box_w / 2, reappear_y)
    pdf.line(75 + box_w / 2, reappear_y, 25 + box_w / 2, reappear_y)
    pdf.line(25 + box_w / 2, reappear_y, 25 + box_w / 2, sm_y)
    pdf.line(25 + box_w / 2, sm_y, 25 + box_w / 2 - 2, sm_y - 2)
    pdf.line(25 + box_w / 2, sm_y, 25 + box_w / 2 + 2, sm_y - 2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(0, 80, 180)
    pdf.set_xy(38, reappear_y - 4)
    pdf.cell(45, 4, "boxmot re-detects object")

    pdf.set_y(recovery_y + 8)

    pdf.body(
        "ACTIVE: Object is visible. Updated every frame with new bbox, EMA velocity (alpha=0.3), "
        "trajectory history, heatmap contribution, and appearance embedding (every 5th frame).\n\n"
        "OCCLUDED: Object unseen for 3+ frames but < max_occlusion_frames (30). Position is "
        "linearly extrapolated using smoothed velocity, clamped to frame boundaries. A ghost "
        "outline is rendered at the predicted position. If boxmot re-detects the object, it "
        "returns to ACTIVE.\n\n"
        "LOST: Object unseen for 30+ frames. Extrapolation stops. The Re-ID module actively "
        "compares new detection embeddings against this track's appearance gallery. A confident "
        "match (cosine similarity > 0.6) reactivates the track to ACTIVE with its original ID.\n\n"
        "DELETED: Object unseen for 300+ frames (10 seconds at 30fps). Track is removed from the "
        "active dictionary and archived in deleted_tracks. Entry/exit counters are updated."
    )

    # ── 3.4 Technical Stack ──
    pdf.sub_title("3.4  Technical Stack")
    pdf.kv("Language", "Python 3.10")
    pdf.kv("Detection", "YOLOv11s (Ultralytics) - single-shot multi-scale detector, 72.8 FPS on RTX 2050")
    pdf.kv("Tracking", "BoT-SORT via boxmot v16.0.10 - Kalman filter + Hungarian algorithm + deep appearance features")
    pdf.kv("Re-ID", "OSNet x1.0 via torchreid - lightweight CNN producing 512-d L2-normalized embeddings (2.2M params)")
    pdf.kv("Visualization", "OpenCV 4.x with custom multi-layer rendering engine (800+ lines)")
    pdf.kv("GPU", "NVIDIA GeForce RTX 2050 (4 GB VRAM), CUDA 13.0, Compute 8.6")
    pdf.kv("Framework", "PyTorch 2.10.0+cu128")
    pdf.kv("Tracker Library", "boxmot 16.0.10 (standalone, not ultralytics built-in)")
    pdf.kv("Re-ID Library", "torchreid (PyPI) with OSNet pretrained on ImageNet")

    # ================================================================
    # 4. MODULE DESIGN
    # ================================================================
    pdf.add_page()
    pdf.section_title("4", "Module Design (Technical Details)")

    # ── 4.1 Detection ──
    pdf.sub_title("4.1  Detection Module - YOLOv11")
    pdf.kv("File", "detection/detector.py")
    pdf.kv("Owner", "Divyansh Singh Maiwar")
    pdf.kv("Model", "YOLOv11s (9.4M parameters, 21.5 GFLOPs)")
    pdf.ln(2)

    pdf.sub_sub_title("Architecture")
    pdf.body(
        "YOLOv11 (You Only Look Once, version 11) is a single-stage anchor-free object detector "
        "from Ultralytics. It processes the input image through a CSPDarknet53 backbone for "
        "multi-scale feature extraction, a PAFPN (Path Aggregation Feature Pyramid Network) neck "
        "for feature fusion across scales, and decoupled detection heads that independently predict "
        "bounding box regression and class probabilities at three scales (P3/8, P4/16, P5/32)."
    )
    pdf.body(
        "We use the yolo11s variant (small), which provides the best speed-accuracy tradeoff for "
        "our 720p video streams. On the RTX 2050, YOLO inference alone runs at 72.8 FPS "
        "(13.7ms/frame), well within the real-time budget."
    )

    pdf.sub_sub_title("Implementation Details")
    pdf.bullet("Automatic device selection: detects CUDA availability via torch.cuda.is_available() and torch.cuda.device_count(), falls back to CPU transparently.")
    pdf.bullet("Class filtering: YOLO's native 'classes' parameter restricts inference output at the model level (e.g., classes=[0] for COCO 'person'), adding zero computational overhead compared to post-hoc filtering.")
    pdf.bullet("FP16 half-precision: When --half flag is set and GPU is available, inference runs in float16, reducing memory bandwidth and improving throughput by ~30% on Turing+ architectures.")
    pdf.bullet("Output format: Each detection is packaged as a Detection dataclass with bbox (np.ndarray [x1,y1,x2,y2]), confidence (float), class_name (str), and class_id (int).")

    pdf.sub_sub_title("Benchmark Results")
    pdf.code_line("yolo11n.pt:  70.5 FPS  (mean 14.2ms, std 4.3ms)  - RTX 2050")
    pdf.code_line("yolo11s.pt:  72.8 FPS  (mean 13.7ms, std 3.9ms)  - RTX 2050")
    pdf.code_line("yolo11m.pt:  ~45 FPS   (estimated)                - RTX 2050")

    # ── 4.2 Tracking ──
    pdf.add_page()
    pdf.sub_title("4.2  Tracking Module - BoT-SORT")
    pdf.kv("File", "tracking/tracker.py")
    pdf.kv("Owner", "Dhruvish Shah")
    pdf.kv("Algorithm", "BoT-SORT (Bag of Tricks for SORT) via boxmot v16")
    pdf.ln(2)

    pdf.sub_sub_title("Why BoT-SORT?")
    pdf.body(
        "BoT-SORT combines the strengths of three prior trackers:\n\n"
        "- SORT (Simple Online Realtime Tracker): Kalman filter for motion prediction + Hungarian "
        "algorithm for IoU-based assignment. Fast but fragile under occlusion.\n\n"
        "- DeepSORT: Adds a deep appearance descriptor (128-d Re-ID embedding) for association. "
        "Better identity persistence but slower and less robust to low-confidence detections.\n\n"
        "- ByteTrack: Two-pass association strategy that recovers low-confidence detections in a "
        "second matching round, significantly reducing missed tracks.\n\n"
        "BoT-SORT integrates all three techniques and adds Camera Motion Compensation (CMC) via "
        "ECC alignment for handling camera shake. On MOT17, BoT-SORT achieves 80.5 MOTA and "
        "80.2 IDF1, outperforming both DeepSORT and ByteTrack."
    )

    pdf.sub_sub_title("BoT-SORT Internal Pipeline")
    pdf.numbered(1, "Kalman Prediction: Predict each track's next position using a constant-velocity Kalman filter with state vector [cx, cy, aspect_ratio, height, vx, vy, va, vh].")
    pdf.numbered(2, "Camera Motion Compensation: Estimate inter-frame camera motion via ECC (Enhanced Correlation Coefficient) and warp predicted positions accordingly.")
    pdf.numbered(3, "First Association (High Confidence): Match high-confidence detections (conf > 0.45) to predicted tracks using a weighted cost matrix: cost = lambda * IoU_distance + (1 - lambda) * appearance_distance.")
    pdf.numbered(4, "Second Association (Low Confidence): Match remaining low-confidence detections (0.1 < conf < 0.45) to unmatched tracks using IoU only (ByteTrack strategy).")
    pdf.numbered(5, "Track Initialization: Unmatched high-confidence detections (conf > 0.5) initialize new tracks.")
    pdf.numbered(6, "Track Deletion: Tracks unmatched for track_buffer frames (90) are removed from boxmot's internal state.")

    pdf.sub_sub_title("Custom State Machine (Dual-Ledger Architecture)")
    pdf.body(
        "boxmot only outputs currently active tracks. To support occlusion visualization and Re-ID "
        "integration, we maintain a parallel track dictionary (self.tracks) that persists across "
        "frames independently. Each frame, we compare boxmot's output IDs against our dictionary "
        "to detect which tracks have disappeared, then transition them through the "
        "ACTIVE -> OCCLUDED -> LOST -> DELETED state machine."
    )

    pdf.sub_sub_title("EMA Velocity and Trajectory Prediction")
    pdf.body(
        "For each active track, we compute frame-to-frame displacement of the bounding box center "
        "and apply Exponential Moving Average smoothing:\n\n"
        "v_smoothed = alpha * v_raw + (1 - alpha) * v_previous,  where alpha = 0.3\n\n"
        "This smoothed velocity serves two purposes: (1) linear extrapolation of occluded track "
        "positions (ghost bbox), and (2) predicted future trajectory visualization for active tracks "
        "(15-frame lookahead). Extrapolated positions are clamped to frame boundaries to prevent "
        "ghost bboxes from drifting off-screen."
    )

    pdf.sub_sub_title("Tuned Hyperparameters")
    pdf.code_line("track_high_thresh   = 0.45   # primary association confidence gate")
    pdf.code_line("track_low_thresh    = 0.10   # secondary (ByteTrack) confidence gate")
    pdf.code_line("new_track_thresh    = 0.50   # minimum confidence to init new track")
    pdf.code_line("track_buffer        = 90     # frames before boxmot drops a track (3s@30fps)")
    pdf.code_line("match_thresh        = 0.85   # IoU matching threshold")
    pdf.code_line("proximity_thresh    = 0.55   # spatial gate for appearance matching")
    pdf.code_line("appearance_thresh   = 0.20   # appearance similarity gate (lower = lenient)")
    pdf.code_line("botsort_frame_rate  = 30     # assumed FPS for Kalman motion model")

    # ── 4.3 Re-ID ──
    pdf.add_page()
    pdf.sub_title("4.3  Re-Identification Module - OSNet")
    pdf.kv("File", "reid/reidentifier.py")
    pdf.kv("Owner", "Dharmik Kothari")
    pdf.kv("Model", "OSNet x1.0 (2.2M params, 978M FLOPs)")
    pdf.ln(2)

    pdf.sub_sub_title("OSNet Architecture")
    pdf.body(
        "OSNet (Omni-Scale Network) is a lightweight CNN designed specifically for person "
        "re-identification. Its key innovation is the Omni-Scale Feature Learning block, which "
        "uses multiple convolutional streams at different receptive field sizes (1x1, 3x3, 5x5 "
        "via factorized depthwise convolutions) and dynamically aggregates them using channel "
        "attention gates. This allows the network to learn features at multiple spatial scales "
        "simultaneously, capturing both fine-grained texture (clothing pattern) and coarse-grained "
        "structure (body shape)."
    )
    pdf.body(
        "The model produces a 512-dimensional L2-normalized embedding vector for each person crop. "
        "Two embeddings from the same person should have high cosine similarity (> 0.6), while "
        "embeddings from different people should have low similarity (< 0.4)."
    )

    pdf.sub_sub_title("Feature Bank with Temporal Decay")
    pdf.body(
        "Each track maintains an appearance gallery: a list of (embedding, timestamp) pairs, "
        "capped at 50 entries. When computing similarity between a query embedding and a track's "
        "gallery, we apply temporal decay weighting:"
    )
    pdf.code_line("w_i = decay_rate ^ (N - 1 - i)    where i=0 is oldest, decay_rate=0.95")
    pdf.code_line("similarity = sum(w_i * cos_sim(query, gallery[i])) / sum(w_i)")
    pdf.body(
        "This ensures recent appearances are weighted more heavily (~14 entries back = half weight), "
        "making the system robust to gradual appearance changes (e.g., person removes jacket)."
    )

    pdf.sub_sub_title("Matching Pipeline")
    pdf.numbered(1, "For each unmatched detection, extract a 512-d embedding from the cropped region (resized to 128x256 for OSNet input).")
    pdf.numbered(2, "Compare against all LOST tracks' galleries using temporally-weighted cosine similarity.")
    pdf.numbered(3, "Select the best match. If similarity > reid_confidence_threshold (0.6), emit a ReIDMatch with is_confident=True.")
    pdf.numbered(4, "Prevent double-assignment: matched track IDs are added to a claimed_ids set and excluded from subsequent comparisons within the same frame.")
    pdf.numbered(5, "Send ReIDResult back to tracker.apply_reid_results(), which reactivates matched tracks to ACTIVE state.")

    pdf.sub_sub_title("Gallery Update Strategy")
    pdf.body(
        "To populate appearance galleries, the main pipeline calls reidentifier.update_gallery() "
        "for every active track every 5th frame. This balances embedding quality (active tracks "
        "have good bounding boxes) against computational cost (~100ms per OSNet forward pass on CPU)."
    )

    pdf.sub_sub_title("Multi-Camera Re-ID (Final Scope)")
    pdf.body(
        "For the final product, the Re-ID module will be extended with: (1) a global feature gallery "
        "shared across all camera-specific trackers, (2) camera-invariant embedding normalization "
        "using domain adaptation (e.g., camera-aware batch normalization), and (3) a handoff protocol "
        "that transfers a track's full state (appearance gallery, velocity, trajectory history) from "
        "one camera's tracker to another when a cross-camera match is detected."
    )

    # ── 4.4 Visualization ──
    pdf.add_page()
    pdf.sub_title("4.4  Visualization Module - OpenCV Overlay Engine")
    pdf.kv("File", "visualization/visualizer.py (800+ lines)")
    pdf.kv("Owner", "Agastya Shetty")
    pdf.ln(2)

    pdf.sub_sub_title("Rendering Layers")
    pdf.body("The visualizer composes multiple overlay layers onto each frame:")
    pdf.numbered(1, "Bounding Boxes: Color-coded per track ID (persistent color assignment using hash-based color generation). Solid border for ACTIVE, dashed border for OCCLUDED.")
    pdf.numbered(2, "Track Labels: 'ID:N class_name' with semi-transparent background for readability.")
    pdf.numbered(3, "Trajectory Trails: Last 60 positions rendered as a polyline with distance-based fade (newer positions are more opaque). Trail color matches track color.")
    pdf.numbered(4, "Ghost Outlines: For OCCLUDED tracks, a filled semi-transparent rectangle at the extrapolated position with a ghost icon (circle + eyes) and an expanding uncertainty ring (radius grows with frames_since_seen).")
    pdf.numbered(5, "Predicted Path: For ACTIVE tracks, the 15-frame predicted trajectory rendered as a line with uncertainty cones (expanding circles) at each predicted position.")
    pdf.numbered(6, "Speed Indicators: Per-track instantaneous speed displayed near the bounding box.")
    pdf.numbered(7, "Re-ID Notifications: Temporary green overlay when a track is re-identified, showing the matched track ID and similarity score.")

    pdf.sub_sub_title("Analytics Dashboard")
    pdf.body(
        "The dashboard (toggled with 'd') renders a semi-transparent panel on the right side of "
        "the frame containing:\n\n"
        "- Current object count (ACTIVE + OCCLUDED tracks)\n"
        "- Total entries and exits (cumulative)\n"
        "- Per-track speeds (pixels/second) with track ID labels\n"
        "- Per-track dwell times (seconds since first appearance)\n"
        "- Recent Re-ID events (last 3 matches with similarity percentages)"
    )

    pdf.sub_sub_title("Heatmap")
    pdf.body(
        "The spatial heatmap (toggled with 'h') accumulates object center positions into a "
        "64x48 grid (config.heatmap_resolution), which is upsampled and alpha-blended onto the "
        "frame using a JET colormap. Areas with high object density appear as warm (red) zones."
    )

    # ================================================================
    # 5. INTEGRATION
    # ================================================================
    pdf.add_page()
    pdf.section_title("5", "Integration and Pipeline Orchestration")
    pdf.kv("File", "main.py (PhantomTracker class)")
    pdf.kv("Shared Contracts", "core/interfaces.py")
    pdf.ln(2)

    pdf.sub_sub_title("Pipeline Orchestration (process_frame)")
    pdf.body(
        "The PhantomTracker class in main.py initializes all four modules and orchestrates the "
        "per-frame pipeline. The process_frame() method executes the following sequence for each "
        "video frame:"
    )
    pdf.code_line("1. state.detections = detector.detect(frame, frame_id, timestamp)")
    pdf.code_line("2. active, occluded, lost = tracker.update(detections, frame)")
    pdf.code_line("3. for track in active_tracks:  # every 5th frame")
    pdf.code_line("       reidentifier.update_gallery(track, frame)")
    pdf.code_line("4. if lost_tracks and detections:")
    pdf.code_line("       reid_results = reidentifier.match(dets, lost, active_ids, frame)")
    pdf.code_line("       tracker.apply_reid_results(reid_results)")
    pdf.code_line("5. analytics = tracker.get_analytics(frame_id, timestamp)")
    pdf.code_line("6. output = visualizer.render(frame, state, analytics, fps)")

    pdf.sub_sub_title("Shared Data Contracts (core/interfaces.py)")
    pdf.body(
        "All inter-module communication uses strictly typed dataclasses defined in "
        "core/interfaces.py. This contract-first design ensures modules can be developed and tested "
        "independently. Key data structures:"
    )
    pdf.bullet("Detection: bbox (ndarray), confidence (float), class_name (str), class_id (int), embedding (optional ndarray)")
    pdf.bullet("FrameDetections: frame_id, timestamp, list[Detection], source, inference_time_ms")
    pdf.bullet("Track: track_id, state (enum), bbox, confidence, class_name, color, velocity, predicted_position, trajectory_history, appearance_gallery, speed metrics")
    pdf.bullet("ReIDMatch: new_detection_idx, matched_track_id, similarity_score, is_confident")
    pdf.bullet("ReIDResult: frame_id, list[ReIDMatch], unmatched_detection_indices")
    pdf.bullet("AnalyticsSnapshot: frame_id, track_speeds, track_dwell_times, heatmap, entry/exit counts, reid_events")
    pdf.bullet("PipelineConfig: 30+ configuration fields covering all modules (detection thresholds, tracking parameters, Re-ID settings, visualization toggles)")

    pdf.sub_sub_title("Configuration System")
    pdf.body(
        "PipelineConfig is a single dataclass with sensible defaults for all parameters. CLI "
        "arguments (--yolo-model, --confidence, --classes, --half, --input, --output) override "
        "defaults at runtime. All modules read from the same config instance, ensuring consistency."
    )

    # ================================================================
    # 6. MEMBER CONTRIBUTIONS
    # ================================================================
    pdf.add_page()
    pdf.section_title("6", "Member Contributions")

    pdf.sub_title("6.1  Divyansh Singh Maiwar - Detection Module")
    pdf.bullet("Implemented YOLOv11 inference pipeline with automatic GPU detection (CUDA:0 vs CPU) and device management.")
    pdf.bullet("Added class filtering via YOLO's native 'classes' parameter for zero-overhead selective detection.")
    pdf.bullet("Implemented FP16 half-precision inference support for GPU (--half flag).")
    pdf.bullet("Built benchmark utility for FPS profiling across YOLO model variants.")
    pdf.bullet("Standardized detection output format (Detection and FrameDetections dataclasses).")

    pdf.sub_title("6.2  Dhruvish Shah - Tracking Module")
    pdf.bullet("Implemented BoT-SORT multi-object tracking via boxmot with deep appearance features.")
    pdf.bullet("Designed the dual-ledger occlusion state machine (Active/Occluded/Lost/Deleted) operating independently on top of boxmot's output.")
    pdf.bullet("Implemented EMA-smoothed velocity estimation and linear trajectory prediction for occluded position extrapolation and future path visualization.")
    pdf.bullet("Tuned 8 BoT-SORT hyperparameters for indoor walking scenarios.")
    pdf.bullet("Built automatic CUDA/CPU device fallback and IoU-based fallback tracker.")
    pdf.bullet("Integrated Re-ID events pipeline and appearance gallery update wiring in the main pipeline.")
    pdf.bullet("Maintained shared interfaces (core/interfaces.py) and main pipeline orchestration (main.py).")

    pdf.sub_title("6.3  Dharmik Kothari - Re-Identification Module")
    pdf.bullet("Implemented OSNet feature extraction via torchreid producing 512-d L2-normalized embeddings.")
    pdf.bullet("Designed FeatureBank with per-track appearance gallery and temporal decay weighting (decay_rate=0.95).")
    pdf.bullet("Built cosine similarity matching pipeline with configurable confidence threshold.")
    pdf.bullet("Implemented on-the-fly embedding extraction from raw frame crops.")
    pdf.bullet("Added safe cropping with boundary clamping and minimum size validation.")
    pdf.bullet("Provided statistics API for ablation study metrics.")

    pdf.sub_title("6.4  Agastya Shetty - Visualization Module")
    pdf.bullet("Built 800+ line visualization engine with all overlay types: bounding boxes, trajectory trails, ghost outlines, predicted paths, speed indicators, Re-ID notifications.")
    pdf.bullet("Implemented real-time analytics dashboard with object counts, speeds, dwell times, and Re-ID events.")
    pdf.bullet("Built spatial heatmap overlay with JET colormap blending.")
    pdf.bullet("Added interactive keyboard controls for toggling individual features.")
    pdf.bullet("Created test suite and visualization showcase demo.")

    # ================================================================
    # 7. CURRENT PROGRESS
    # ================================================================
    pdf.add_page()
    pdf.section_title("7", "Current Progress (Midterm)")

    pdf.sub_title("7.1  Completed Milestones")
    pdf.bullet("Full end-to-end pipeline: Detection -> Tracking -> Re-ID -> Visualization running on GPU.")
    pdf.bullet("BoT-SORT tracking with persistent IDs across occlusion events, dual-ledger state machine.")
    pdf.bullet("OSNet-based Re-ID successfully re-identifying returning persons with 60-75% cosine similarity.")
    pdf.bullet("Real-time visualization with all overlays, analytics dashboard, and spatial heatmap.")
    pdf.bullet("GPU acceleration: YOLO at 72.8 FPS, full pipeline at 6-7 FPS on RTX 2050.")
    pdf.bullet("Tested on 6 diverse demo videos: subway pedestrians, slow walkers, busy urban street, dog park, Tokyo indoor subway, and night street scene.")
    pdf.bullet("Class filtering (--classes person) and FP16 inference (--half) for demo flexibility.")
    pdf.bullet("Automatic device detection with CPU fallback for portability.")

    pdf.sub_title("7.2  Performance Metrics")
    pdf.kv("YOLO Detection (GPU)", "72.8 FPS (yolo11s, 1280x720, RTX 2050)")
    pdf.kv("Full Pipeline (GPU)", "6-7 FPS (all 4 stages, Re-ID every 5th frame)")
    pdf.kv("Re-ID Similarity", "60-75% cosine similarity on confident matches")
    pdf.kv("Occlusion Tolerance", "3-second buffer (90 frames at 30fps)")
    pdf.kv("Track State Recovery", "LOST -> ACTIVE via Re-ID within 300 frames")
    pdf.kv("Detection Latency", "13.7ms/frame (yolo11s, GPU)")
    pdf.kv("Re-ID Latency", "~100ms/crop (OSNet forward pass, CPU path)")
    pdf.ln(2)

    pdf.sub_title("7.3  Known Limitations and Mitigations")
    pdf.bullet("Pipeline FPS bottleneck: Re-ID embedding extraction (~100ms/crop) dominates latency. Mitigation: extract only every 5th frame for gallery updates; plan TensorRT export for final.")
    pdf.bullet("Re-ID similarity ceiling: 60-75% range due to single-camera viewpoint changes and similar clothing. Mitigation: temporal decay weighting and gallery size tuning; plan camera-aware normalization for multi-camera.")
    pdf.bullet("Linear motion prediction: Does not handle curved or accelerating trajectories during occlusion. Mitigation: planned LSTM motion model for final submission.")
    pdf.bullet("Single-camera limitation: Current Re-ID only operates within one camera feed. Multi-camera handoff is the primary final project deliverable.")

    # ================================================================
    # 8. FINAL PRODUCT SCOPE
    # ================================================================
    pdf.add_page()
    pdf.section_title("8", "Final Product Scope")

    pdf.sub_title("8.1  Multi-Camera Re-Identification")
    pdf.body(
        "The centerpiece of the final product is cross-camera Re-ID. The architecture extends the "
        "current single-camera system with:"
    )
    pdf.bullet("Global Feature Gallery: A centralized appearance database shared across all camera-specific tracker instances. Each camera's tracker writes embeddings; the Re-ID module reads from all cameras when matching.")
    pdf.bullet("Camera-Invariant Normalization: Domain adaptation techniques (e.g., camera-aware batch normalization, histogram equalization of feature distributions) to reduce the domain gap between cameras with different viewpoints, lighting, and white balance.")
    pdf.bullet("Handoff Protocol: When a cross-camera Re-ID match is detected, the full track state (appearance gallery, velocity history, trajectory, dwell time) is transferred from the source camera's tracker to the destination camera's tracker, ensuring seamless identity continuity.")
    pdf.bullet("Topology-Aware Matching: Exploit known camera placement topology (e.g., 'camera A exit zone overlaps camera B entry zone') to reduce the search space and improve matching accuracy.")

    pdf.sub_title("8.2  Open-Vocabulary Detection (Grounding DINO)")
    pdf.body(
        "Grounding DINO enables natural language-driven detection. Instead of being limited to "
        "COCO's 80 classes, users can specify arbitrary text prompts like 'person with red backpack' "
        "or 'abandoned luggage'. The detection module will run Grounding DINO on-demand to initialize "
        "specific tracks, then hand off to YOLO for frame-to-frame tracking."
    )

    pdf.sub_title("8.3  LSTM Motion Prediction")
    pdf.body(
        "Replace the current linear velocity extrapolation with a trained LSTM sequence model. The "
        "LSTM takes the last N trajectory points as input and predicts the next K positions, handling "
        "curved paths, acceleration, deceleration, and common motion patterns (e.g., turning at "
        "corridors, stopping at displays)."
    )

    pdf.sub_title("8.4  TensorRT Optimization")
    pdf.body(
        "Export both YOLOv11 and OSNet models to TensorRT FP16 engines. Expected speedup: 2-4x "
        "for YOLO inference and 3-5x for OSNet embedding extraction. Target: 20+ FPS for the "
        "full pipeline on a single RTX 2050."
    )

    pdf.sub_title("8.5  Ablation Study")
    pdf.body(
        "Comprehensive quantitative evaluation using standard MOT metrics on MOT17 and MOT20 "
        "benchmarks. Variables: YOLO model size (n/s/m/l), Re-ID threshold (0.4/0.5/0.6/0.7), "
        "track buffer (30/60/90/120 frames), appearance threshold (0.1/0.2/0.3). Metrics: MOTA, "
        "IDF1, HOTA, ID switches, fragmentation rate."
    )

    # ================================================================
    # 9. CONCLUSION
    # ================================================================
    pdf.section_title("9", "Conclusion")
    pdf.body(
        "Phantom Tracker demonstrates a fully functional, modular multi-object tracking system that "
        "addresses the core challenge of identity persistence through occlusion. The midterm "
        "deliverable integrates state-of-the-art detection (YOLOv11), robust tracking (BoT-SORT "
        "with a custom occlusion state machine), deep appearance-based re-identification (OSNet "
        "with temporal-decay feature banks), and comprehensive real-time visualization."
    )
    pdf.body(
        "The modular architecture, with strictly typed data contracts between all four components, "
        "enables independent development and testing while ensuring seamless integration. The system "
        "runs on commodity GPU hardware (RTX 2050, 4GB VRAM) and includes automatic fallbacks for "
        "CPU-only environments."
    )
    pdf.body(
        "For the final submission, the focus shifts to multi-camera Re-ID (the central differentiator), "
        "open-vocabulary detection via Grounding DINO, learned motion prediction via LSTM, TensorRT "
        "optimization for real-time performance, and a comprehensive ablation study with standard "
        "MOT benchmarks. Together, these extensions will transform Phantom Tracker from a "
        "single-camera prototype into a multi-camera, production-grade tracking and analytics platform."
    )

    # ── References ──
    pdf.ln(4)
    pdf.section_title("", "References")
    refs = [
        "[1] Aharon, N. et al. 'BoT-SORT: Robust Associations Multi-Pedestrian Tracking.' arXiv:2206.14651, 2022.",
        "[2] Zhou, K. et al. 'Omni-Scale Feature Learning for Person Re-Identification.' ICCV, 2019.",
        "[3] Jocher, G. et al. 'Ultralytics YOLOv11.' https://github.com/ultralytics/ultralytics, 2024.",
        "[4] Zhang, Y. et al. 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box.' ECCV, 2022.",
        "[5] Wojke, N. et al. 'Simple Online and Realtime Tracking with a Deep Association Metric.' ICIP, 2017.",
        "[6] Liu, S. et al. 'Grounding DINO: Marrying DINO with Grounded Pre-Training.' ECCV, 2024.",
        "[7] boxmot library: https://github.com/mikel-brostrom/boxmot",
        "[8] torchreid library: https://github.com/KaiyangZhou/deep-person-reid",
    ]
    for ref in refs:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*DARK)
        pdf.multi_cell(0, 4.5, ref)
        pdf.ln(1)

    # Save
    pdf.output("docs/Phantom_Tracker_Midterm_Report.pdf")
    print("Report saved: docs/Phantom_Tracker_Midterm_Report.pdf")


if __name__ == "__main__":
    build()
