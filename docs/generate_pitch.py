"""Generate the Phantom Tracker pitch/presentation script as a PDF."""

from fpdf import FPDF


class PitchScript(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Phantom Tracker  - Presentation Script", align="R")
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

    def section(self, title, duration=""):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 70, 140)
        dur = f"  [{duration}]" if duration else ""
        self.cell(0, 10, f"{title}{dur}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 70, self.get_y())
        self.ln(4)

    def speak(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def stage_direction(self, text):
        self.set_font("Helvetica", "BI", 10)
        self.set_text_color(0, 120, 60)
        self.multi_cell(0, 5.5, f"[{text}]")
        self.ln(2)

    def key_point(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(15, 6, "")
        self.set_font("Helvetica", "B", 10.5)
        self.cell(5, 6, "-" + " ")
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def demo_command(self, cmd):
        self.set_font("Courier", "", 9)
        self.set_text_color(80, 80, 80)
        self.set_fill_color(240, 240, 240)
        self.cell(15, 6, "")
        self.cell(0, 7, f"  $ {cmd}  ", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10.5)
        self.ln(2)


def build():
    pdf = PitchScript()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER ──
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 15, "Phantom Tracker", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Midterm Presentation Script", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(0, 102, 204)
    pdf.set_line_width(0.8)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "Presenter: Dhruvish Parekh", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Date: March 4, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 7, "Estimated Duration: 12-15 minutes", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Structure: Explain (5 min) + Live Demo (7-8 min) + Q&A (2 min)", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── OPENING ──
    pdf.add_page()
    pdf.section("1. Opening  - The Problem", "~1.5 min")
    pdf.speak(
        "Good morning everyone. I'm Dhruvish, and today I'll be presenting our team's midterm project  - Phantom Tracker."
    )
    pdf.speak(
        "Let me start with a question. Think about any CCTV footage you've seen  - a mall, a subway station, "
        "an airport. Now imagine a person walks behind a pillar, disappears for just two seconds, and comes "
        "out the other side. To us, it's obviously the same person. But to a computer vision system? That person "
        "just vanished. And when they reappear, the system thinks it's a completely new person. Gives them a new ID. "
        "Loses all their tracking history."
    )
    pdf.speak(
        "This is the occlusion problem  - and it's one of the hardest challenges in multi-object tracking. "
        "It breaks identity persistence, it corrupts analytics like people counting and dwell time, and it "
        "makes surveillance systems fundamentally unreliable."
    )
    pdf.speak(
        "Phantom Tracker is our solution to this problem."
    )

    # ── PROJECT OVERVIEW ──
    pdf.section("2. What is Phantom Tracker?", "~2 min")
    pdf.speak(
        "Phantom Tracker is a real-time multi-object tracking system that does four things:"
    )
    pdf.key_point("It DETECTS objects in every video frame using YOLOv11  - that's the latest YOLO model, running at over 70 frames per second on our GPU.")
    pdf.key_point("It TRACKS those objects across frames using BoT-SORT  - which stands for Bag of Tricks for SORT. This handles the frame-to-frame association: matching this frame's detections to last frame's tracks using a combination of motion prediction, IoU matching, and appearance features.")
    pdf.key_point("It RE-IDENTIFIES objects that were lost. When someone walks out of frame and comes back 10 seconds later, our Re-ID module  - built on OSNet deep features  - recognizes them and restores their original identity.")
    pdf.key_point("And it VISUALIZES everything in real-time: bounding boxes, trajectory trails, ghost outlines showing where occluded objects are predicted to be, a heatmap, and a live analytics dashboard.")

    # ── ARCHITECTURE ──
    pdf.section("3. How It Works  - The Architecture", "~1.5 min")
    pdf.speak(
        "The system is a four-stage pipeline. Each frame flows through these stages sequentially:"
    )
    pdf.speak(
        "Stage 1 is Detection. Divyansh built the detection module using YOLOv11s. It takes a raw video frame "
        "and outputs bounding boxes with class labels and confidence scores. We added class filtering  - so for "
        "our demos, we can tell it to only detect people, or only cars, right at the YOLO level. On our RTX 2050 GPU, "
        "YOLO alone runs at 73 FPS."
    )
    pdf.speak(
        "Stage 2 is Tracking  - that's my module. I implemented BoT-SORT through the boxmot library. "
        "Internally, it uses a Kalman filter for motion prediction and the Hungarian algorithm for data "
        "association. But boxmot only tells us which tracks are currently visible. So I built a custom "
        "state machine on top of it that tracks four states: Active, Occluded, Lost, and Deleted. "
        "When an object disappears for 3 frames, it goes to Occluded  - we extrapolate its position using "
        "EMA-smoothed velocity. After 30 frames unseen, it goes to Lost  - that's when Re-ID kicks in. "
        "After 300 frames, it's Deleted."
    )
    pdf.speak(
        "Stage 3 is Re-Identification  - Dharmik's module. He built an OSNet-based feature extractor that "
        "produces 512-dimensional embeddings for each person crop. Each track maintains an appearance gallery "
        "with temporal decay  - recent appearances matter more. When a new detection appears and there are lost "
        "tracks, the Re-ID module computes cosine similarity against all lost track galleries. If the similarity "
        "exceeds 60%, the lost track is reactivated with its original ID."
    )
    pdf.speak(
        "Stage 4 is Visualization  - Agastya's module. He built an 800-line rendering engine with trajectory "
        "trails, ghost outlines for occluded objects, predicted future paths, a spatial heatmap, and a live "
        "analytics dashboard showing speeds, dwell times, and Re-ID events."
    )

    # ── DEMO SECTION ──
    pdf.add_page()
    pdf.section("4. Live Demonstrations", "~7-8 min")
    pdf.speak(
        "Now let me show you the system in action. I have six different demo videos prepared to show different "
        "capabilities."
    )

    # Demo 1
    pdf.stage_direction("Open terminal, activate venv, navigate to project directory")
    pdf.demo_command("python main.py --input demos/slow_walkers.mp4 --classes person")
    pdf.speak(
        "Demo 1  - Slow Walkers. This is the cleanest demo to start with. A few people walking slowly "
        "across a town. Notice the colored bounding boxes  - each person has a unique color and ID that persists "
        "as they move. You can see the trajectory trails showing their paths."
    )
    pdf.stage_direction("Press 'd' to show dashboard")
    pdf.speak(
        "I'm pressing 'd' to bring up the analytics dashboard. You can see the current object count, "
        "total entries and exits, and per-track speeds. This data updates in real-time."
    )
    pdf.stage_direction("Press 'h' to show heatmap")
    pdf.speak(
        "And pressing 'h' toggles the heatmap overlay  - this shows where objects spend the most time. "
        "You can see it building up along the walking paths."
    )
    pdf.stage_direction("Press 'q' to quit")

    # Demo 2
    pdf.demo_command("python main.py --input demos/pedestrians_subway.mp4 --classes person")
    pdf.speak(
        "Demo 2  - Subway Station. This is a much more challenging scene  - many people, fast movement, "
        "frequent occlusions. Watch how the tracker handles people crossing paths and briefly disappearing "
        "behind each other. You'll see the ghost outlines appear when someone gets occluded  - those little "
        "ghost icons show where the system predicts the hidden person is."
    )
    pdf.speak(
        "And watch the Re-ID events  - when a person who was lost reappears, you'll see the log showing "
        "'Track #X re-identified' with a similarity percentage. The system is matching their appearance "
        "features against the stored gallery."
    )
    pdf.stage_direction("Press 'q' to quit")

    # Demo 3
    pdf.demo_command("python main.py --input demos/dog_park.mp4 --classes dog,person")
    pdf.speak(
        "Demo 3  - Dog in the Park. This shows that our system isn't limited to people. Here we're "
        "filtering for both 'dog' and 'person' classes. The tracker maintains separate IDs for the dog "
        "and the person walking it. YOLO detects both, and BoT-SORT tracks both independently."
    )
    pdf.stage_direction("Press 'q' to quit")

    # Demo 4
    pdf.demo_command("python main.py --input demos/busy_street.mp4")
    pdf.speak(
        "Demo 4  - Busy Street. No class filter here  - we're detecting everything: people, cars, buses, "
        "motorcycles. This shows multi-class tracking in a fast-paced urban environment. Notice how "
        "each object type gets tracked with its own persistent ID."
    )
    pdf.stage_direction("Press 'q' to quit")

    # Demo 5
    pdf.demo_command("python main.py --input demos/night_street.mp4")
    pdf.speak(
        "Demo 5  - Night Scene. This tests the system under low-light conditions. YOLO and the tracker "
        "still work because we're using the yolo11s model which handles varied lighting well. You can see "
        "it detecting and tracking vehicles and pedestrians even at night."
    )
    pdf.stage_direction("Press 'q' to quit")

    # Demo 6 optional
    pdf.speak(
        "If time permits, I can also show Demo 6  - the Tokyo subway indoor scene, or even run it live "
        "on a webcam."
    )

    # ── KEY METRICS ──
    pdf.add_page()
    pdf.section("5. Key Results and Metrics", "~1 min")
    pdf.speak("Let me share some numbers:")
    pdf.key_point("YOLO detection runs at 72.8 FPS on our RTX 2050 with the yolo11s model.")
    pdf.key_point("The full pipeline  - detection, tracking, Re-ID, and visualization combined  - runs at 6 to 7 FPS on GPU. The bottleneck is Re-ID embedding extraction.")
    pdf.key_point("Re-ID successfully matches returning persons with 60 to 75 percent cosine similarity, well above our 60% threshold.")
    pdf.key_point("The occlusion state machine handles up to 3 seconds of occlusion  - 90 frames  - before transitioning a track to Lost state.")
    pdf.key_point("We tested on 6 diverse videos covering different scenarios: few people, crowded scenes, non-person objects, indoor, and night conditions.")

    # ── FUTURE SCOPE ──
    pdf.section("6. What's Next  - Final Submission", "~1 min")
    pdf.speak("For the final submission, we have four major improvements planned:")
    pdf.key_point("Grounding DINO integration  - this will let us detect objects using natural language prompts like 'person with red backpack', enabling open-vocabulary tracking.")
    pdf.key_point("An LSTM motion model to replace the current linear prediction with a learned model that handles curves and non-linear paths.")
    pdf.key_point("TensorRT optimization  - exporting our models to TensorRT to push the full pipeline above 20 FPS.")
    pdf.key_point("And a comprehensive ablation study comparing different model sizes, thresholds, and tracking parameters with standard MOT metrics like MOTA and IDF1.")

    # ── CLOSING ──
    pdf.section("7. Closing", "~30 sec")
    pdf.speak(
        "To summarize  - Phantom Tracker solves the occlusion problem in multi-object tracking by combining "
        "state-of-the-art detection, robust tracking with a custom state machine, deep appearance-based "
        "re-identification, and comprehensive real-time visualization. All four modules are implemented, "
        "integrated, and running on GPU."
    )
    pdf.speak(
        "I'm happy to take any questions."
    )

    # ── BACKUP: Q&A PREP ──
    pdf.add_page()
    pdf.section("Appendix: Anticipated Questions and Answers")
    pdf.speak("")

    qa = [
        ("Why BoT-SORT instead of ByteTrack or DeepSORT?",
         "BoT-SORT combines the best of both  - it has ByteTrack's two-stage association for handling low-confidence detections, plus deep appearance features like DeepSORT. It also adds camera motion compensation. In benchmarks, BoT-SORT consistently outperforms both on MOT17 and MOT20 datasets."),
        ("Why is the full pipeline only 6-7 FPS?",
         "The bottleneck is Re-ID. OSNet embedding extraction takes about 100ms per crop when running on the CPU path for some operations. For the final submission, we plan to batch the extractions and move everything to GPU with TensorRT, which should push us above 20 FPS."),
        ("How do you handle ID switches?",
         "BoT-SORT's appearance matching significantly reduces ID switches compared to pure IoU-based trackers. Our state machine adds an additional layer  - tracks don't immediately get new IDs when briefly lost. They go through Occluded state first with position prediction. And the Re-ID module catches any switches that do occur by matching appearance features."),
        ("What happens when two people look very similar?",
         "This is an inherent limitation of appearance-based Re-ID. OSNet produces 512-dimensional embeddings that capture discriminative features like clothing color, texture, and body proportions. In practice, we find it works well when people have distinct clothing. For identical-looking people, the system relies more on spatial proximity and motion continuity from BoT-SORT."),
        ("Can this run on a laptop without GPU?",
         "Yes. The system automatically falls back to CPU for all modules. YOLO detection will be slower  - around 5-8 FPS instead of 70+  - and the tracker also has an IoU-based fallback if boxmot can't initialize. The experience is slower but fully functional."),
        ("What is the ghost icon on occluded objects?",
         "That's Agastya's visualization for occluded tracks. When an object becomes hidden, we show a ghost outline at the predicted position along with a ghost icon  - a circle with two dots for eyes. The expanding ring around it shows increasing uncertainty as the object stays hidden longer."),
    ]

    for q, a in qa:
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(0, 70, 140)
        pdf.multi_cell(0, 6, f"Q: {q}")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, f"A: {a}")
        pdf.ln(4)

    # Save
    pdf.output("docs/Phantom_Tracker_Pitch_Script.pdf")
    print("Pitch script saved: docs/Phantom_Tracker_Pitch_Script.pdf")


if __name__ == "__main__":
    build()
