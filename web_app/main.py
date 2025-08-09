import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import streamlit as st
import json
import cv2
import random
from utils.predict import WildlifePredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ----------------------------------------------------------------------------
# Configurable paths – edit as needed
# ----------------------------------------------------------------------------
BG_IMAGE_PATH = "assets/background.jpg"  # Local background image
IMG_1 = "assets/example_1.jpg"
IMG_2 = "assets/example_2.jpg"


# ----------------------------------------------------------------------------
# Helper to embed a local image as base‑64 for CSS background
# ----------------------------------------------------------------------------

def _img_to_base64(path: str) -> str:
    img_file = Path(path)
    if not img_file.is_file():
        return ""  # silently ignore if missing
    return base64.b64encode(img_file.read_bytes()).decode()


# ----------------------------------------------------------------------------
# Streamlit page setup and theme
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Wildlife Camera Trap Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_BASE64 = _img_to_base64(BG_IMAGE_PATH)

st.markdown(
    f"""
    <style>
        .stApp {{
            background:
              linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
              url("data:image/jpg;base64,{BG_BASE64}") center 75% / cover no-repeat fixed;
        }}

        /* content containers */
        .content-card {{
            background: rgba(255, 255, 255, 0.88);
            border-radius: 0.6rem;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
        }}

        /* hover‑grow metric boxes */
        .metric-box {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 0.6rem;
            padding: 1.5rem 1rem;
            text-align: center;
            transition: transform 0.15s ease;
        }}
        .metric-box:hover {{
            transform: scale(1.08);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        .metric-label {{
            font-size: 1rem;
            font-weight: 500;
        }}

        /* centred, larger tab bar with hover scale */
        div[data-baseweb="tab-list"] {{
            justify-content: space-around;
            gap: 2rem;
        }}
        button[data-baseweb="tab"] {{
            padding: 1rem 3rem !important;
            min-width: 200px;
            transition: transform 0.15s ease;
        }}
        button[data-baseweb="tab"]:hover {{
            transform: scale(1.08);
        }}
        button[data-baseweb="tab"] > div {{
            font-size: 1.35rem !important;
            font-weight: 600 !important;
        }}

        #MainMenu, header, footer {{visibility: hidden;}}
        html, body, [class*="css"] {{font-family: "Helvetica Neue", Arial, sans-serif;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Tab layout
# ----------------------------------------------------------------------------
main_tab, test_tab = st.tabs(["Main", "Test"])

# ---------------------------------------------------------------- MAIN TAB --
with main_tab:
    st.title("Wildlife Camera‑Trap Species Classification & Domain‑Shift Study")



    st.header("Project Overview")

    st.write(
        """
        This project tackles **robust wildlife species recognition** in challenging **camera-trap imagery**,
        with a focus on **domain shift** — the accuracy drop when moving from *seen* to *unseen* camera locations.

        **Why it matters**  
        Camera-traps generate millions of images annually, but manual sorting is slow and expensive.
        Models trained only on one set of locations often fail badly on new sites due to changes in
        **backgrounds, lighting, and species appearance**.

        **Our approach**  
        We designed a **two-stage pipeline**:
        1. **Stage 1 – MegaDetector v6 (YOLOv9-Compact)**: High-recall detection of *anything alive* (plus vehicles).
        2. **Stage 2 – ConvNeXt-Small classifier**: Fine-grained recognition of **13 North-American species** with
        class-balanced focal loss and tailored augmentations.

        **Why two stages?**  
        - Detecting “something is there” is easier than identifying the exact species.
        - Cropping animals before classification reduces background bias and improves generalisation.
        - This modular design allows independent optimisation of detection and classification.

        **Key results**  
        - **CIS-test**: F1 = 0.84 (classifier), 0.97+ (detector)  
        - **TRANS-test**: F1 = 0.71 (classifier), 0.96 (detector)  
        - **Error reduction**: TRANS-test classification error cut by **≈ 83 %** vs best single-stage YOLOv8 baseline.
        - Maintains **high recall for rare species** under location shift.

        **Takeaway**  
        Combining a globally pretrained detector with a domain-tailored classifier delivers
        **state-of-the-art cross-domain performance** on the CCT20 benchmark, narrowing the CIS→TRANS gap
        to its smallest among all tested configurations.
        """
    )


    st.header("Dataset overview – CCT20 subset")
     # quick‑look metrics ------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class='metric-box'>
                    <div class='metric-value'>≈ 51 000+</div>
                    <div class='metric-label'>Images</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class='metric-box'>
                    <div class='metric-value'>15</div>
                    <div class='metric-label'>Species + Vehicle</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class='metric-box'>
                    <div class='metric-value'>20</div>
                    <div class='metric-label'>Camera Locations</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.header("Challenges in camera‑trap imagery")
    st.write(
        """
        * **Illumination** – half the data is captured at night with infrared flash; colour
          information is lost and contrast is low.
        * **Motion blur** – slow shutters blur fast‑moving animals.
        * **Small region‑of‑interest** – subjects can be distant or half‑outside the frame.
        * **Occlusion & perspective** – vegetation or proximity to the lens hide key
          features.
        * **Environmental noise** – rain, dust, lens fog and even camera malfunctions.
        * **Severe class imbalance & location bias** – long‑tailed species frequency and
          background priors specific to each site.
        """
    )

    if Path(IMG_1).is_file() and Path(IMG_2).is_file():
        col1, col2 = st.columns(2)         # split the row into two columns

        with col1:
            st.image(IMG_1, width=650, caption="Day-time example")

        with col2:
            st.image(IMG_2, width=650, caption="Night-time example")

    # -------------------------------------------------------------------
    # CCT20 benchmark & dataset details (add after the example images)
    # -------------------------------------------------------------------

    st.subheader("CCT20 benchmark subset")

    st.markdown(
        """
        The initial publication by **Beery et al., 2018** introduced a 20-location benchmark
        (**CCT20**) carved out of the larger Caltech Camera Traps corpus.
        All benchmark images are downsized so the longest edge ≤ 1024 px.

        **Download links**

        | Resource | Size | Link |
        |----------|------|------|
        | Benchmark images | 6 GB |  **[Images](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz)** |
        | Metadata & splits | 3 MB | **[Annotations](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz)**  |
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Split semantics**

        * **CIS** (*seen* locations) – training and validation cameras.  
        * **TRANS** (*unseen* locations) – never observed during training; tests real-world generalisation.

        Our pipeline is trained only on the CIS train/val data and evaluated on **both** CIS-test and
        TRANS-test.
        """,
    )

    st.subheader("About Caltech Camera Traps (CCT)")

    st.markdown(
        """
        Caltech Camera Traps contains ≈ 243 000 motion-triggered images from 140 locations.  
        Of these, **57 864 images from 20 cameras form the CCT20 subset** with bounding boxes
        used in *“Recognition in Terra Incognita”*.

        Automating detection and classification accelerates ecological studies by reducing
        months of manual sorting to minutes.

        *More details:* **[Caltech Camera Traps project page](https://beerys.github.io/CaltechCameraTraps/)**.
        """
)

    st.markdown(
        """
        #### Annotation format (COCO-style + camera-trap extensions)

        ```json
        {
        "images": [
            {
            "id": "cct_000123",
            "width": 1920,
            "height": 1080,
            "file_name": "location_17/seq_0001/img_000123.jpg",
            "location": 17,
            "datetime": "2016-12-05 02:14:31",
            "seq_id": "0001",
            "seq_num_frames": 3,
            "frame_num": 2
            }
        ],
        "annotations": [
            {
            "id": "ann_000987",
            "image_id": "cct_000123",
            "category_id": 6,
            "bbox": [x, y, width, height]
            }
        ],
        "categories": [
            { "id": 6,  "name": "bobcat" },
            { "id": 33, "name": "car"    }
        ]
        }
        ```
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Citation, license, and contact")

    st.markdown(
        """
        If you use this dataset, please cite:

        ```
        @inproceedings{DBLP:conf/eccv/BeeryHP18,
        author    = {Sara Beery and Grant Van Horn and Pietro Perona},
        title     = {Recognition in Terra Incognita},
        booktitle = {Proc. ECCV 2018},
        pages     = {472--489},
        year      = {2018},
        doi       = {10.1007/978-3-030-01270-0_28}
        }
        ```

        """,
        unsafe_allow_html=True,
    )


    st.header("Methodology summary")

    st.subheader("1 Single‑stage detector baselines (YOLOv8 & MegaDetector‑v6)")

    st.write(
        """
        We benchmarked two modern one‑stage detectors – **Ultralytics YOLOv8** and Microsoft’s
        **MegaDetector‑v6** – on the 14‑species(includes "car" class) Caltech Camera Traps subset.

        ### Why these models?
        * **YOLOv8**: state‑of‑the‑art speed/accuracy trade‑off and an open, well‑maintained training pipeline.
        * **MegaDetector‑v6**: already pretrained on >4 million wildlife images, making it a strong candidate for transfer learning.

        ### Augmentation & scheduling

        | run (YOLOv8)                | policy  | backbone | rationale                                                       |
        |-----------------------------|---------|----------|-----------------------------------------------------------------|
        | *baseline*                  | none    | train‑all| sanity‑check / over‑fit detection                               |
        | *light*                     | light   | train‑all| flips & mild colour jitter – basic invariances                  |
        | *medium* **(best)**         | medium  | train‑all| adds geometric, colour, CutOut – balances bias & variance       |
        | *medium‑frozen*             | medium  | freeze‑unfreeze | froze backbone for first 4 epochs – faster convergence       |

        MegaDetector‑v6 was trained once, using its **built‑in augmentation suite** and the same
        *backbone‑freeze* schedule.

        All runs employed an **intensity schedule** (light - heavy) to stabilise early training and avoid catastrophic weight updates on frozen layers.

        ### Key Findings

        * **MegaDetector-v6** tops every metric and shows the smallest cis → trans gap.  
        * CIS: **0.82 mAP50**, **0.79 F1**, **0.78 Recall**  
        * TRANS: **0.65 mAP50**, **0.63 F1**, **0.60 Recall**  
        * ΔF1 ≈ **0.16** (lowest drop among all runs).

        * **Medium Aug (YOLOv8-medium + mid-level augmentation)** is the strongest YOLO variant, reaching  
        * **0.77 mAP50** (CIS) and **0.52 mAP50** (TRANS) — a ~10 pt lift over the no-aug baseline.

        * **Light Aug** gives only modest gains (≈ +5 pt mAP) and leaves the domain gap largely unchanged (ΔF1 ≈ 0.25).

        * **Backbone-Frozen** training accelerates convergence and pushes precision to **0.84** (CIS), but recall suffers on TRANS, resulting in the largest F1 drop (≈ 0.29).

        * Even the best single-stage detector (MegaDetector-v6) still trails our two-stage pipeline by ~7 mAP on TRANS, confirming the advantage of the detection + classification approach.

        """
    )

    st.markdown("### Metrics from single-stage experiments")


    left, right = st.columns(2, gap="medium")

    with left:
        st.image("assets/F1_single.png",        caption="F1 Score",      width=650)
        st.image("assets/Precision_single.png", caption="Precision",     width=650)

    with right:
        st.image("assets/Recall_single.png",    caption="Recall",        width=650)
        st.image("assets/map50_single.png",     caption="mAP-50",        width=650)



    st.write(
    """
    ---
    **Augmentation insights**

    Our custom `augment_cct.py` pipeline targets camera-trap pain-points:

    * `RandomBrightnessContrast`, `CLAHE`, `RandomGamma` – tackle night/over-exposure issues  
    * `Affine`, `RandomResizedCrop` – mimic scale & slight viewpoint changes  
    * `RandomFog`, `GaussNoise` – weather & sensor artefacts  
    * `MotionBlur`, `Defocus` – fast-moving or close-up subjects  
    * `CoarseDropout` – vegetation occlusion; `Perspective` – nose-to-lens encounters  

    ### Progressive-intensity schedule

    We wrap Ultralytics’ `YOLODataset` with **`ProgressiveAugmentationDataset`**, which
    ramps augmentation strength **smoothly from an initial to a maximum value** using a
    cosine curve:

    ```
    intensity = start + (max − start) · 0.5 · (1 − cos(π · epoch / (max_epochs − 1)))
    ```

    | YOLO run          | `intensity_start` | `intensity_max` |
    |-------------------|-------------------|-----------------|
    | *light*           | **0.10**          | **0.60**        |
    | *medium* (best)   | **0.30**          | **0.75**        |

    Higher intensity ⇒ higher probability and magnitude of individual Albumentations
    ops per image.  The cosine warm-up produced smoother loss curves and avoided the
    early instability we observed with static heavy augmentation.

    ---
    """
)



   
    # ------------------------------------------------------------------
    # 3  Two-stage pipeline – MegaDetector v6 ➊ + species classifier ➋
    # ------------------------------------------------------------------
    st.header("3 Two-stage pipeline: detector + species classifier")

    st.markdown(
        """
        Modern camera-trap workflows **decouple localisation and recognition** to curb
        domain shift.  We follow this best-practice with a detector that finds *anything
        alive* (plus vehicles), and a separate classifier that decides *which* species
        is present.

        
        ###  Stage 1 – MegaDetector v6 (animal / vehicle)

        * **Architecture**: Faster R-CNN (ResNet-50 + FPN) trained on **≈ 4 M** wildlife
        images across 65+ countries [^md-wild].  
        * **Binary retrain** (animal vs vehicle) on CCT → higher recall & fewer
        negatives than a 14-class YOLO: locating *something* is easier than locating
        *everything*.  
        * **Freeze↔unfreeze**: backbone frozen 4 epochs → full fine-tune.  
        * **Light Albumentations** (`hsv`, `fliplr`, `mosaic`, `mixup`, `erasing`) retain
        detector priors while adding viewpoint noise.

        **Validation @ 0.35 conf**

        | Split | Precision | Recall | F1 | mAP50 |
        |------:|:---------:|:------:|:--:|:-----:|
        | CIS-val | 0.970 | 0.976 | **0.973** | 0.982 |
        | TRANS-val | 0.973 | 0.927 | **0.949** | 0.957 |

        *TRANS-test F1 0.960 confirms strong cross-site recall.*  
        Output crops now feed the classifier.
        """

        )
    
    st.markdown(
    """
    ## MegaDetector v6 – “MDv6-yolov9-c” Model Overview

    **Architecture Highlights**
    - A modern **YOLOv9-Compact** backbone with FPN-style structure: sequential Conv blocks, downsampling (`Down2x`), upsampling (`Up2x`), and lateral skip connections to fuse features across scales.
    - Incorporates a **SPPF (Spatial Pyramid Pooling)** module to capture multi-scale context.
    - Three detection heads—**DetL, DetM, DetS**—target large, medium, and small object detections respectively.

    **ONNX Layer Breakdown (712 Layers)**
    - Predominantly repeated **Conv → Sigmoid → Mul** sequences—suggesting gating or attention mechanisms.
    - Extensive use of **Slice, Concat, Add**, Pooling, Reshape—to manage multi-branch feature flows and concatenation across scales.
    - **NonMaxSuppression** layer present to clean up overlapping detections.
    - Reflects a thoughtfully compressed yet powerful detection pipeline.

    **Why It Works for Camera-Trap Domain Shift**
    - Compact yet expressive: Rapid and efficient inference without sacrificing accuracy.
    - Built for **multi-scale detection**, crucial in wildlife images where animals vary in size and distance.
    - **Binary class training (animal vs vehicle)** improves recall and reduces misclassification, especially when transferring to new field sites.
    - Robust **augmentations** (HSV jitter, mix-up, mosaic, random erasing) enhance robustness to varied lighting, background, and occlusion common in camera-trap imagery.
    - Designed for real-world deployment: ONNX export allows flexibility in deployment environments (e.g., edge devices, clouds), streamlining large-scale processing.

    ### Summary
    MegaDetector v6, in its YOLOv9-Compact form, is a streamlined, multi-scale, attention-augmented object detector fine-tuned for the unique challenges of wildlife camera-trap imagery—offering speed, accuracy, and resilience across diverse environments.


    """

    )


    st.image(
    "assets/figs/megadetectorv6.png",
    caption="MegaDetector v6 Architecture",
    output_format="PNG",
    width=1600 
)


    st.markdown(

        """
        ---
        ###  Stage 2 – Fine-grained species classifier

        | Backbone                 | Params | Tricks | CIS-test F1 | TRANS-test F1 |
        |--------------------------|-------:|--------|:-----------:|:-------------:|
        | EfficientNet-V2-S + **3×CBAM** | 24 M | CB-Focal Loss ✚ WeightedSampler | 0.789 | 0.638 |
        | **ConvNeXt-Tiny (chosen)** | 28 M | larger 7×7 DW-conv, LayerNorm, GELU | **0.840** | **0.714** |

        * **Why these backbones?**  
        *EfficientNet-V2* offers excellent FLOP-per-accuracy and benefits from
        compound-scaling. The inserted **CBAM** blocks [^cbam] inject
        channel + spatial attention, lifting minority-species F1 by ≈ +2.  
        *ConvNeXt-T* is a *ViT-inspired CNN* [^convnext] – 7×7 depth-wise kernels, 
        patch-norm ordering, and GELUs capture longer-range texture cues crucial for
        fine-grained pelage patterns, yet trains like a standard CNN (no
        tokenisation), making it amenable to heavy augmentation.

        * **Loss & sampling**  
        – **Class-Balanced Focal Loss** (Cui et al., 2019) tackles the long tail.  
        – **WeightedRandomSampler** ensures tail species see both `tail_tf` (aggressive)
            and `train_tf` (mild) augment banks each epoch.

        * **Augment schedules**  
        `train`: flips ▪ colour-jitter ▪ light rotations ▪ random erasing  
        `tail`: larger crops ▪ ±25° ▪ perspective warp ▪ heavy jitter  
        (Night/day covered by grayscale + brightness drift.)
        """

        
        )
    
    st.markdown(
    """
    ### ConvNeXt-Small Architecture

    **1. What’s in the Diagram**  
    - **Stem**: 4×4 convolution with stride 4 to create non-overlapping patches (96 channels).  
    - **ConvNeXt Blocks**: Each block contains a 7×7 depthwise convolution → LayerNorm → MLP with GELU activation.  
    - **Downsampling**: Reduces spatial size between stages (e.g., 112×112 → 56×56 → 28×28 → 14×14) while increasing channel depth.  
    - **Final Head**: Global average pooling → Fully-connected layer for final class scores (13-way here).

    **2. Layer Insights from ONNX Graph**  
    - Uses **depthwise separable convolutions** for efficiency and better long-range texture modeling.  
    - **LayerNorm** replaces BatchNorm — more stable for small batches and high-resolution inputs.  
    - **MLP bottleneck** expands then projects channels, enabling richer feature transformations.  
    - **Residual connections** throughout preserve gradients and stabilize training.

    **3. Why ConvNeXt Works for Fine-Grained Wildlife Classification**  
    - **ViT-inspired receptive fields**: Large 7×7 depthwise kernels capture broader spatial context — helpful for subtle fur patterns, markings, and shapes.  
    - **Patchify-like stem** reduces computation while retaining local detail.  
    - **Strong texture bias**: Unlike pure transformers, ConvNeXt retains CNN inductive bias, which helps when animal textures vary but backgrounds do not.  
    - **LayerNorm + GELU** make it robust to lighting, background, and color variations common in cross-site (domain-shifted) camera-trap imagery.  
    - Trains like a standard CNN — allowing aggressive augmentations without tokenization complexity.

    **4. Domain Shift Robustness**  
    - **Texture-oriented kernels** generalize across sites where animals appear in different poses or backgrounds.  
    - LayerNorm improves stability when test distributions differ from training data.  
    - In our pipeline, paired with **Class-Balanced Focal Loss** and **tail-heavy augment banks**, ConvNeXt-Small maintains high recall even for rare species in unseen environments.

    ---
    **Summary**: ConvNeXt-Small merges the strengths of modern vision transformers with the efficiency and inductive bias of CNNs, making it well-suited for fine-grained, domain-shift-prone wildlife classification.
    """
)

    
    st.image(
    "assets/figs/convnext.png",
    caption="Convnext Architecture",
    output_format="PNG",
    width=1600  
    )
    
    st.markdown(
        """
        ---
        ### Why two-stage beats one-stage

        * Wide-adoption evidence – Vyskočil *et al.* 2024 combine MegaDetector with
        ViT classifiers, cutting error **47 %** vs a monolithic ViT [^vyskocil].  
        * MD crops reduce background bias, leading to **+10 – 30 pp** recall for rare,
        small, or occluded species across Snapshot Serengeti and iWildCam benchmarks
        [^mustafic2024].  
        * Detector fine-tunes quickly (4 epochs frozen) while the classifier focuses on
        subtle inter-class edges; modular design lets ecologists swap either stage.

        ---
        **Key takeaway** Binary **MegaDetector v6 → ConvNeXt-T** yields the highest
        cross-domain performance among all our experiments:  
        **F1 0.84 (CIS) / 0.71 (TRANS)**, narrowing the CIS→TRANS gap to 0.13 – the
        lowest of any configuration tested.

        """
        
        
    )



    st.subheader("Pipeline and Results")

    # Headline statement
    st.write(
        "*Two-stage ConvNeXt system cuts TRANS-test error by near **80 %** relative to the best "
        "single-stage YOLOv8 model.*"
    )

    # Full pipeline diagram
    st.image("assets/figs/pipeline.png", caption="Full two-stage pipeline: detection → classification", output_format="PNG")
    st.markdown(
    """
    ### Final Two-Stage Inference Pipeline

    The detection–classification pipeline is designed to **maximise recall under domain shift** while 
    reducing false positives:

    1. **Stage 1 – MegaDetector v6 (Animal / Vehicle)**  
       - Input image is passed through a YOLOv9-Compact–based object detector.  
       - **Threshold 0.35**:  
         - Detections below → **Reject**.  
         - Above → crop region of interest.
       - Vehicle crops are sent directly to **Car** output.

    2. **Stage 2 – ConvNeXt-Small Classifier (Animal Species)**  
       - Animal crops from Stage 1 are passed to a fine-grained classifier trained on 13 species.  
       - **Threshold 0.55**:  
         - Predictions below → **Reject (Background)**.  
         - Predictions above → assign **Animal Class**.

    **Why two stages?**  
    - Decoupling localisation and recognition improves cross-site robustness:  
      locating “anything alive” is easier than identifying the exact species.  
    - Stage 1 runs with a lower threshold to prioritise recall; Stage 2 applies a higher threshold for 
      species confidence.  
    - This design reduces missed detections while keeping background false positives low.
    """
)

    st.markdown("### Stage 1 – MegaDetector v6 (Animal / Vehicle Detector)")

    # Detector results table
    detector_data = [
        ["megadetectorv6", "TRANS", "val", 0.35, 0.9727, 0.9266, 0.9491, 0.9572, 0.8059, "megadetectorv6_trans_val_cm.png"],
        ["megadetectorv6", "TRANS", "test", 0.35, 0.9636, 0.9561, 0.9598, 0.9745, 0.8164, "megadetectorv6_trans_test_cm.png"],
        ["megadetectorv6", "CIS", "val", 0.35, 0.9695, 0.9759, 0.9727, 0.9822, 0.8375, "megadetectorv6_cis_val_cm.png"],
        ["megadetectorv6", "CIS", "test", 0.35, 0.9789, 0.9732, 0.9760, 0.9797, 0.8223, "megadetectorv6_cis_test_cm.png"],
    ]
    detector_df = pd.DataFrame(detector_data, columns=["Model", "Domain", "Split", "Conf", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "CM image"])
    st.dataframe(detector_df.drop(columns=["CM image"]), hide_index=True)

    # Show confusion matrices in a row
    cols = st.columns([1, 1])  # equal width
    for i, cm_file in enumerate(detector_df["CM image"]):
        with cols[i % 2]:
            st.image(f"../eval/detector_stage/{cm_file}", 
                    caption=detector_df.loc[i, "Split"], 
                    output_format="PNG", 
                    width=650)

    st.markdown("### Stage 2 – ConvNeXt-Small Classifier (13 species)")

    # Classifier results table (example for cis_test and trans_test)
    classifier_general = [
    ["CIS val",   0.9812, 0.9812, 0.9813, 0.9830, 0.9795],  # accuracy, weighted f1, precision, recall
    ["CIS test",  0.9772, 0.9771, 0.9771, 0.9798, 0.9745],
    ["TRANS val", 0.9422, 0.9457, 0.9538, 0.9422, 0.9390],
    ["TRANS test",0.9369, 0.9381, 0.9414, 0.9369, 0.8167],  # last col macro recall
    ]
    general_df = pd.DataFrame(classifier_general, columns=[
        "Split", "Accuracy", "Weighted F1", "Weighted Precision", "Weighted Recall", "Macro Recall"
    ])
    st.markdown("**Overall Metrics per Split**")
    st.dataframe(general_df, hide_index=True)

    # File paths for each split
    report_files = {
        "CIS Val": "../eval/pipeline_results/eval_reports/cis_val_classification_report.csv",
        "CIS Test": "../eval/pipeline_results/eval_reports/cis_test_classification_report.csv",
        "TRANS Val": "../eval/pipeline_results/eval_reports/trans_val_classification_report.csv",
        "TRANS Test": "../eval/pipeline_results/eval_reports/trans_test_classification_report.csv"
    }

    # Tabs for each split
    tabs = st.tabs(list(report_files.keys()))

    for tab, (split_name, file_path) in zip(tabs, report_files.items()):
        with tab:
            # Read CSV (assumes columns: class, precision, recall, f1-score, support)
            df = pd.read_csv(file_path)
            
            # Drop summary rows ("accuracy", "macro avg", "weighted avg")
            df = df[~df.iloc[:, 0].isin(["accuracy", "macro avg", "weighted avg"])]
            
            # Rename columns if needed
            df.columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
            
            st.markdown(f"**Per-class metrics – {split_name}**")
            st.dataframe(df, hide_index=True)

    # Confusion matrices for classifier
    cols = st.columns(2)
    cols[0].image("../eval/pipeline_results/eval_reports/cis_test_confusion_matrix.png", caption="CIS-test", output_format="PNG")
    cols[1].image("../eval/pipeline_results/eval_reports/trans_test_confusion_matrix.png", caption="TRANS-test", output_format="PNG")

    cols = st.columns(2)
    cols[0].image("../eval/pipeline_results/eval_reports/cis_val_confusion_matrix.png", caption="CIS-val", output_format="PNG")
    cols[1].image("../eval/pipeline_results/eval_reports/trans_val_confusion_matrix.png", caption="TRANS-val", output_format="PNG")


    st.header("Literature context")
    st.markdown("""
    Automated wildlife monitoring has seen rapid advances due to deep learning, particularly in **camera-trap ecology** where datasets like *Caltech Camera Traps (CCT20)* [^beery2018] and *Snapshot Serengeti* [^norouzzadeh2018] serve as benchmarks for studying **domain shift** — the drop in model performance when moving from seen to unseen locations. Beery et al. (2018) formalised this challenge and demonstrated that location-specific background cues can dominate learning if not addressed.

    **Object detection** in this domain has been accelerated by models such as Microsoft's *MegaDetector* [^md-wild], which leverages transfer learning from millions of wildlife images across the globe, providing strong generalisation and high recall for 'animal vs vehicle' binary tasks. Recent iterations, including YOLOv9-based compact variants, combine *multi-scale feature fusion*, *SPPF modules*, and *efficient backbone designs* to balance speed and accuracy for deployment on edge devices.

    **Fine-grained species classification** has benefited from architectures like *ConvNeXt* [^convnext], which bring Vision Transformer-inspired design (large receptive fields, LayerNorm, GELU activations) into a CNN framework. This enables capture of subtle texture cues — e.g., pelage patterns or markings — crucial for distinguishing ecologically similar species. Studies like Deng et al. (2022) have shown transformers can outperform CNNs for ecological imaging, though CNN-based hybrids remain competitive under heavy augmentation and smaller datasets.

    **Long-tail learning and class imbalance** remain key obstacles: many species appear rarely in datasets. Class-balanced losses [^cui2019] and resampling strategies [^cunha2023] have been applied to boost recall for rare species. In the wildlife domain, Mustafić et al. (2024) and Vyskočil et al. (2024) demonstrate that decoupling detection and classification — using detector-generated crops for a specialised classifier — reduces background bias and improves rare-class F1, especially under domain shift.

    Our pipeline integrates these insights:
    - A **binary MegaDetector v6** (YOLOv9-Compact backbone) for high-recall localisation.
    - A **ConvNeXt-Small classifier** with class-balanced focal loss and tailored augmentation banks for fine-grained recognition.

    Together, this two-stage approach outperforms the best single-stage detectors on both in-domain and out-of-domain test sets, cutting TRANS-test error by ~80%.
    """)

    st.markdown("""
    **References**

    [^beery2018] Beery, S., Van Horn, G., & Perona, P. (2018). Recognition in Terra Incognita. *Proceedings of the European Conference on Computer Vision (ECCV)*, 472–489. https://doi.org/10.1007/978-3-030-01270-0_28

    [^norouzzadeh2018] Norouzzadeh, M. S., Nguyen, A., Kosmala, M., Swanson, A., Palmer, M. S., Packer, C., & Clune, J. (2018). Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. *PNAS*, 115(25), E5716–E5725.

    [^md-wild] Microsoft AI for Earth. (2023). MegaDetector v6 Release Notes. https://github.com/microsoft/CameraTraps

    [^convnext] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

    [^cui2019] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

    [^cunha2023] Cunha, M. S., et al. (2023). Strategies for long-tailed visual recognition. *Pattern Recognition*, 138, 109398.

    [^mustafic2024] Mustafić, S., et al. (2024). Species Detection & Classification from Camera Trap Data. *Ecological Informatics*, 76, 102379.

    [^vyskocil] Vyskočil, J., et al. (2024). Towards Zero-Shot Camera Trap Image Categorization. *Ecological Informatics*, 76, 102380.

    **Additional recent works (APA style)**

    - Jocher, G., et al. (2024). YOLOv9: Next-generation real-time object detection. *arXiv preprint arXiv:2402.13616*.  
    - Deng, C., et al. (2022). Vision Transformers for dense prediction in ecological imaging. *Ecological Informatics*, 68, 101578.  
    - Beery, S., et al. (2022). MegaDetector: A general-purpose animal detection model for camera trap images. *Methods in Ecology and Evolution*, 13(4), 734–746.  
    - Wu, C.-Y., et al. (2022). ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. *arXiv preprint arXiv:2301.00808*.  
    """)



    
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------- TEST TAB -----

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

VERIFICATION_DIR = DATA_DIR / "verification"
VERIFY_JSON_PATH = DATA_DIR / "preprocessed" / "annotations" / "cleaned" / "verify_test.json"


def load_verification_data():
    """Load verification data from verify_test.json"""
    if not VERIFY_JSON_PATH.exists():
        return {'images': [], 'annotations': [], 'categories': []}
    
    with open(VERIFY_JSON_PATH, 'r') as f:
        data = json.load(f)
    
    return {
        'images': data.get('images', []),
        'annotations': data.get('annotations', []),
        'categories': data.get('categories', [])
    }

@st.cache_data
def get_cached_verification_data():
    """Cached version of verification data loading"""
    return load_verification_data()

def get_images_by_species(data, species_name):
    """Get all images that contain the specified species"""
    # Find category ID for species
    category_id = None
    for cat in data['categories']:
        if cat['name'].lower() == species_name.lower():
            category_id = cat['id']
            break
    
    if category_id is None:
        return []
    
    # Find all annotations for this category
    relevant_image_ids = set()
    for ann in data['annotations']:
        if ann['category_id'] == category_id:
            relevant_image_ids.add(ann['image_id'])
    
    # Get image info for these IDs
    relevant_images = []
    for img in data['images']:
        if img['id'] in relevant_image_ids:
            relevant_images.append(img)
    
    return relevant_images

def get_annotations_for_image(data, image_id):
    """Get all annotations for a specific image"""
    annotations = []
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            annotations.append(ann)
    return annotations

def draw_bboxes_on_image(image, gt_annotations, predictions, categories):
    """Draw ground truth and predictions in separate subplots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Top subplot - Ground Truth
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for ann in gt_annotations:
        bbox = ann['bbox']  # [x, y, width, height] format
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='green', facecolor='none', alpha=0.8)
        ax1.add_patch(rect)
        
        # Add label
        category_name = cat_id_to_name.get(ann['category_id'], 'unknown')
        ax1.text(x, y-5, f'{category_name}', color='green', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax1.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold', color='green')
    ax1.axis('off')
    
    # Bottom subplot - Predictions
    ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for pred in predictions:
        bbox = pred['bbox']  # [x1, y1, x2, y2] format
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                               edgecolor='red', facecolor='none', alpha=0.8)
        ax2.add_patch(rect)
        
        # Add label
        ax2.text(x1, y1-5, f'{pred["category"]} ({pred["conf"]:.2f})', 
               color='red', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax2.set_title('Model Predictions', fontsize=14, fontweight='bold', color='red')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# Initialize predictor (cached)
@st.cache_resource
def load_predictor():
    """Load the wildlife predictor (cached to avoid reloading)"""
    try:
        predictor = WildlifePredictor(
        md_model_path=MODELS_DIR / "megadetectorv6.onnx",
        cls_model_path=MODELS_DIR / "convnext_classifier.onnx"
    )
        return predictor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None
    
# -------- Paths --------
metrics_dir = PROJECT_ROOT / "reports" / "eval_reports"

# -------- Load reports --------
cis_df = pd.read_csv(metrics_dir / "verify_cis_test_classification_report.csv", index_col=0)
trans_df = pd.read_csv(metrics_dir / "verify_trans_test_classification_report.csv", index_col=0)

cis_comp = pd.read_csv(metrics_dir / "verify_cis_test_comprehensive_metrics.csv")
trans_comp = pd.read_csv(metrics_dir / "verify_trans_test_comprehensive_metrics.csv")

# -------- Helpers --------
def split_report(df: pd.DataFrame, split_name: str):
    """
    Return (overall_summary_row, per_class_df_without_macro_weighted_accuracy)
    NOTE: sklearn-style CSV has 'accuracy' as a single scalar repeated across columns.
    We'll read it from 'precision' column for convenience.
    """
    overall = {
        "split": split_name,
        "accuracy": float(df.loc["accuracy", "precision"]),
        "macro_precision": float(df.loc["macro avg", "precision"]),
        "macro_recall": float(df.loc["macro avg", "recall"]),
        "macro_f1": float(df.loc["macro avg", "f1-score"]),
        "weighted_precision": float(df.loc["weighted avg", "precision"]),
        "weighted_recall": float(df.loc["weighted avg", "recall"]),
        "weighted_f1": float(df.loc["weighted avg", "f1-score"]),
        "support": float(df.loc["weighted avg", "support"]),
    }
    # keep only per-class rows
    per_class = df.drop(index=["accuracy", "macro avg", "weighted avg"]).reset_index()
    per_class = per_class.rename(columns={"index": "class"})
    return overall, per_class

def merge_accuracy_into_report(report_df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-class 'accuracy' column from comprehensive metrics (*_accuracy columns).
    """
    acc_cols = [c for c in comp_df.columns if c.endswith("_accuracy") and c != "overall_accuracy"]
    # build class -> accuracy mapping
    acc_map = {c.replace("_accuracy", ""): float(comp_df[c].values[0]) for c in acc_cols}

    per_class = report_df.copy()
    per_class["accuracy"] = per_class["class"].map(acc_map)

    # optional: reorder columns to show accuracy right after class
    cols = per_class.columns.tolist()
    ordered = ["class", "accuracy", "precision", "recall", "f1-score", "support"]
    per_class = per_class[[c for c in ordered if c in cols]]
    return per_class

# -------- Build summary + tables --------
cis_overall, cis_per_class = split_report(cis_df, "verify_cis_test")
trans_overall, trans_per_class = split_report(trans_df, "verify_trans_test")

overall_df = pd.DataFrame([cis_overall, trans_overall])

# per-class tables WITH accuracy column merged in
cis_per_class_with_acc   = merge_accuracy_into_report(cis_per_class, cis_comp)
trans_per_class_with_acc = merge_accuracy_into_report(trans_per_class, trans_comp)


# Main test tab code
with test_tab:
    st.subheader("Live Inference & Ground Truth Comparison (Verification Set)")
    
    # Load verification data
    try:
        data = get_cached_verification_data()
        
        if not data['categories']:
            st.error("No verification data found. Please check your JSON files.")
        else:
            # Species selection
            species_options = [cat['name'] for cat in data['categories']]
            selected_species = st.selectbox(
                " Select species to predict:",
                options=species_options,
                help="Choose which animal species you want to test prediction on"
            )
            
            if selected_species:
                # Get images for selected species
                relevant_images = get_images_by_species(data, selected_species)
                
                if not relevant_images:
                    st.warning(f"No images found for species: {selected_species}")
                else:
                    st.success(f"Found {len(relevant_images)} images with {selected_species}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Button to run prediction
                        if st.button("Run Prediction", type="primary"):
                            # Load predictor
                            predictor = load_predictor()
                            
                            if predictor is None:
                                st.error("Could not load prediction models!")
                            else:
                                # Select random image
                                selected_image_info = random.choice(relevant_images)
                                image_path = VERIFICATION_DIR / selected_image_info['file_name']
                                
                                if not image_path.exists():
                                    st.error(f"Image not found: {image_path}")
                                else:
                                    st.info(f"Selected image: {selected_image_info['file_name']}")
                                    
                                    # Load image
                                    image = cv2.imread(str(image_path))
                                    
                                    with st.spinner("Running prediction..."):
                                        try:
                                            # Get predictions
                                            predictions = predictor.predict(image_path)
                                            
                                            # Get ground truth annotations
                                            gt_annotations = get_annotations_for_image(
                                                data, selected_image_info['id']
                                            )
                                            
                                            # Store in session state for display
                                            st.session_state.current_image = image
                                            st.session_state.current_predictions = predictions
                                            st.session_state.current_gt = gt_annotations
                                            st.session_state.current_categories = data['categories']
                                            st.session_state.current_filename = selected_image_info['file_name']
                                            
                                        except Exception as e:
                                            st.error(f"Prediction failed: {e}")
                        
                        # Display prediction summary
                        if hasattr(st.session_state, 'current_predictions'):
                            st.subheader("Results Summary")
                            
                            pred_count = len(st.session_state.current_predictions)
                            gt_count = len(st.session_state.current_gt)
                            
                            st.metric("Predictions", pred_count)
                            st.metric("Ground Truth", gt_count)
                            
                            if st.session_state.current_predictions:
                                st.write("**Detected Species:**")
                                for i, pred in enumerate(st.session_state.current_predictions):
                                    st.write(f"• {pred['category']} ({pred['conf']:.2f})")
                    
                    with col2:
                        # Display visualization
                        if hasattr(st.session_state, 'current_image'):
                            st.subheader(f"{st.session_state.current_filename}")
                            
                            # Create visualization
                            fig = draw_bboxes_on_image(
                                st.session_state.current_image,
                                st.session_state.current_gt,
                                st.session_state.current_predictions,
                                st.session_state.current_categories
                            )
                            
                            st.pyplot(fig, use_container_width=True)
                            
                            # Legend
                            st.markdown("""
                            **Legend:**
                            -  **Green boxes**: Ground truth annotations
                            -  **Red boxes**: Model predictions  
                            - Numbers in parentheses show confidence scores
                            """)
                        else:
                            st.info(" Click 'Run Prediction' to see results")
            
            # -------- Render (no expander) --------
        st.markdown("###  How it works")
        st.markdown("""
        1. **Select Species** → choose an animal  
        2. **Run Prediction** → we pick a random image containing that species  
        3. **Compare** → Ground truth (green) vs Predictions (red), plus metrics below

        **Models:** MegaDetector v6 (detector) + ConvNeXt (species classifier)  
        **Confidence thresholds:** MegaDetector = 0.35, Classifier = 0.55
        """)

        st.markdown("###  Overall metrics")
        st.dataframe(
            overall_df.style.format({
                "accuracy": "{:.3f}",
                "macro_precision": "{:.3f}",
                "macro_recall": "{:.3f}",
                "macro_f1": "{:.3f}",
                "weighted_precision": "{:.3f}",
                "weighted_recall": "{:.3f}",
                "weighted_f1": "{:.3f}",
                "support": "{:.0f}",
            }),
            use_container_width=True
        )

        st.markdown("###  Per‑class report  — CIS")
        st.dataframe(
            cis_per_class_with_acc.style.format({
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }),
            use_container_width=True
        )

        st.markdown("###  Per‑class report  — TRANS")
        st.dataframe(
            trans_per_class_with_acc.style.format({
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }),
            use_container_width=True
        )


        st.markdown("###  Confusion matrices")
        c1, c2 = st.columns(2)
        with c1:
            st.image(metrics_dir / "verify_cis_test_confusion_matrix.png", caption="CIS Verification")
        with c2:
            st.image(metrics_dir / "verify_trans_test_confusion_matrix.png", caption="TRANS Verification")

                
    except Exception as e:
        st.error(f"Failed to load verification data: {e}")
        st.info("Please ensure your verification JSON files exist at the specified paths.")