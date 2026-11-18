"""
scripts/report_builder/chapter5.py
----------------------------------
Chapter 5: Experiments and Model Development
- 5.1 Experimental Setup and Hardware Constraints (RTX 3050 4GB VRAM Cap)
- 5.2 Training Configurations and Optimization Details
- 5.3 Dataset 1 Experiments: SegFormer vs. U-Net Comparison
- 5.4 Dataset 1 Zero-Shot Cross-City Transfer (Pune Core, Hinjwadi, Chennai)
- 5.5 Dataset 2 Hierarchical Model Training Dynamics and Convergence
- 5.6 Vashi Controlled Model Progression (Baseline -> E1 -> E2 -> E3 -> E4)
- 5.7 Model Selection Protocol and Strict Decision Rules
"""

import docx
from docx.shared import Inches, Pt
import styles
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table
)

def add_chapter_5(doc):
    add_heading_1(doc, "Chapter 5: Experiments and Model Development")
    
    # ----------------------------------------------------
    # 5.1 Experimental Setup and Hardware Constraints
    # ----------------------------------------------------
    add_heading_2(doc, "5.1 Experimental Setup and Hardware Constraints")
    add_paragraph(doc, 
        "All deep learning experiments were executed locally on an NVIDIA GeForce RTX 3050 Laptop GPU equipped with 4,096 MB (4.0 GB) of GDDR6 VRAM. In academic and edge-computing environments, large server clusters with 80 GB A100 GPUs are frequently unavailable. Consequently, a core research priority was enforcing rigorous hardware compliance: ensuring models converge efficiently without triggering CUDA Out-of-Memory (OOM) faults or causing thermal throttling on commercial 4 GB hardware.",
        bold_prefix="Hardware Reality: "
    )
    add_paragraph(doc, 
        "To guarantee operational safety, a strict per-process memory fraction limit of 60% (max 2.457 GB) was programmatically enforced at initialization via PyTorch. Furthermore, a compute duty cycle pacing delay (0.012s sleep per iteration) was embedded in training loops, keeping GPU core temperatures stable below 68°C with GPU utilization capped at ~60%. Table 5.1 summarizes the hardware compliance parameters.",
        bold_prefix="Memory and Compute Governance: "
    )

    p_t51 = doc.add_paragraph()
    p_t51.paragraph_format.space_before = Pt(8)
    p_t51.paragraph_format.space_after = Pt(2)
    p_t51.paragraph_format.keep_with_next = True
    r_t51 = p_t51.add_run("Table 5.1: Hardware enforcement and compute duty cycle parameters (RTX 3050 Laptop GPU).")
    r_t51.font.name = "Times New Roman"
    r_t51.font.size = Pt(9.5)
    r_t51.font.bold = True
    r_t51.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    hw_data = [
        ["Total Physical VRAM", "4,096 MB (4.00 GB GDDR6)", "NVIDIA GeForce RTX 3050 Laptop GPU"],
        ["Programmatic Memory Limit", "2,457 MB (2.40 GB / 60%)", "Enforced via torch.cuda.set_per_process_memory_fraction(0.60)"],
        ["Operating Steady-State VRAM", "480 MB – 520 MB", "Continuous baseline footprint during training"],
        ["Operating Peak VRAM", "1,740 MB (1.70 GB / 70.8%)", "Maximum transient memory during backward pass execution"],
        ["Mini-Batch Size", "1 Tile (512 × 512 × 4 channels)", "Padded batch dimension to prevent OOM spikes"],
        ["Gradient Accumulation Steps", "4 Mini-Batches", "Effective mini-batch size = 4 before optimizer.step()"],
        ["Compute Duty Cycle Pacing", "12 ms pause per iteration", "Prevents continuous 100% thermal throttling (Temp < 68°C)"]
    ]
    add_styled_table(doc, 
        ["Operational Parameter", "Configured Value", "Technical Implementation & Rationale"], 
        hw_data, 
        col_widths=[2.0, 1.8, 2.7], 
        alignment=['L', 'C', 'L']
    )

    # ----------------------------------------------------
    # 5.2 Training Configurations and Optimization Details
    # ----------------------------------------------------
    add_heading_2(doc, "5.2 Training Configurations and Optimization Details")
    add_paragraph(doc, 
        "All models were trained using PyTorch with Mixed Precision (torch.cuda.amp.GradScaler) to accelerate computation and reduce tensor footprint. The AdamW optimizer (Loshchilov and Hutter 2017) was employed across all experiments, initialized with a base learning rate of 1e-4 and a weight decay coefficient of 1e-2. Learning rates were scheduled using Cosine Annealing with a minimum learning rate floor of 1e-6. Data augmentations were restricted to spatial operations that preserve spectral integrity: random horizontal and vertical flips, random 90° rotations, and random resized cropping (scale: 0.85 to 1.15). Photometric color jittering was strictly avoided because altering digital numbers distorts physical biophysical NDVI reflectance ratios.",
        bold_prefix="Optimization Hyperparameters: "
    )

    # ----------------------------------------------------
    # 5.3 Dataset 1 Experiments: SegFormer vs. U-Net Comparison
    # ----------------------------------------------------
    add_heading_2(doc, "5.3 Dataset 1 Experiments: SegFormer vs. U-Net Comparison")
    add_paragraph(doc, 
        "On Cartosat-1 2.5m panchromatic data, both SegFormer-B2 and UltraRoad ResNet-34 U-Net were trained under identical conditions on 512×512 tiles derived from Navi Mumbai and Mumbai Central scenes. Training converged over 25 epochs. SegFormer-B2 achieved 81.16% Overall Test Accuracy, while UltraRoad U-Net achieved 81.89%. Quantitative analysis revealed a profound architectural trade-off: SegFormer leveraged its global multi-head attention to effortlessly delineate expansive natural features (Water: 89.8% IoU, Vegetation: 78.5% IoU) with zero clutter. However, U-Net's direct high-resolution skip connections proved superior for compact built structures (63.6% IoU) and linear transport corridors (reaching 66.63% on the SpaceNet 10m buffer road metric).",
        bold_prefix="Complementary Strengths: "
    )

    # ----------------------------------------------------
    # 5.4 Dataset 1 Zero-Shot Cross-City Transfer
    # ----------------------------------------------------
    add_heading_2(doc, "5.4 Dataset 1 Zero-Shot Cross-City Transfer (Pune Core, Hinjwadi, Chennai)")
    add_paragraph(doc, 
        "To test true out-of-domain generalizability, the frozen SegFormer-B2 checkpoint (trained solely on Mumbai 2.5m data) was evaluated in a zero-shot cross-city transfer experiment across 30 completely unseen strategic patches (49.15 km² total urban area) extracted from 0.60m Cartosat-2E imagery over Pune City, Pune Hinjwadi, and Chennai. To preserve physical scale without retraining, Cartosat-2E tiles were downsampled from 2,133×2,133 windows to 512×512 via area averaging, matching the 2.50m ground coverage (1,280m × 1,280m per tile).",
        bold_prefix="Transfer Protocol: "
    )
    add_paragraph(doc, 
        "Remarkably, despite never having seen Pune or Chennai, SegFormer achieved up to 70.0% SpaceNet Road IoU on planned arterial transit corridors (e.g., pune_hinjwadi_10_toll: 70.0%, pune_hinjwadi_05_wakad: 69.3%, pune_city_01_core: 68.8%). The hierarchical transformer encoder successfully transferred abstract topological corridor features across metropolitan domains without fine-tuning.",
        bold_prefix="Zero-Shot Findings: "
    )

    # ----------------------------------------------------
    # 5.5 Dataset 2 Hierarchical Model Training Dynamics
    # ----------------------------------------------------
    add_heading_2(doc, "5.5 Dataset 2 Hierarchical Model Training Dynamics and Convergence")
    add_paragraph(doc, 
        "On Cartosat-2E 0.60m multispectral imagery, the Unified Hybrid SegFormer + UltraRoad Attention U-Net was trained using the In-Epoch Decoupled Two-Step Hierarchical pipeline. Table 5.2 records the loss and validation IoU convergence trajectory across training epochs.",
        bold_prefix="Hierarchical Convergence: "
    )

    p_t52 = doc.add_paragraph()
    p_t52.paragraph_format.space_before = Pt(8)
    p_t52.paragraph_format.space_after = Pt(2)
    p_t52.paragraph_format.keep_with_next = True
    r_t52 = p_t52.add_run("Table 5.2: In-epoch training loss dynamics and subclass convergence trajectory.")
    r_t52.font.name = "Times New Roman"
    r_t52.font.size = Pt(9.5)
    r_t52.font.bold = True
    r_t52.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    dyn_data = [
        ["Epoch 1", "1.3251", "1.0140", "0.2592", "39.49%", "64.60%", "49.93%", "0.00%", "57.27%", "Saved Best (score: 38.84%)"],
        ["Epoch 2", "1.2960", "1.0122", "0.2365", "39.49%", "57.90%", "49.65%", "0.00%", "53.78%", "Maintained Best"],
        ["Epoch 3", "1.2770", "1.0130", "0.2200", "39.49%", "64.90%", "51.25%", "0.65%", "58.08%", "Saved Best (score: 39.21%)"],
        ["Epoch 4", "1.2584", "1.0170", "0.2012", "39.49%", "70.78%", "47.06%", "1.43%", "58.92%", "Saved Best (score: 39.62%)"],
        ["Held-Out Test (TTA)", "—", "—", "—", "55.65%", "73.22%", "50.33%", "1.53%", "61.78%", "Final Best Model (best_hybrid.pth)"]
    ]
    add_styled_table(doc, 
        ["Training Epoch", "Total Loss", "Step 1 Loss (5C)", "Step 2 Loss (Sub)", "Built-Up Buffer IoU", "Residential IoU", "Industrial IoU", "Slum IoU", "Active Subclass Mean", "Model Selection Action"], 
        dyn_data, 
        col_widths=[1.0, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 0.7, 1.0, 1.3], 
        alignment=['C', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L']
    )
    add_paragraph(doc, 
        "As documented in Table 5.2, Step 2 subclass loss steadily declined from 0.2592 to 0.2012. On the 186 held-out test patches, Residential Built-up IoU reached an outstanding 73.22% (exceeding the 50% target by +23.22%), and Industrial Built-up reached 50.33%, yielding an Active Subclass Mean of 61.78% (+11.78% over target).",
        bold_prefix="Subclass Target Exceeded: "
    )

    # ----------------------------------------------------
    # 5.6 Vashi Controlled Model Progression (Baseline to E4)
    # ----------------------------------------------------
    add_heading_2(doc, "5.6 Vashi Controlled Model Progression (Baseline -> E1 -> E2 -> E3 -> E4)")
    add_paragraph(doc, 
        "On Dataset 3 (Vashi SPOT 4-Band & Mumbai Pléiades), an intensive series of controlled, single-variable experimental iterations was executed to systematically diagnose and resolve persistent segmentation failures. Table 5.3 presents the chronological progression from the original baseline through Experiment E4.",
        bold_prefix="Disciplined Progression: "
    )

    p_t53 = doc.add_paragraph()
    p_t53.paragraph_format.space_before = Pt(8)
    p_t53.paragraph_format.space_after = Pt(2)
    p_t53.paragraph_format.keep_with_next = True
    r_t53 = p_t53.add_run("Table 5.3: Vashi controlled experimental progression (Baseline to Experiment E4).")
    r_t53.font.name = "Times New Roman"
    r_t53.font.size = Pt(9.5)
    r_t53.font.bold = True
    r_t53.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    vprog_data = [
        ["Baseline (Original)", "ResNet-18 U-Net", "43 Vashi + 13 Mumbai", "CE Loss", "58.56%", "34.36%", "47.19%", "44.09%", "Severe Road overprediction (48.2% road pred)"],
        ["Retrained v1", "ResNet-18 U-Net", "Same Pool", "CE + Soft Dice", "57.38%", "33.65%", "46.12%", "43.50%", "Minor degradation; road bloat persisted"],
        ["Corrected v2", "Deep ResNet-18", "43 Vashi + 13 Mum + 68 Mend", "Unweighted CE", "51.77%", "27.32%", "38.20%", "35.10%", "Severe regression; Mendeley data dominated features"],
        ["Corrected v3", "ResNet-18 + scSE", "Same Mixed Pool", "clDice + Lovász", "51.59%", "27.72%", "40.14%", "37.15%", "Mendeley removed, but partial-label conflict persisted"],
        ["Experiment E1 (Reference)", "ResNet18 + scSE Dual Stem", "43 Vashi + 13 Mumbai (Clean)", "Masked CE + Dice", "75.01%", "61.40%", "59.43%", "42.32%", "Major breakthrough: Highest global Val mIoU (61.40%)"],
        ["Experiment E2 (Hard-Mining)", "Same as E1 Architecture", "Hard-Mining Sampler [0.35x-2.5x]", "Masked CE + Dice", "62.47%", "50.15%", "38.52%", "38.20%", "REJECTED: Oversampled shadow label noise; collapsed mIoU"],
        ["Experiment E3 (Boundary Gate)", "BoundaryGuidedHighResUNet", "dec1 Gated Shallow Skip", "Masked CE + Dice", "66.08%", "53.38%", "44.20%", "40.60%", "REJECTED: Gating suppressed low-contrast building roofs"],
        ["Experiment E4 (NDVI + FocalDice)", "HighResNDVIAdaptiveUNet", "5-Band (RGB+NIR+NDVI)", "Focal-Dice (γ=1.5)", "66.22%", "53.19%", "44.47%", "40.14%", "Targeted Breakthrough: Solved road bloat on vashi_0041"]
    ]
    add_styled_table(doc, 
        ["Experiment Iteration", "Architecture Mechanism", "Training Dataset Split", "Loss Formulation", "Val Accuracy", "4-Class mIoU", "Non-Slum IoU", "Road IoU", "Diagnostic Outcome & Technical Assessment"], 
        vprog_data, 
        col_widths=[1.2, 1.3, 1.2, 1.0, 0.7, 0.7, 0.7, 0.7, 1.5], 
        alignment=['L', 'L', 'L', 'L', 'R', 'R', 'R', 'R', 'L']
    )

    add_heading_3(doc, "Root-Cause Analysis of Failed Ablations (E2 and E3)")
    add_paragraph(doc, 
        "1. Why Hard-Mining Sampling (E2) Failed: In Experiment E2, we hypothesized that oversampling hard patches (vashi_0020, vashi_0011, vashi_0049) with sampling weights up to 2.50x would force the network to learn narrow boundaries. Instead, validation accuracy dropped precipitously by -12.54 percentage points (75.01% -> 62.47%) and Non-slum IoU collapsed from 59.43% to 38.52%. Forensic analysis revealed that these 'hard' patches contained inherent label noise: building shadows were inconsistently partitioned (40% labeled Road vs. 45% labeled Non-slum), and vector boundaries suffered from 1.0x to 1.2x pixel alignment offsets. Oversampling amplified this noise, destabilizing clean open land and building representations.",
        bold_prefix="Ablation E2 Post-Mortem: "
    )
    add_paragraph(doc, 
        "2. Why Guided Boundary Attention (E3) Regressed: In Experiment E3, deep semantic features from dec1 (64 channels) were used to dynamically gate the native shallow spatial skip (sh_feat, 32 channels) via an attention gate. While this successfully sharpened edges on high-contrast corridors, the sigmoid gating mechanism over-suppressed untextured concrete flat roofs where contrast against surrounding paving was subtle. Consequently, Non-slum Building IoU degraded by -15.23 percentage points (59.43% -> 44.20%).",
        bold_prefix="Ablation E3 Post-Mortem: "
    )

    # ----------------------------------------------------
    # 5.7 Model Selection Protocol and Strict Decision Rules
    # ----------------------------------------------------
    add_heading_2(doc, "5.7 Model Selection Protocol and Strict Decision Rules")
    add_paragraph(doc, 
        "To uphold scientific discipline, Phase 8 enforced a strict model selection protocol:\n"
        "A candidate model may only supersede the reference checkpoint (E1) if it improves Overall Accuracy, 4-class mIoU, Boundary F1, and per-class IoU WITHOUT introducing substantial regressions in other classes. Any candidate introducing regressions in other classes must be rejected as the universal baseline.",
        bold_prefix="Governance Protocol: "
    )
    add_paragraph(doc, 
        "Under this rule, Experiment E1 remains our Primary Universal Reference Checkpoint (75.01% Val Accuracy, 61.40% mIoU, 66.88% BF1). Experiment E4 (5-band NDVI + Focal-Dice), while achieving an outstanding localized breakthrough on commercial conflict sectors like vashi_0041 (boosting patch accuracy from 68.59% to 70.95% and Non-slum IoU from 61.54% to 67.11%, while reducing road bloat from 46.11% to 31.70%), experienced an asymmetric trade-off on global bare terrain (Open Land IoU dropped from 67.27% to 54.46% because bare soil and flat concrete roofs share identical flat NDVI). Therefore, E4 is retained as the Specialized High-Density Urban Model, while E1 remains the Global Baseline.",
        bold_prefix="Final Governance Decision: "
    )
    
    doc.add_page_break()
