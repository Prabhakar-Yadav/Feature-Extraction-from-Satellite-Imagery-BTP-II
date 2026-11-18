"""
scripts/report_builder/chapter4.py
----------------------------------
Chapter 4: Methodology and Model Architecture
- 4.1 Overall Architectural Philosophy
- 4.2 Dataset 1 Architecture: SegFormer-B2 vs. ResNet-34 U-Net
- 4.3 Dataset 2 Architecture: Unified Hybrid SegFormer + UltraRoad Attention U-Net
- 4.4 In-Epoch Decoupled Two-Step Hierarchical Joint Optimization
- 4.5 Physics-Based Asphalt-Safe Guard
- 4.6 Composite Loss Formulations (Focal, Lovász-Softmax, Focal-Dice)
"""

import docx
from docx.shared import Inches, Pt
import styles
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table, add_equation
)
from omml_equations import (
    add_display_equation, OMML_FUSION, OMML_GATING, 
    OMML_ASPHALT_GUARD, OMML_LMACRO, OMML_LVASHI
)

def add_chapter_4(doc):
    add_heading_1(doc, "Chapter 4: Methodology and Model Architecture")
    
    # ----------------------------------------------------
    # 4.1 Overall Architectural Philosophy
    # ----------------------------------------------------
    add_heading_2(doc, "4.1 Overall Architectural Philosophy")
    add_paragraph(doc, 
        "The overarching architectural strategy of this project is driven by two empirical principles established through remote sensing observation: (1) no single neural paradigm is universally optimal for all urban features—transformers excel at broad natural context while CNNs excel at sub-meter edge preservation; and (2) urban land-cover cannot be treated as a flat, single-tier classification problem—functional subclasses (such as informal slums) must be constrained by biophysical macro envelopes to eliminate false detections.",
        bold_prefix="Dual Principles: "
    )
    add_paragraph(doc, 
        "Consequently, the methodology bifurcates into tailored architectures: a comparative study of pure Vision Transformers versus deep CNNs on single-band 2.5m Cartosat-1 data, and an integrated Hybrid Hierarchical Network combining SegFormer-B2, UltraRoad Attention U-Net, and scSE Subclass Morphology Heads on 0.60m multispectral Cartosat-2E imagery.",
        bold_prefix="Methodological Structuring: "
    )

    # ----------------------------------------------------
    # 4.2 Dataset 1 Architecture: SegFormer-B2 vs. ResNet-34 U-Net
    # ----------------------------------------------------
    add_heading_2(doc, "4.2 Dataset 1 Architecture: SegFormer-B2 vs. ResNet-34 U-Net")
    add_paragraph(doc, 
        "To rigorously investigate how architectural priors interact with single-band panchromatic imagery, two distinct networks were engineered to ingest 1-channel 512×512 patches and predict 5 macro classes:",
        bold_prefix="Comparative Setup: "
    )
    
    add_heading_3(doc, "Architecture A: Road-Enhanced SegFormer-B2")
    add_paragraph(doc, 
        "SegFormer-B2 utilizes a 4-stage hierarchical Mix Transformer (MiT-B2) encoder backbone. The input 1-band patch is mapped into multi-scale feature hierarchies via overlapped patch merging operations with receptive fields scaling from H/4 to H/32. Stage 1 operates with embedding dimension C1=64, Stage 2 with C2=128, Stage 3 with C3=320, and Stage 4 with C4=512. Spatial sequence lengths in self-attention modules are downsampled via reduction ratios R = [8, 4, 2, 1], reducing computational complexity from quadratic to manageable sub-quadratic regimes. The All-MLP Decoder unifies multi-level channels to a uniform dimension of C=256, applies bilinear upsampling (4x), and projects features through a 1×1 convolution into 5-class logits.",
        bold_prefix="SegFormer Specifications: "
    )

    add_heading_3(doc, "Architecture B: ResNet-34 UltraRoad Attention U-Net")
    add_paragraph(doc, 
        "The CNN baseline utilizes a deep ResNet-34 encoder with skip connections routed to a symmetric decoder. To preserve high-frequency road centerlines, an Atrous Spatial Pyramid Pooling (ASPP) module with dilation rates [1, 6, 12, 18] is placed at the bottleneck to capture multi-scale context without reducing spatial resolution. Each decoder upsampling block (32x, 16x, 8x, 4x, 2x, 1x) incorporates a Spatial and Channel Squeeze-and-Excitation (scSE) attention gate, which dynamically re-weights skip feature channels based on global boundary saliency before concatenation.",
        bold_prefix="UltraRoad U-Net Specifications: "
    )

    # Table 4.1: Architecture Specs
    p_t41 = doc.add_paragraph()
    p_t41.paragraph_format.space_before = Pt(8)
    p_t41.paragraph_format.space_after = Pt(2)
    p_t41.paragraph_format.keep_with_next = True
    r_t41 = p_t41.add_run("Table 4.1: Architectural hyperparameters across Dataset 1 and Dataset 2 models.")
    r_t41.font.name = "Times New Roman"
    r_t41.font.size = Pt(9.5)
    r_t41.font.bold = True
    r_t41.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    arch_data = [
        ["SegFormer-B2", "Cartosat-1 (2.5m PAN)", "MiT-B2 Transformer", "All-MLP Decoder", "24.7 Million", "Efficient Self-Attention (R=[8,4,2,1])", "O(N²/R)"],
        ["UltraRoad U-Net", "Cartosat-1 (2.5m PAN)", "ResNet-34 CNN", "scSE Attention Decoder", "21.5 Million", "ASPP Bridge + scSE Skip Connections", "O(K²·N)"],
        ["Unified Hybrid Model", "Cartosat-2E (0.60m 4-Band)", "Parallel SegFormer-B2 + UltraRoad U-Net", "Dual Decoder + Subclass Morphology Head", "46.2 Million", "Learned Dynamic Gating (α) + Decoupled Backward", "Hybrid Linear"],
        ["Vashi Sensor-Adaptive", "Vashi SPOT & Pléiades", "Dual-Stem ResNet-18", "Native Shallow Skip dec0", "14.3 Million", "Source-Aware Conditional Softmax", "O(N)"]
    ]
    add_styled_table(doc, 
        ["Model Name", "Dataset Domain", "Encoder Backbone", "Decoder Mechanism", "Parameter Count", "Specialized Feature Processing", "Computational Complexity"], 
        arch_data, 
        col_widths=[1.2, 1.2, 1.4, 1.2, 0.9, 1.4, 0.8], 
        alignment=['L', 'L', 'L', 'L', 'R', 'L', 'C']
    )

    # ----------------------------------------------------
    # 4.3 Dataset 2 Architecture: Unified Hybrid SegFormer + UltraRoad U-Net
    # ----------------------------------------------------
    add_heading_2(doc, "4.3 Dataset 2 Architecture: Unified Hybrid SegFormer + UltraRoad Attention U-Net")
    add_paragraph(doc, 
        "In sub-meter multispectral imagery (0.60m GSD), pure CNNs frequently confuse large barren fields with paved industrial yards due to limited receptive fields, whereas pure Vision Transformers blur narrow residential alleyways and shack boundaries due to patch tokenization. To exploit the strengths of both paradigms while eliminating their weaknesses, we designed the Unified Hybrid Model:",
        bold_prefix="The Hybrid Rationale: "
    )
    add_bullet(doc, 
        "Parallel Dual-Stream Feature Extraction: 4-band multispectral imagery (RGB + NIR) is fed simultaneously into parallel SegFormer-B2 and UltraRoad U-Net backbones.",
        bold_prefix="1. Dual Encoders: "
    )
    add_bullet(doc, 
        "Dynamic Learned Gating Coefficient (α_macro): The 5-class macro logits from both streams are fused via a learned scalar gating parameter α_macro initialized to 0.50 and constrained via a sigmoid activation according to Equation (4.1):",
        bold_prefix="2. Dynamic Fusion: "
    )
    add_display_equation(doc, OMML_FUSION, "4.1")
    add_paragraph(doc, 
        "This dynamic parameterization allows the network to automatically balance global transformer context against local CNN boundary sharpness during training.",
        bold_prefix=""
    )
    add_bullet(doc, 
        "Subclass Morphology Head: To resolve fine-grained built-up subclasses, a specialized convolutional head ingests multi-scale fused features, the continuous built-up probability map P_builtup = Softmax(Logits_5c)[:, 3], and the raw 4-band input, refining features through two 3×3 conv blocks, scSE attention, and a 1×1 projection into 3 subclass logits.",
        bold_prefix="3. Subclass Head: "
    )

    # ----------------------------------------------------
    # 4.4 In-Epoch Decoupled Two-Step Hierarchical Joint Optimization
    # ----------------------------------------------------
    add_heading_2(doc, "4.4 In-Epoch Decoupled Two-Step Hierarchical Joint Optimization")
    add_paragraph(doc, 
        "To satisfy the core requirement that the model must train macro 5-class land-cover, predict built-up areas, and in the SAME epoch train 3 subclasses strictly inside the predicted built-up region, we formulated an in-epoch two-step joint optimization pipeline:",
        bold_prefix="Pipeline Formulation: "
    )

    add_heading_3(doc, "Step 1: Macro 5-Class Training and Spatial Gating Mask Generation")
    add_paragraph(doc, 
        "The mini-batch (1 × 4 × 512 × 512) is passed through both parallel encoders to predict macro 5-class logits Z_5c. The loss is computed against ground-truth macro labels Y_5c using a composite Focal + Lovász loss (Eq. 4.4). From Z_5c, the binary predicted built-up mask M_builtup is extracted according to Equation (4.2):",
        bold_prefix=""
    )
    add_display_equation(doc, OMML_GATING, "4.2")
    add_paragraph(doc, 
        "The mask M_builtup is detached from the PyTorch computational graph to prevent subclass gradients from destabilizing the macro encoder weights.",
        bold_prefix=""
    )

    add_heading_3(doc, "Step 2: Subclass Training Inside Predicted Built-Up Space")
    add_paragraph(doc, 
        "The subclass morphology head receives the detached features and predicts 3-subclass logits Z_sub. Ground-truth subclass targets Y_sub (Residential: 0, Industrial: 1, Slum: 2) are dynamically gated by the predicted built-up mask M_builtup:\n"
        "Y_gated(x, y) = Y_sub(x, y)  if (M_builtup(x, y) == 1  and  Y_sub(x, y) ∈ {0, 1, 2}),  else  255 (ignore_index)\n"
        "Loss_sub is computed strictly over non-ignored pixels using Masked Focal + Masked Lovász-Softmax loss.",
        bold_prefix=""
    )

    add_heading_3(doc, "Step 3: Unified Hierarchical 7-Class Inference")
    add_paragraph(doc, 
        "During evaluation and inference, a deterministic hierarchical decision tree integrates Step 1 and Step 2 outputs into a unified 7-class thematic map:\n"
        "• For pixels where M_builtup == 0: Semantic class is assigned directly from Step 1 macro prediction (Water: 0, Vegetation: 1, Barren Land: 2, Road Network: 6).\n"
        "• For pixels where M_builtup == 1: Semantic class is assigned from Step 2 subclass prediction (Residential: 3, Industrial: 4, Slum: 5).\n"
        "Result: This spatial gating rule guarantees ZERO false positive slums inside water bodies, agricultural fields, or forests.",
        bold_prefix=""
    )

    add_callout(doc, 
        "MEMORY DECOUPLING INNOVATION: PREVENTING CUDA OOM ON 4 GB GPU", 
        "Hardware Challenge: If backpropagation is performed on Loss_total = Loss_5c + Loss_sub simultaneously, PyTorch must retain the complete activation graphs of both SegFormer-B2 and UltraRoad U-Net alongside the subclass head in VRAM. This consumes >3.2 GB of GPU memory, instantly triggering CUDA Out-of-Memory (OOM) crashes on a 4 GB laptop GPU.\n"
        "Algorithmic Solution: We decoupled the backward passes in-epoch:\n"
        "1. Execute scaler.scale(Loss_5c / accum).backward() immediately. This updates the macro backbones and instantly FREES their intermediate activation memory.\n"
        "2. Forward-propagate through the lightweight subclass head with detached features.\n"
        "3. Execute scaler.scale(Loss_sub / accum).backward() on the subclass head.\n"
        "Result: Peak GPU memory dropped from >3.2 GB to 1.74 GB, with a steady-state footprint of 490 MB—strictly satisfying our 2.4 GB (60%) hardware safety cap."
    )

    # ----------------------------------------------------
    # 4.5 Physics-Based Asphalt-Safe Guard
    # ----------------------------------------------------
    add_heading_2(doc, "4.5 Physics-Based Asphalt-Safe Guard")
    add_paragraph(doc, 
        "During early training rounds, an aggressive heuristic water-boosting threshold inadvertently reclassified dark asphalt roads (R ≈ G ≈ B ≈ 40) and deep cast shadows as water bodies. Because asphalt exhibits low visible reflectance, naive spectral rules confused it with deep turbid water.",
        bold_prefix="The Asphalt-Water Confusion Bug: "
    )
    add_paragraph(doc, 
        "To permanently resolve this failure mode, we engineered the Asphalt-Safe Guard, a physical-deep learning hybrid gating rule. A pixel is permitted to be classified as Water if and only if it satisfies both physical water absorption criteria AND deep learning model confidence priors according to Equation (4.3):",
        bold_prefix="Dual Gating Rule: "
    )
    add_display_equation(doc, OMML_ASPHALT_GUARD, "4.3")
    add_paragraph(doc, 
        "If deep learning logits predict road or building probabilities exceeding 0.35, the water classification is vetoed. Furthermore, small isolated water puddles under 800 pixels (~288 m² on the ground) are suppressed as shadow artifacts. This rule completely eradicated false water on transportation networks, preserving road continuity.",
        bold_prefix=""
    )

    # ----------------------------------------------------
    # 4.6 Composite Loss Formulations
    # ----------------------------------------------------
    add_heading_2(doc, "4.6 Composite Loss Formulations")
    add_paragraph(doc, 
        "To balance pixel-wise discrimination and structural overlap, the macro 5-class loss synthesizes Focal Loss (gamma=2.0) and Lovász-Softmax according to Equation (4.4):",
        bold_prefix="Macro Loss: "
    )
    add_display_equation(doc, OMML_LMACRO, "4.4")
    add_paragraph(doc, 
        "For Dataset 3 (Vashi SPOT), where road overprediction into commercial roofs was the primary failure mode, we formulated Calibrated Focal-Dice Loss with class reweighting (Non-slum=1.25, Road=0.65, Veg=1.10, Open Land=1.05) as defined in Equation (4.5):",
        bold_prefix="Calibrated Focal-Dice Loss: "
    )
    add_display_equation(doc, OMML_LVASHI, "4.5")
    add_paragraph(doc, 
        "Setting gamma=1.5 downweights well-classified road segments, while reducing the Road weight to 0.65 penalizes false-positive road widening into adjacent buildings.",
        bold_prefix=""
    )
    
    doc.add_page_break()
