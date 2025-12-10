# PatchTST - Clean Vectorized ML Architecture Diagram (ViT-style)

## Yêu cầu vẽ hình

**Mô tả:** Vẽ sơ đồ kiến trúc ML dạng vectorized, clean, minimal theo style Vision Transformer (ViT). Sơ đồ sạch sẽ, các khối rõ ràng, mũi tên thẳng, layout gọn gàng, không có chi tiết thừa.

**Font:** Roboto, kích thước chữ rõ ràng, style minimal và clean.

---

## Vectorized Architecture Diagram

```
Input → Data Prep → Optuna → PatchTST Model → Post-proc → Smooth → Output
                      Tuning   (with Best Params)            Correction
                                    
                    ┌──────────────────────────────┐
                    │   PatchTST Model            │
                    │   (with Best Params)        │
                    └──────────────────────────────┘
                                 │
                                 ▼
        RevIN → Patch → Embedding → Transformer Encoder → Output → RevIN Reverse → Baseline Predictions
                           (Attention + FFN)        (h=100)
                                 │
                                 ├──► Post-processing (Bias Correction)
                                 │
                                 └──► Smooth Correction (blends baseline + post-processed)
```

---


---

## Prompt cho AI vẽ hình (ViT-style Vectorized)

```
[LOẠI ẢNH] Technical architecture diagram, neural network diagram, machine learning flowchart

[CHỦ THỂ] PatchTST model architecture for time series forecasting

[HÀNH ĐỘNG] Display horizontal flow diagram showing exactly 5 rectangular blocks (Data Prep, Optuna, PatchTST Model, Post-processing, Smooth Correction) connected by rightward arrows, with PatchTST Model's internal architecture shown as a separate horizontal flow below the main block. **IMPORTANT: There is only ONE Smooth Correction block that receives two input arrows**

[ĐẶC ĐIỂM VẬT LÝ - LAYOUT]
- Horizontal layout: **Exactly 5 main 3D blocks** arranged horizontally from left to right: Block 1 (Data Preparation), Block 2 (Optuna Tuning), Block 3 (PatchTST Model), Block 4 (Post-processing), Block 5 (Smooth Correction). **There is only ONE Smooth Correction block (Block 5), not multiple blocks**
- Each block: 3D rectangular prism (isometric view), showing three visible faces (front, top, right side)
- Block dimensions: **ALL blocks MUST have identical dimensions** - Front face width 300px, height 100px; depth 20px (perspective view) - This creates a technical, uniform appearance. No block should be larger or smaller than others, ensuring visual consistency and professional technical diagram aesthetic
- Spacing: 40px horizontal gap between blocks
- Arrows between main blocks: Straight horizontal lines with arrowhead at right end, 2px thickness, color #424242, pointing rightward (left to right), connecting center-right of each block to center-left of next block
- **CRITICAL: There is ONLY ONE Smooth Correction block (Block 5)** - Do NOT create multiple Smooth Correction blocks
- Special connection for Smooth Correction: **TWO arrows feed into THE SAME Block 5 (Smooth Correction)** to show it receives inputs from both sources:
  * Arrow 1 (Baseline - direct connection): From Block 3 (PatchTST) right side to Block 5 left side, carrying Baseline Predictions - This can be shown as a curved arrow or diagonal line connecting from right side of Block 3 to left side of Block 5, bypassing Block 4, OR as a direct horizontal connection if layout allows. Arrow style: 2px thickness, color #424242, solid line (or dashed line #757575 to distinguish from main flow)
  * Arrow 2 (Post-processed - main flow): From Block 4 (Post-processing) right side to Block 5 left side, carrying Post-processed Predictions - This is the main horizontal flow arrow (2px thickness, color #424242, solid line)
- **BOTH arrows connect to THE SAME Block 5 (Smooth Correction)** - They both point to the left side of Block 5, indicating Smooth Correction receives and blends both inputs using weighted interpolation formula: pred_smooth = (1 - weights) * baseline + weights * post_processing
- Visual representation: Both arrows should be clearly visible converging into the same Block 5. Arrow 1 can be dashed or lighter color to distinguish from main flow, Arrow 2 is main flow solid line. The dual-input nature must be clear: two arrows, one block
- Background: Solid white color #FFFFFF
- 3D blocks have subtle isometric perspective, light shadow on right and bottom edges

[ĐẶC ĐIỂM VẬT LÝ - BLOCKS (3D STYLE)]
- Block 1 "Data Preparation": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Data Preparation"
- Block 2 "Optuna Tuning": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Optuna Tuning"
- Block 3 "PatchTST Model (with Best Params)": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 2px (thicker to emphasize importance, but block size remains identical), centered text "PatchTST Model (with Best Params)" only, no internal flow text
- Block 4 "Post-processing": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Post-processing" and "(Linear Regression)" below
- Note: Post-processing is trained using TimeSeriesSplit (3 folds) to learn bias pattern from validation folds, then applies correction formula: y_corrected = coef * y_predicted + intercept
- Block 5 "Smooth Correction": **THERE IS ONLY ONE Smooth Correction block** - 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Smooth Correction"
- **CRITICAL: Block 5 receives TWO arrows into the SAME block** - One arrow from Block 3 (Baseline Predictions) and one arrow from Block 4 (Post-processed Predictions). Both arrows connect to the left side of Block 5. There is NO second Smooth Correction block
- Note: Smooth Correction receives inputs from BOTH Block 3 (PatchTST Baseline Predictions) and Block 4 (Post-processing Post-processed Predictions), blending them using formula: pred_smooth = (1 - weights) * baseline + weights * post_processing
- This dual-input design is critical: Smooth Correction does NOT simply process Post-processing output, but actively blends both the original Baseline Predictions and the corrected Post-processed Predictions to create a smooth transition

[ĐẶC ĐIỂM VẬT LÝ - 3D BLOCK STYLE]
- Each 3D block shows isometric perspective (30-45 degree angle)
- **CRITICAL REQUIREMENT: All main flow blocks (Blocks 1-5) MUST have identical dimensions** - width 300px, height 100px, depth 20px - This uniform sizing creates a professional, technical diagram aesthetic. No block should be larger or smaller than others, ensuring visual consistency
- Front face: Main color (light forest green #B8E6C8)
- Top face: 20% darker shade of main color (forest green #A5D6A7)
- Right side face: 30% darker shade of main color (darker forest green #90C695)
- Edges: Thin black lines (1px) separating the three faces
- Shadow: Very subtle shadow on right and bottom edges (light gray #E0E0E0, 2px blur)

[ĐẶC ĐIỂM VẬT LÝ - PATCHTST MODEL INTERNAL FLOW]
- Below Block 3 "PatchTST Model", a separate horizontal flow diagram is displayed
- This flow shows the internal architecture steps as small 3D blocks or text boxes connected by arrows
- Flow sequence (left to right): "RevIN" → "Patch" → "Embedding" → "Transformer Encoder (Attention + FFN)" → "Output (h=100)" → "RevIN Reverse" → "Baseline Predictions"
- Each step is a small 3D rectangular prism or text box, light forest green fill #B8E6C8, with darker shades for 3D effect
- Steps are connected by horizontal arrows (1.5px thickness, color #424242) pointing rightward
- Flow is positioned directly below Block 3, centered or aligned with Block 3's center
- Vertical arrow (2px thickness, color #424242) connects from bottom center of Block 3 downward to the start of the internal flow
- Internal flow blocks (below PatchTST Model) are smaller: width 80-120px, height 40-50px each - These are separate from main flow blocks and serve only to show internal architecture details
- Spacing between internal flow blocks: 20px horizontal gap
- After internal flow completes at "Baseline Predictions", the output flows upward back to main flow:
  * Vertical arrow (2px thickness, color #424242) connects from "Baseline Predictions" (end of internal flow) upward to Block 3 output
  * From Block 3, Baseline Predictions split into two paths:
    - Path 1: Baseline Predictions → Block 4 (Post-processing) → Post-processed Predictions → Block 5 (Smooth Correction) via Arrow 2
    - Path 2: Baseline Predictions → Block 5 (Smooth Correction) directly via Arrow 1 (bypassing Block 4)
  * **BOTH paths converge at THE SAME Block 5 (Smooth Correction)** - There is only ONE Block 5 that receives both arrows. Do NOT create a second Smooth Correction block

[ĐẶC ĐIỂM VẬT LÝ - TEXT]
- Font: roboto, clean and modern (Roboto style)
- Text color: Dark gray #212121
- Title text: Bold, size 14pt, centered
- Body text: Regular, size 11pt, centered
- Nested block text: Regular, size 10pt, left-aligned

[ĐẶC ĐIỂM VẬT LÝ - COLORS]
- Background: White #FFFFFF
- Block borders: Gray #E0E0E0, 1px solid
- Main flow arrows: Dark gray #424242, 2px solid lines with arrowhead pointing rightward (left to right)
- Text: Dark gray #212121
- Block fills: Light forest green color scheme (#B8E6C8 main, #A5D6A7 top, #90C695 right) as specified above

[BỐI CẢNH]
- White background, no decorative elements
- Technical diagram style, similar to academic paper figures
- Clean, minimal, professional appearance

[CHI TIẾT BỔ SUNG]
- Input label at left: "Time Series"
- Output label at right: "Forecast"
- Blocks show only essential architecture information, no detailed learning descriptions
- PatchTST Model block is simplified - internal flow is shown separately below as a horizontal sequence of steps connected by arrows
- Flow logic (CRITICAL - must match blog exactly):
  * Step 1: PatchTST Model generates Baseline Predictions (output from internal flow)
  * Step 2: Post-processing is trained first using TimeSeriesSplit (3 folds) on training data to learn bias pattern from validation folds, collecting (prediction, ground_truth) pairs
  * Step 3: Post-processing applies bias correction to Baseline Predictions using Linear Regression: y_corrected = coef * y_predicted + intercept (e.g., y_corrected = 0.7267 * y_predicted + 9.3249), producing Post-processed Predictions
  * Step 4: Smooth Correction receives BOTH Baseline Predictions (from PatchTST Block 3) AND Post-processed Predictions (from Post-processing Block 4)
  * Step 5: Smooth Correction blends both inputs using weighted interpolation: pred_smooth = (1 - weights) * baseline + weights * post_processing
  * Step 6: Weights transition from 0 to 1 over first 20% of predictions (smooth transition), then remain at 1.0 for remaining 80% (direct post-processing)
  * Step 7: First value is kept unchanged: pred_smooth[0] = pred_baseline[0] (weight[0] = 0.0) - This preserves the first prediction value for reliability
- Legend in bottom left corner showing (optional, can be omitted for cleaner diagram):
  * 3D block symbol with label "Feature Map"
  * Arrow symbol with label "Data flow"

[PHONG CÁCH NGHỆ THUẬT]
- Technical diagram style
- Vector graphics style with 3D isometric perspective
- Academic paper figure style (similar to Pyramid Vision Transformer PVT architecture diagrams)
- Clean line art with 3D depth
- Minimalist design with subtle 3D effects
- Isometric 3D blocks showing dimensions (H×W×C style)
- Light shadows on 3D blocks for depth perception
- Similar to Vision Transformer (ViT) and Pyramid Vision Transformer (PVT) architecture diagrams in research papers
```

---

## Alternative: Horizontal Layout (Optional)

```
Input → Data Prep → Optuna → PatchTST Model → Post-proc → Smooth → Output
                      │              │                      │       │
                      │              ▼                      │       │
                      │    RevIN → Patch → Embedding → ... │       │
                      │              │                      │       │
                      │              ▼                      │       │
                      │    Baseline Predictions             │       │
                      │              │                      │       │
                      │              ├──► Post-processing  │       │
                      │              │    (Bias Correction) │       │
                      │              │         │            │       │
                      │              │         ▼            │       │
                      │              │    Post-processed    │       │
                      │              │         │            │       │
                      │              │         └────────────┤       │
                      │              │                      │       │
                      │              └──────────────────────┼───────┤
                      │                                     │       │
                      │                    Smooth Correction│       │
                      │                    (blends: baseline│      │
                      │                     + post-processed)       │
                      │
                      └─► [Internal Flow Below]
```

---

## Thông tin kỹ thuật

- **Input:** Time Series (T × 1)
- **Output:** Forecast (100 × 1)
- **What the Model Learns:**
  - **RevIN**: Distribution adaptation
  - **Patch Creation**: Local patterns (32)
  - **Patch Embedding**: Feature representations
  - **Position Encoding**: Temporal order
  - **Self-Attention**: Long-range dependencies
  - **FFN**: Non-linear transformations
  - **Post-processing**: Bias correction
  - **Smooth Correction**: Transition blending

