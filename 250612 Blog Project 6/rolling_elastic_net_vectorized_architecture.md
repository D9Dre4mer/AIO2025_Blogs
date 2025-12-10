# Rolling Elastic Net - Clean Vectorized ML Architecture Diagram (ViT-style)

## Yêu cầu vẽ hình

**Mô tả:** Vẽ sơ đồ kiến trúc ML dạng vectorized, clean, minimal theo style Vision Transformer (ViT) cho phương pháp Rolling Elastic Net. Sơ đồ sạch sẽ, các khối rõ ràng, mũi tên thẳng, layout gọn gàng, không có chi tiết thừa.

**Font:** Roboto, kích thước chữ rõ ràng, style minimal và clean.

---

## Vectorized Architecture Diagram

```
Input → Preprocess → Feature Eng → Normalize → Rolling ElasticNet → Eval → Forecast → Output

                                    ┌──────────────────────────────┐
                                    │   Rolling ElasticNet         │
                                    └──────────────────────────────┘
                                                │
                                                ▼
                    Window Selection → Check Variance → ElasticNet Train → Prediction
```

---

## Prompt cho AI vẽ hình (ViT-style Vectorized)

```
[LOẠI ẢNH] Technical architecture diagram, machine learning pipeline flowchart, data processing diagram

[CHỦ THỂ] Rolling Elastic Net model architecture for time series forecasting

[HÀNH ĐỘNG] Display horizontal flow diagram showing exactly 6 rectangular blocks (Data Preprocessing, Feature Engineering, Normalization, Rolling ElasticNet, Evaluation, Forecast 100 days) connected by rightward arrows, with Rolling ElasticNet's internal architecture shown as a separate horizontal flow below the main block

[ĐẶC ĐIỂM VẬT LÝ - LAYOUT]
- Horizontal layout: 6 main 3D blocks arranged horizontally from left to right
- Each block: 3D rectangular prism (isometric view), showing three visible faces (front, top, right side)
- Block dimensions: **ALL blocks MUST have identical dimensions** - Front face width 300px, height 100px; depth 20px (perspective view) - This creates a technical, uniform appearance. No block should be larger or smaller than others, ensuring visual consistency and professional technical diagram aesthetic
- Spacing: 40px horizontal gap between blocks
- Arrows between main blocks: Straight horizontal lines with arrowhead at right end, 2px thickness, color #424242, pointing rightward (left to right), connecting center-right of each block to center-left of next block
- Background: Solid white color #FFFFFF
- 3D blocks have subtle isometric perspective, light shadow on right and bottom edges

[ĐẶC ĐIỂM VẬT LÝ - BLOCKS (3D STYLE)]
- Block 1 "Data Preprocessing": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Data Preprocessing"
- Block 2 "Feature Engineering": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Feature Engineering"
- Block 3 "Normalization": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Normalization"
- Block 4 "Rolling ElasticNet": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 2px (thicker to emphasize importance, but block size remains identical), centered text "Rolling ElasticNet" only, no internal flow text
- Block 5 "Evaluation": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Evaluation"
- Block 6 "Forecast 100 days": 3D rectangular prism, **SAME dimensions as all other blocks** (width 300px, height 100px, depth 20px), light forest green fill #B8E6C8 on front face, darker forest green #A5D6A7 on top face, darker forest green #90C695 on right face, black border 1px, centered text "Forecast 100 days"

[ĐẶC ĐIỂM VẬT LÝ - 3D BLOCK STYLE]
- Each 3D block shows isometric perspective (30-45 degree angle)
- **CRITICAL: All main flow blocks (Blocks 1-6) MUST have identical dimensions** - width 300px, height 100px, depth 20px - This uniform sizing creates a professional, technical diagram aesthetic
- Front face: Main color (light forest green #B8E6C8)
- Top face: 20% darker shade of main color (forest green #A5D6A7)
- Right side face: 30% darker shade of main color (darker forest green #90C695)
- Edges: Thin black lines (1px) separating the three faces
- Shadow: Very subtle shadow on right and bottom edges (light gray #E0E0E0, 2px blur)

[ĐẶC ĐIỂM VẬT LÝ - ROLLING ELASTICNET INTERNAL FLOW]
- Below Block 4 "Rolling ElasticNet", a separate horizontal flow diagram is displayed
- This flow shows the internal architecture steps as small 3D blocks or text boxes connected by arrows
- Flow sequence (left to right): "Window Selection" → "Check Variance" → "ElasticNet Training" → "Prediction"
- Each step is a small 3D rectangular prism or text box, light forest green fill #B8E6C8, with darker shades for 3D effect
- Steps are connected by horizontal arrows (1.5px thickness, color #424242) pointing rightward
- Flow is positioned directly below Block 4, centered or aligned with Block 4's center
- Vertical arrow (2px thickness, color #424242) connects from bottom center of Block 4 downward to the start of the internal flow
- Internal flow blocks (below Rolling ElasticNet) are smaller: width 80-120px, height 40-50px each - These are separate from main flow blocks and serve only to show internal architecture details
- Spacing between internal flow blocks: 20px horizontal gap
- After internal flow completes at "Prediction", the output flows upward back to main flow via vertical arrow to Block 4 output

[ĐẶC ĐIỂM VẬT LÝ - TEXT]
- Font: Roboto, clean and modern (Roboto style)
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
- Input label at left: "OHLCV Data"
- Output label at right: "Forecast"
- Flow logic (CRITICAL - must match blog exactly):
  * Step 1: Data Preprocessing - Calculate log-return and log-volume change, then winsorize outliers
  * Step 2: Feature Engineering - Build lag features, technical indicators, calendar features
  * Step 3: Normalization - Apply StandardScaler on training data, split train/validation
  * Step 4: Rolling ElasticNet - For each time point i ≥ window_size:
    - Window Selection: Choose sliding window (i-w, i) or expanding window (0, i)
    - Check Variance: Skip if y_train has insufficient variance (unique values < 2)
    - ElasticNet Training: Fit new ElasticNet model on X_train, y_train with alpha and l1_ratio
    - Prediction: Predict X[i] and store in preds[i]
  * Step 5: Evaluation - Evaluate on return-space and price-space, calibrate predictions
  * Step 6: Forecast 100 days - Use final model to recursively predict 100 steps ahead

[PHONG CÁCH NGHỆ THUẬT]
- Technical diagram style
- Vector graphics style with 3D isometric perspective
- Academic paper figure style (similar to Vision Transformer ViT architecture diagrams)
- Clean line art with 3D depth
- Minimalist design with subtle 3D effects
- Light shadows on 3D blocks for depth perception
```

---

## Thông tin kỹ thuật

- **Input:** OHLCV Data (T × 5 features)
- **Output:** Forecast (100 × 1)
- **What the Model Learns:**
  - **Data Preprocessing**: Return patterns, outliers
  - **Feature Engineering**: Temporal, technical, seasonal patterns
  - **Normalization**: Distribution adaptation
  - **Rolling Window**: Adaptive time windows
  - **ElasticNet**: Feature importance, regularization
  - **Evaluation**: Error patterns, calibration
  - **Multi-step Forecasting**: Long-term dependencies

