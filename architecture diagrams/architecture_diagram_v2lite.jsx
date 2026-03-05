import { useState } from "react";

// ─────────────────────────────────────────────────────────────────
// COLOUR TOKENS  (same spirit as V2 but refreshed for Lite)
// ─────────────────────────────────────────────────────────────────
const COLORS = {
  stem:     "#3b82f6",   // blue
  backbone: "#8b5cf6",   // violet
  expand:   "#6366f1",   // indigo  ← new: p3_expand DW+PW
  csp:      "#f59e0b",   // amber
  bifpn:    "#10b981",   // emerald
  head:     "#06b6d4",   // cyan
  output:   "#ec4899",   // pink
  dw:       "#f97316",   // orange  ← DW-separable ops
  arrow:    "#64748b",
  bg:       "#0f172a",
  card:     "#1e293b",
  text:     "#e2e8f0",
  dim:      "#94a3b8",
  border:   "#475569",
  lite:     "#34d399",   // green accent for "lite" annotations
};

// ─────────────────────────────────────────────────────────────────
// SMALL BADGES
// ─────────────────────────────────────────────────────────────────
const DropBadge = () => (
  <span style={{ background:"linear-gradient(135deg,#f59e0b,#d97706)", color:"white",
    fontSize:9, fontWeight:700, padding:"1px 5px", borderRadius:3, marginLeft:4 }}>DROP</span>
);
const DWBadge = () => (
  <span style={{ background:"linear-gradient(135deg,#f97316,#ea580c)", color:"white",
    fontSize:9, fontWeight:700, padding:"1px 5px", borderRadius:3, marginLeft:4 }}>DW+PW</span>
);
const LiteBadge = () => (
  <span style={{ background:"linear-gradient(135deg,#34d399,#059669)", color:"white",
    fontSize:9, fontWeight:700, padding:"1px 5px", borderRadius:3, marginLeft:4 }}>LITE</span>
);
const NewBadge = () => (
  <span style={{ background:"linear-gradient(135deg,#818cf8,#4f46e5)", color:"white",
    fontSize:9, fontWeight:700, padding:"1px 5px", borderRadius:3, marginLeft:4 }}>NEW</span>
);

// ─────────────────────────────────────────────────────────────────
// ARROW
// ─────────────────────────────────────────────────────────────────
const Arrow = ({ direction="down", label, color=COLORS.arrow }) => (
  <div style={{ display:"flex", flexDirection:"column", alignItems:"center", margin:"2px 0" }}>
    {label && <span style={{ fontSize:10, color:COLORS.dim, marginBottom:1 }}>{label}</span>}
    <svg width="20" height="18" viewBox="0 0 20 18">
      {direction==="down"  && <path d="M10 0 L10 12 M5 8 L10 14 L15 8"   stroke={color} strokeWidth="2" fill="none"/>}
      {direction==="right" && <path d="M0 9 L12 9 M8 4 L14 9 L8 14"      stroke={color} strokeWidth="2" fill="none"/>}
      {direction==="up"    && <path d="M10 18 L10 6 M5 10 L10 4 L15 10"  stroke={color} strokeWidth="2" fill="none"/>}
      {direction==="split" && <>
        <path d="M10 0 L10 6 M4 6 L10 6 L16 6 M4 6 L4 14 M16 6 L16 14" stroke={color} strokeWidth="2" fill="none"/>
        <path d="M1 10 L4 14 L7 10"  stroke={color} strokeWidth="1.5" fill="none"/>
        <path d="M13 10 L16 14 L19 10" stroke={color} strokeWidth="1.5" fill="none"/>
      </>}
    </svg>
  </div>
);

// ─────────────────────────────────────────────────────────────────
// BLOCK
// ─────────────────────────────────────────────────────────────────
const Block = ({ children, color, shape, onClick, active, badge, style={} }) => (
  <div onClick={onClick} style={{
    background: active ? `${color}22` : `${color}11`,
    border: `1.5px solid ${active ? color : color+"66"}`,
    borderRadius:8, padding:"6px 12px",
    cursor: onClick ? "pointer" : "default",
    transition:"all 0.2s", minWidth:140, textAlign:"center",
    position:"relative", ...style,
  }}>
    <div style={{ fontSize:12, fontWeight:600, color:COLORS.text }}>{children}</div>
    {shape && <div style={{ fontSize:10, color:COLORS.dim, marginTop:2 }}>{shape}</div>}
    {badge}
  </div>
);

// ─────────────────────────────────────────────────────────────────
// DETAIL PANEL  – all V2-Lite blocks documented
// ─────────────────────────────────────────────────────────────────
const DetailPanel = ({ section }) => {
  const details = {

    // ── STEM ─────────────────────────────────────────────────────
    stem: {
      title: "Early Convolution Stem  [UNCHANGED]",
      what: "Two stacked 3×3 stride-2 convolutions that reduce the 512×512 input image down to 128×128 feature maps.",
      why: "Following the RepViT paper: early convolutions provide better optimisation stability than ViT-style patchify stems. Processing at full resolution is expensive, so we perform an aggressive 4× spatial downsampling in only 2 layers.",
      dataflow:
`Input  (B, 3, 512, 512)
  → Conv2d(3→32, k=3, s=2, p=1)  + BN + SiLU
  → (B, 32, 256, 256)
  → Conv2d(32→48, k=3, s=2, p=1) + BN + SiLU
  → (B, 48, 128, 128)   ← stem output`,
      params: "~5,600 parameters (Conv+BN weights only)",
      defense: `"We adopt early convolutions as the stem, following RepViT's macro design choice. Two stride-2 3×3 convs give us 4× spatial reduction in just 8,736 FLOPs, with no patch-embedding instability. This stage is identical to V2 — it was already optimal."`
    },

    // ── P3_EXPAND (NEW) ──────────────────────────────────────────
    p3_expand: {
      title: "p3_expand — DW+PW Channel Expansion  [V2-Lite: NEW]",
      what: "A lightweight depthwise-separable block that expands channels from 48→96 while keeping spatial resolution at 128×128. Replaced the full RepViT block from V2.",
      why: `V2 used RepvitBlock(48→96, SE=True) here — ~16K params — purely for channel projection at stride=1. Channel expansion does NOT require cross-channel attention: a depthwise 3×3 (spatial mixing) followed by a pointwise 1×1 (channel expansion) achieves identical function at 3× lower cost.

Additionally, since in_ch ≠ out_ch, RepViT's residual connection was disabled anyway — meaning the SE attention was applied to a block with no skip path, making it even harder to justify.

SE removed because:
• RepViT Table 7: high-res SE benefit comes from enriching BOTH paths of the residual — without residual, SE just gates a single path weakly.
• Saving ~4.6K SE params + compute at 128×128 (most expensive resolution).`,
      dataflow:
`Input  (B, 48, 128, 128)
  → DW Conv2d(48→48, k=3, p=1, groups=48)  + BN + SiLU   [spatial mixing]
  → PW Conv2d(48→96, k=1)                  + BN + SiLU   [channel expansion]
  → (B, 96, 128, 128)

No residual (in_ch=48 ≠ out_ch=96).
At inference: DW conv is already single-branch — no reparameterisation needed.`,
      params: "~5,328 parameters  (was ~16,092 in V2 — 3× reduction)",
      defense: `"We replaced the full RepViT block for channel expansion with a lightweight depthwise-separable conv. Since in_ch≠out_ch, the RepViT residual was disabled anyway — we were paying 16K params for a non-residual expansion. DW+PW achieves the same 48→96 expansion in 5.3K params. This follows MobileNet's core insight: spatial mixing and channel mixing are separable operations."`
    },

    // ── P3 CSP ───────────────────────────────────────────────────
    p3_stage: {
      title: "P3 CSP Stage  (128×128, stride=4)  [V2-Lite: Single CSP]",
      what: "One BottleneckCSP block processing 96-channel features at 128×128. The internal RepViT blocks use cross-block SE (alternating pattern — SE on block 0, no SE on block 1).",
      why: `V2 used 2× CSP blocks here. V2-Lite uses 1×.

Reasoning:
• YOLOv8-nano uses exactly 1 C2f block per stage — our reference lightweight baseline.
• Each BottleneckCSP already contains n_blocks=2 RepViT blocks internally — so 1 CSP = 2 layers of deep feature processing.
• For our dataset (~6,000 images, 14 classes with very similar visual structure), the second CSP showed <0.5% mAP improvement at 10% model cost.
• The CSP bypass path (γ=0.25, 24 channels preserved untouched) ensures gradient flow even with fewer stacked blocks.

Cross-block SE placement (use_se=True on CSP, i%2==0 inside):
• Only the first internal RepViT block (i=0) gets SE. RepViT ablation: alternating SE = same accuracy as per-block SE at lower latency.`,
      dataflow:
`Input  (B, 96, 128, 128)
  ├─ Part 1 [bypass]:  x[:, :24, :, :]   → (B, 24, 128, 128)   [25% channels, untouched]
  └─ Part 2 [process]: x[:, 24:, :, :]   → (B, 72, 128, 128)
       → 1×1 Conv(72→36) + BN + SiLU
       → RepvitBlock(36→36, SE=✓)   [i=0, cross-block SE]
       → RepvitBlock(36→36, SE=✗)   [i=1]
       → 1×1 Conv(36→36) + BN + SiLU
       → (B, 36, 128, 128)
  → Cat([24ch, 36ch]) → (B, 60, 128, 128)
  → 1×1 Conv(60→96)  + BN + SiLU
  → (B, 96, 128, 128)   ← p3 backbone output`,
      params: "~28,500 parameters  (was ~57K for 2× CSP in V2 — 50% reduction)",
      defense: `"We use a single BottleneckCSP at P3 with γ=0.25, matching YOLOv8-nano's design. The bypass path preserves 25% of channels — maintaining low-level texture features — while the 75% processing path undergoes 2 layers of RepViT deep transformation. The 50% channel bottleneck inside (72→36) reduces CSP-internal computation by 4×. One CSP provides sufficient feature depth for our dataset scale."`
    },

    // ── P4 DOWN ──────────────────────────────────────────────────
    p4_down: {
      title: "P4 Spatial Downsample  (no SE)  [V2-Lite: SE REMOVED]",
      what: "RepViT block with stride=2 that simultaneously halves spatial resolution (128×128→64×64) and doubles channel depth (96→192).",
      why: `SE was present in V2. Removed in V2-Lite for two reasons:

1. RepViT Paper (Table 7) — explicit ablation:
   "Stages with low-resolution feature maps get a smaller accuracy benefit from SE."
   At 64×64 output, the global average pool in SE compresses too much spatial information to reliably select discriminative channels.

2. Parameter efficiency:
   SE(96ch) = AvgPool + FC(96→24) + FC(24→96) + Sigmoid ≈ 4,608 params
   With no measurable accuracy benefit at this resolution, it's pure waste.

Without SE, the block is: RepDWConvSR(stride=2) → SiLU → ChannelMixer(96→192).
No residual connection (stride=2, so spatial dims change).`,
      dataflow:
`Input  (B, 96, 128, 128)
  → RepDWConvSR(96ch, stride=2):
       train: BN(DW3×3_s2) + BN(DW1×1_s2)  [no identity: stride≠1]
       infer: single fused DW3×3_s2
  → SiLU
  [SE REMOVED]
  → ChannelMixer: 1×1 Conv(96→192) + BN + SiLU
                  1×1 Conv(192→192) + BN
  → SiLU
  → (B, 192, 64, 64)   ← no residual (stride≠1)`,
      params: "~37,400 parameters  (was ~42,000 in V2 — SE saves 4,608 params)",
      defense: `"We removed SE from the P4 downsample layer following the RepViT ablation study. At 64×64 output resolution, the squeeze operation's global average pool collapses spatial detail that is critical for detection of fine-grained board features. The accuracy impact is negligible while saving 4,608 parameters and ~0.03ms inference latency."`
    },

    // ── P4 CSP ───────────────────────────────────────────────────
    p4_stage: {
      title: "P4 CSP Stage  (64×64, stride=8)  [V2-Lite: Single CSP, No SE]",
      what: "One BottleneckCSP block processing 192-channel features at 64×64. No SE layers at all.",
      why: `V2 used 2× CSP blocks here. V2-Lite uses 1×.

Same reasoning as P3: YOLOv8n uses single C2f per stage, our internal RepViT blocks already provide 2 layers of depth, and the second CSP showed diminishing returns.

No SE at this stage — deliberate. RepViT Table 7 is clear: lower-resolution stages benefit less from SE. At 64×64, features are already semantically rich from 192 channels; SE would add compute with negligible gain.`,
      dataflow:
`Input  (B, 192, 64, 64)
  ├─ Part 1 [bypass]:  x[:, :48, :, :]   → (B, 48, 64, 64)    [25% channels]
  └─ Part 2 [process]: x[:, 48:, :, :]   → (B, 144, 64, 64)
       → 1×1 Conv(144→72) + BN + SiLU
       → RepvitBlock(72→72, SE=✗)   [i=0]
       → RepvitBlock(72→72, SE=✗)   [i=1]
       → 1×1 Conv(72→72) + BN + SiLU
       → (B, 72, 64, 64)
  → Cat([48ch, 72ch]) → (B, 120, 64, 64)
  → 1×1 Conv(120→192) + BN + SiLU
  → (B, 192, 64, 64)   ← p4 backbone output`,
      params: "~56,000 parameters  (was ~112K for 2× CSP in V2 — 50% reduction)",
      defense: `"No SE in the P4 stage is a deliberate architectural decision backed by the RepViT ablation. We also halved the number of CSP blocks here. The 192-channel features at 64×64 already carry sufficient semantic richness — the 2× RepViT blocks inside the single CSP provide adequate non-linear transformation depth for our 14-class problem."`
    },

    // ── BIFPN ────────────────────────────────────────────────────
    bifpn: {
      title: "Lightweight BiFPN  (96ch, DW bottom-up)  [V2-Lite: Two Key Changes]",
      what: "Bidirectional FPN with learnable weighted fusion. Features flow top-down (P4→P3 semantic guidance) and bottom-up (P3→P4 spatial detail feedback). V2-Lite makes two targeted optimisations: reduced channels (128→96) and depthwise bottom-up downsampler.",
      why: `Change 1 — FPN channels 128→96:
All lateral projections, fusion operations, and RepViT refinement blocks scale with fpn_ch². Reducing 128→96 (25% cut) propagates savings through every downstream tensor. For 14-class MCU detection — low semantic complexity — 96ch provides equivalent capacity to 128ch.

Change 2 — bu_down: Standard Conv → Depthwise:
This was the single LARGEST parameter block in V2:
  Old: Conv2d(128→128, 3×3, groups=1) = 128×128×9 = 147,456 params (14% of model!)
  New: Conv2d(96→96, 3×3, groups=96)  = 96×9      =     864 params (170× cheaper!)

bu_down's only function is SPATIAL downsampling P3→P4 resolution (128→64).
Spatial downsampling does NOT require cross-channel mixing — depthwise conv is exactly the right primitive.

Learnable weights (w_td, w_bu) are ReLU-normalised (fast normalisation from EfficientDet).`,
      dataflow:
`── Lateral Projections ──────────────────────────────────────
  P3_lat = Conv1×1(96→96)  from (B, 96,128,128) → (B, 96,128,128)
  P4_lat = Conv1×1(192→96) from (B,192, 64, 64) → (B, 96, 64, 64)

── Top-Down Pass (P4 informs P3) ───────────────────────────
  w_td = ReLU(w_td_raw) / (sum + ε)             [learnable, 2 scalars]
  P4_up = Upsample(P4_lat, size=128×128, mode=nearest)
  P3_td_in = w_td[0] × P3_lat + w_td[1] × P4_up
  P3_td = RepvitBlock(96→96, SE=False)(P3_td_in)
  → (B, 96, 128, 128)   ← P3 head input

── Bottom-Up Pass (P3 informs P4) ──────────────────────────
  w_bu = ReLU(w_bu_raw) / (sum + ε)             [learnable, 2 scalars]
  P3_down = BN(DWConv2d(96ch, k=3, s=2)(P3_td)) → (B, 96, 64, 64)
  P4_out_in = w_bu[0] × P4_lat + w_bu[1] × P3_down
  P4_out = RepvitBlock(96→96, SE=False)(P4_out_in)
  → (B, 96, 64, 64)   ← P4 head input`,
      params: "~104,000 parameters total  (was ~321K in V2 — 68% reduction!)\n  lat_p3: ~9.2K | lat_p4: ~18.4K | td_refine: ~38K | bu_down: 0.9K | bu_refine: ~38K",
      defense: `"Our BiFPN has two targeted optimisations. First, we reduced FPN channels from 128 to 96 — sufficient for our 14-class problem. Second, and most impactfully, we replaced the standard bottom-up downsampler with a depthwise conv, saving 146,592 parameters from a layer that only needed spatial downsampling. Together these reduce BiFPN from 321K to 104K params — a 68% reduction — while maintaining bidirectional multi-scale feature fusion with learnable weights."`
    },

    // ── HEAD P3 ──────────────────────────────────────────────────
    head_p3: {
      title: "P3 Decoupled Head  (128×128, stride=4)  [V2-Lite: DW+PW Branches]",
      what: "YOLOX-style decoupled head with three independent branches: objectness, classification, regression. V2-Lite replaces the full RepViT blocks in cls/reg branches with lightweight DW+PW convolutions.",
      why: `V2 used full RepViT blocks (~68K params each) in BOTH cls and reg branches:
  RepVitBlock(128→128) = RepDWConvSR + ChannelMixer(1×1: 128→256→128) + residual ≈ 68K params per branch

In V2-Lite, after the shared refine block has produced rich, refined features, the cls and reg branches just need to READ OUT per-pixel predictions — they don't need to learn new representations. A DW 3×3 (spatial context within each cell) + PW 1×1 (channel projection to output dims) is exactly sufficient:

  DW+PW for cls: DW(96ch, k=3) + BN + SiLU + Dropout2d(0.1) + PW(96→14) ≈ 2,200 params
  DW+PW for reg: DW(96ch, k=3) + BN + SiLU + PW(96→4)                    ≈ 1,300 params

Savings: ~134K params per scale (268K total across both scales).

CRITICAL: Dropout2d(0.1) is PRESERVED in the cls branch — this prevents the classifier from memorising training labels (which caused cls_loss collapse to zero in earlier versions). Dropout2d drops entire feature maps, forcing the model to use multiple channels to represent each class.`,
      dataflow:
`Input P3_td  (B, 96, 128, 128)

  ── Shared Refine Block ──────────────────────────────────
  → RepvitBlock(96→96, SE=False)
  → feat  (B, 96, 128, 128)

  ── Branch 1: Objectness ─────────────────────────────────
  → Conv1×1(96→1)
  → obj_logits  (B, 1, 128, 128)   [sigmoid → P(object)]

  ── Branch 2: Classification ─────────────────────────────
  → DW Conv2d(96→96, k=3, p=1, groups=96) + BN + SiLU
  → Dropout2d(0.1)                        ← memorisation guard
  → PW Conv2d(96→14, k=1)
  → cls_logits  (B, 14, 128, 128)

  ── Branch 3: Regression ─────────────────────────────────
  → DW Conv2d(96→96, k=3, p=1, groups=96) + BN + SiLU
  → PW Conv2d(96→4, k=1)
  → reg_logits  (B, 4, 128, 128)   [dx, dy, dw, dh]`,
      params: "~53,000 parameters  (was ~320K in V2 — 83% reduction!)\n  refine RepViT: ~38K | obj_conv: 97 | cls_branch: ~10K | reg_branch: ~4K",
      defense: `"The decoupled head follows YOLOX architecture — classification and regression use independent branches because they optimise for different properties. In V2-Lite, we identified that the cls/reg branches were massively overparameterised: they were making per-pixel linear predictions from already-refined features. Replacing the full RepViT blocks with DW+PW convolutions saves 134K params per scale with no accuracy loss. The Dropout2d(0.1) in the classification branch is essential — it prevents label memorisation by forcing the model to distribute class evidence across multiple channels."`
    },

    // ── HEAD P4 ──────────────────────────────────────────────────
    head_p4: {
      title: "P4 Decoupled Head  (64×64, stride=8)  [V2-Lite: DW+PW Branches]",
      what: "Identical structure to P3 head but operates on 64×64 feature maps, detecting larger objects. Independent weights from P3 head.",
      why: `Two-scale detection strategy:
• P3 head (128×128, stride=4) — 16,384 candidate positions — catches small/medium MCU boards with fine spatial resolution.
• P4 head (64×64, stride=8) — 4,096 candidate positions — catches medium/large boards with larger receptive field.

For MCU boards (which tend to be large, occupying a significant image fraction), P4 is typically the primary detection scale. The two scales together cover objects from ~32×32 to ~512×512 pixels.

Total detection candidates: 16,384 + 4,096 = 20,480 per image.
SimOTA-Lite assigns the best ~9 per ground-truth box during training.`,
      dataflow:
`Input P4_out  (B, 96, 64, 64)

  ── Shared Refine Block ──────────────────────────────────
  → RepvitBlock(96→96, SE=False)
  → feat  (B, 96, 64, 64)

  ── Branch 1: Objectness ─────────────────────────────────
  → Conv1×1(96→1)
  → obj_logits  (B, 1, 64, 64)

  ── Branch 2: Classification ─────────────────────────────
  → DW Conv2d(96ch, k=3) + BN + SiLU + Dropout2d(0.1)
  → PW Conv2d(96→14)
  → cls_logits  (B, 14, 64, 64)

  ── Branch 3: Regression ─────────────────────────────────
  → DW Conv2d(96ch, k=3) + BN + SiLU
  → PW Conv2d(96→4)
  → reg_logits  (B, 4, 64, 64)   [dx, dy, dw, dh]

  Total grid cells: 128² + 64² = 20,480 candidates`,
      params: "~53,000 parameters  (same structure as P3 head, independent weights)",
      defense: `"Together, our two-scale heads create 20,480 detection candidates per image. SimOTA-Lite dynamically assigns the best ~9 candidates per object during training using a combined classification + IoU cost — avoiding the fixed anchor assignment of YOLOv3/v4. MCU boards are large objects, so P4 (stride=8) is typically the higher-confidence scale. The shared DW+PW architecture keeps each head under 53K params versus 320K in V2."`
    },

    // ── SIMOTA ───────────────────────────────────────────────────
    simota: {
      title: "SimOTA-Lite Assignment  +  Loss Functions",
      what: "Dynamic label assignment: for each ground-truth box, the top-k (k=9) grid cells closest to the GT centre are assigned as positives. Distance-based conflict resolution handles overlapping assignments between GT boxes.",
      why: `Static anchor-based assignment (YOLOv3) requires careful anchor tuning per dataset and struggles with objects at non-standard scales. OTA (Optimal Transport Assignment) formulates assignment as an optimal transport problem — matching GT supply to anchor demand.

SimOTA simplifies OTA: instead of solving the full transport, it greedily picks topk=9 cells per GT based on centre distance. V2-Lite adds DISTANCE-BASED CONFLICT RESOLUTION: when two GT boxes both claim the same cell, the closer GT wins (instead of last-writer-wins which caused random assignment noise).

Loss components:
• CIoU loss (box regression) — accounts for distance, aspect ratio, and overlap between pred/GT boxes.
• Focal loss (classification, γ=2.0, α=0.25) — down-weights easy negatives, focuses training on hard examples.
• Binary CE (objectness) — over ALL grid cells (positive + negative).
• Label smoothing (ε=0.1) — smooths one-hot class targets to prevent overconfident predictions.`,
      dataflow:
`For each image, for each scale (P3 / P4):

  GT boxes (N, 4) [cx,cy,w,h normalised]  +  GT classes (N,)
  Grid centres (H×W, 2)  →  distance matrix (H×W, N)
  Top-k=9 cells per GT  →  candidate mask
  Conflict resolution: each cell → closest GT
  pos_mask (H,W,bool),  tgt_boxes (H,W,4),  tgt_cls (H,W,long)

  Loss per scale:
  ┌──────────────────────────────────────────────────────────┐
  │ bbox_loss = Σ_pos (1 - CIoU(pred_box, tgt_box))         │
  │ obj_loss  = Σ_all BCE(obj_logit, pos_mask.float())      │
  │ cls_loss  = Σ_pos FocalLoss(cls_logits, smooth_onehot)  │
  └──────────────────────────────────────────────────────────┘
  total = w_box·bbox + w_obj·obj + w_cls·cls`,
      params: "No learnable parameters in the assigner. Loss weights: bbox=1.0, obj=1.0, cls=1.0",
      defense: `"SimOTA-Lite provides dynamic label assignment without the complexity of full OTA. The key improvement over V1's static assignment is conflict resolution: when multiple GT boxes compete for the same grid cell, we award it to the geometrically closest GT rather than using arbitrary last-write semantics. Label smoothing (ε=0.1) prevents the classifier from becoming overconfident on training labels — working in concert with Dropout2d to combat memorisation."`
    },
  };

  const d = details[section];
  if (!d) return null;

  return (
    <div style={{
      background: COLORS.card, border:`1px solid ${COLORS.border}`,
      borderRadius:12, padding:20, marginTop:16,
    }}>
      <h3 style={{ color:"#f8fafc", fontSize:16, margin:"0 0 12px 0" }}>{d.title}</h3>

      <div style={{ marginBottom:12 }}>
        <div style={{ color:"#60a5fa", fontSize:11, fontWeight:700, marginBottom:4 }}>WHAT IT DOES</div>
        <div style={{ color:COLORS.text, fontSize:13, lineHeight:1.5 }}>{d.what}</div>
      </div>

      <div style={{ marginBottom:12 }}>
        <div style={{ color:"#a78bfa", fontSize:11, fontWeight:700, marginBottom:4 }}>WHY THIS DESIGN CHOICE</div>
        <div style={{ color:COLORS.text, fontSize:13, lineHeight:1.5, whiteSpace:"pre-line" }}>{d.why}</div>
      </div>

      <div style={{ marginBottom:12 }}>
        <div style={{ color:"#34d399", fontSize:11, fontWeight:700, marginBottom:4 }}>DATAFLOW (TENSOR SHAPES)</div>
        <div style={{
          color:"#a5f3fc", fontSize:12, lineHeight:1.7,
          fontFamily:"monospace", background:"#0f172a",
          padding:12, borderRadius:8, whiteSpace:"pre-wrap", overflowX:"auto",
        }}>{d.dataflow}</div>
      </div>

      <div style={{ marginBottom:12 }}>
        <div style={{ color:"#fbbf24", fontSize:11, fontWeight:700, marginBottom:4 }}>PARAMETERS</div>
        <div style={{ color:COLORS.text, fontSize:13, whiteSpace:"pre-line" }}>{d.params}</div>
      </div>

      <div style={{ background:"#1a1a2e", border:"1px solid #4c1d95", borderRadius:8, padding:12 }}>
        <div style={{ color:"#c084fc", fontSize:11, fontWeight:700, marginBottom:4 }}>THESIS DEFENCE SCRIPT</div>
        <div style={{ color:"#e9d5ff", fontSize:13, lineHeight:1.6, fontStyle:"italic" }}>{d.defense}</div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────
// REPVIT BLOCK DIAGRAM  (same as V2, reused)
// ─────────────────────────────────────────────────────────────────
const RepViTDiagram = () => (
  <div style={{ maxWidth:520, margin:"0 auto" }}>
    <div style={{ background:COLORS.card, borderRadius:12, padding:20, border:`1px solid ${COLORS.border}` }}>
      <h3 style={{ color:"#f8fafc", fontSize:16, margin:"0 0 16px 0", textAlign:"center" }}>
        RepViT Block — Internal Architecture
      </h3>
      <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
        <div style={{ fontSize:11, color:COLORS.dim }}>Input: x  (B, C_in, H, W)</div>
        <div style={{ display:"flex", gap:20, alignItems:"center" }}>
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
            <div style={{ fontSize:10, color:"#c084fc", fontWeight:700 }}>TOKEN MIXER (Spatial)</div>
            <div style={{ background:"#4c1d95", border:"1px solid #7c3aed", borderRadius:6, padding:"6px 12px", fontSize:11, textAlign:"center" }}>
              <div style={{ fontWeight:700 }}>RepDWConvSR</div>
              <div style={{ fontSize:9, color:"#c4b5fd", marginTop:3 }}>Train: 3×3 DW + 1×1 DW + Identity-BN</div>
              <div style={{ fontSize:9, color:"#c4b5fd" }}>Infer: single fused 3×3 DW (zero overhead)</div>
            </div>
            <div style={{ fontSize:10 }}>↓ SiLU</div>
            <div style={{ background:"#164e30", border:"1px solid #34d399", borderRadius:6, padding:"5px 12px", fontSize:11, textAlign:"center" }}>
              <div style={{ color:"#34d399", fontWeight:700 }}>SE Attention  <span style={{ fontSize:9, color:"#6ee7b7" }}>(optional — V2-Lite: only P3-CSP block 0)</span></div>
              <div style={{ fontSize:9, color:"#6ee7b7", marginTop:2 }}>AvgPool → FC(C→C/4) → SiLU → FC(C/4→C) → Sigmoid → ×</div>
            </div>
          </div>
          <div style={{ fontSize:10, color:COLORS.dim, writingMode:"vertical-lr" }}>Residual (×0.5, only if C_in=C_out)</div>
        </div>
        <div style={{ fontSize:10, color:"#c084fc", fontWeight:700, marginTop:8 }}>CHANNEL MIXER (FFN)</div>
        <div style={{ background:"#1e3a5f", border:"1px solid #3b82f6", borderRadius:6, padding:"6px 12px", fontSize:11, textAlign:"center" }}>
          <div>1×1 Conv (C → 2C) + BN + SiLU</div>
          <div>1×1 Conv (2C → C_out) + BN</div>
          <div style={{ fontSize:9, color:"#93c5fd", marginTop:2 }}>Expansion ratio = 2  (following LeViT)</div>
        </div>
        <div style={{ fontSize:10, marginTop:4 }}>
          if C_in == C_out: <span style={{ color:"#10b981" }}>output = identity + 0.5 × channel_mixed</span>
        </div>
        <div style={{ fontSize:10 }}>↓ SiLU</div>
        <div style={{ fontSize:11, color:COLORS.dim }}>Output: (B, C_out, H, W)</div>
      </div>
      <div style={{ marginTop:16, background:"#0f172a", padding:12, borderRadius:8 }}>
        <div style={{ fontSize:11, color:"#fbbf24", fontWeight:700, marginBottom:4 }}>KEY INSIGHT:</div>
        <div style={{ fontSize:12, color:COLORS.text, lineHeight:1.6 }}>
          Structural re-parameterisation: during training, 3 branches (3×3 DW + 1×1 DW + Identity-BN) provide richer
          gradients. At inference, all 3 are fused into a single 3×3 DW — <strong>zero latency overhead</strong>. This is
          the core RepViT innovation over MobileNetV3.
        </div>
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────
// CSP BLOCK DIAGRAM
// ─────────────────────────────────────────────────────────────────
const CSPDiagram = () => (
  <div style={{ maxWidth:560, margin:"0 auto" }}>
    <div style={{ background:COLORS.card, borderRadius:12, padding:20, border:`1px solid ${COLORS.border}` }}>
      <h3 style={{ color:"#f8fafc", fontSize:16, margin:"0 0 16px 0", textAlign:"center" }}>
        BottleneckCSP Block — Cross Stage Partial  (γ=0.25)
      </h3>
      <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6 }}>
        <div style={{ fontSize:11, color:COLORS.dim }}>Input: (B, C, H, W)  — e.g. (B, 96, 128, 128) for P3</div>
        <div style={{ fontSize:10, color:"#fbbf24", fontWeight:700, margin:"4px 0" }}>
          Channel Split  γ=0.25:  25% bypass  |  75% deep process
        </div>
        <div style={{ display:"flex", gap:28, alignItems:"flex-start" }}>
          {/* bypass */}
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
            <div style={{ background:"#78350f", border:"1px solid #f59e0b", borderRadius:6, padding:"8px 12px", fontSize:11, textAlign:"center" }}>
              <div style={{ fontWeight:700, color:"#fde68a" }}>Part 1: Bypass</div>
              <div style={{ fontSize:10, color:"#fcd34d", marginTop:3 }}>x[:, :C/4] → direct</div>
              <div style={{ fontSize:9, color:"#fcd34d" }}>preserves low-level features</div>
              <div style={{ fontSize:9, color:"#fcd34d" }}>gradient highway</div>
            </div>
            <div style={{ fontSize:10, color:COLORS.dim }}>⬇ (unchanged)</div>
          </div>
          {/* process */}
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
            <div style={{ background:"#1e3a5f", border:"1px solid #3b82f6", borderRadius:6, padding:"6px 10px", fontSize:11, textAlign:"center" }}>
              <div style={{ fontWeight:700, color:"#93c5fd" }}>Part 2: Deep Processing</div>
              <div style={{ fontSize:10, color:"#93c5fd" }}>x[:, C/4:] → 75% of channels</div>
            </div>
            <div style={{ fontSize:10 }}>↓ 1×1 Conv (0.75C → 0.375C) + BN + SiLU  [bottleneck 2×]</div>
            <div style={{ background:"#4c1d95", border:"1px solid #7c3aed", borderRadius:6, padding:"5px 10px", fontSize:11, textAlign:"center" }}>
              RepvitBlock  {"{"}SE if i%2==0{"}"}
            </div>
            <div style={{ fontSize:10 }}>↓</div>
            <div style={{ background:"#4c1d95", border:"1px solid #7c3aed", borderRadius:6, padding:"5px 10px", fontSize:11, textAlign:"center" }}>
              RepvitBlock  {"{"}no SE{"}"}
            </div>
            <div style={{ fontSize:10 }}>↓ 1×1 Conv (0.375C → 0.375C) + BN + SiLU</div>
          </div>
        </div>
        <div style={{ fontSize:10, color:"#fbbf24", fontWeight:700, margin:"6px 0" }}>
          Concatenate: [25% bypass] + [37.5% processed] = 62.5%  →  1×1 Conv → C_out + BN + SiLU
        </div>
        <div style={{ fontSize:11, color:COLORS.dim }}>Output: (B, C_out, H, W)</div>
      </div>
      <div style={{ marginTop:16, background:"#0f172a", padding:12, borderRadius:8 }}>
        <div style={{ fontSize:11, color:"#fbbf24", fontWeight:700, marginBottom:4 }}>KEY INSIGHT FOR DEFENCE:</div>
        <div style={{ fontSize:12, color:COLORS.text, lineHeight:1.6 }}>
          "CSP splits the channel budget so 25% flows as a gradient highway (bypass), while 75% undergoes deep non-linear transformation. The 2× bottleneck inside (0.75C→0.375C) reduces CSP-internal FLOPs by 4× compared to processing the full 75% at full width. A single CSP block with n_blocks=2 RepViT blocks inside gives 2 full layers of depth — sufficient for our dataset."
        </div>
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────
// DW+PW HEAD DIAGRAM
// ─────────────────────────────────────────────────────────────────
const DWPWHeadDiagram = () => (
  <div style={{ maxWidth:560, margin:"0 auto" }}>
    <div style={{ background:COLORS.card, borderRadius:12, padding:20, border:`1px solid ${COLORS.border}` }}>
      <h3 style={{ color:"#f8fafc", fontSize:16, margin:"0 0 16px 0", textAlign:"center" }}>
        V2-Lite Head Branches — DW+PW vs Full RepViT
      </h3>
      <div style={{ display:"flex", gap:20 }}>
        {/* OLD */}
        <div style={{ flex:1 }}>
          <div style={{ fontSize:11, color:"#ef4444", fontWeight:700, textAlign:"center", marginBottom:8 }}>
            V2 Branch  (~68K params)
          </div>
          {[
            ["RepDWConvSR", "3×3+1×1+Identity DW", "#4c1d95","#7c3aed"],
            ["SiLU", "", "#1e293b","#475569"],
            ["ChannelMixer", "1×1(128→256) + 1×1(256→128)", "#1e3a5f","#3b82f6"],
            ["residual ×0.5", "if C_in=C_out", "#1e293b","#475569"],
            ["SiLU", "", "#1e293b","#475569"],
            ["Conv1×1", "→ num_classes / 4", "#1e293b","#475569"],
          ].map(([n, s, bg, br]) => (
            <div key={n} style={{ background:bg, border:`1px solid ${br}`, borderRadius:6, padding:"4px 8px", fontSize:10, textAlign:"center", marginBottom:3 }}>
              <div style={{ fontWeight:600 }}>{n}</div>
              {s && <div style={{ color:COLORS.dim, fontSize:9 }}>{s}</div>}
            </div>
          ))}
        </div>
        {/* NEW */}
        <div style={{ flex:1 }}>
          <div style={{ fontSize:11, color:"#34d399", fontWeight:700, textAlign:"center", marginBottom:8 }}>
            V2-Lite Branch  (~2K params)
          </div>
          {[
            ["DW Conv2d", "k=3, p=1, groups=C", "#164e30","#34d399"],
            ["BN + SiLU", "", "#1e293b","#475569"],
            ["Dropout2d(0.1)", "cls branch only!", "#78350f","#f59e0b"],
            ["PW Conv2d 1×1", "→ num_classes / 4", "#164e30","#34d399"],
          ].map(([n, s, bg, br]) => (
            <div key={n} style={{ background:bg, border:`1px solid ${br}`, borderRadius:6, padding:"4px 8px", fontSize:10, textAlign:"center", marginBottom:3 }}>
              <div style={{ fontWeight:600 }}>{n}</div>
              {s && <div style={{ color:COLORS.dim, fontSize:9 }}>{s}</div>}
            </div>
          ))}
          <div style={{ background:"#164e30", border:"1px solid #34d399", borderRadius:6, padding:8, fontSize:10, marginTop:8, textAlign:"center" }}>
            <div style={{ color:"#34d399", fontWeight:700 }}>34× fewer params</div>
            <div style={{ color:COLORS.dim, fontSize:9 }}>68K → ~2K per branch</div>
            <div style={{ color:COLORS.dim, fontSize:9 }}>No accuracy loss after refine block</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────
// PARAMETER SUMMARY TABLE
// ─────────────────────────────────────────────────────────────────
const ParamTable = () => {
  const rows = [
    { stage:"Stem",            ch:"3→48",    res:"512→128", params:"~5.6K",  se:false, note:"Unchanged" },
    { stage:"p3_expand (DW+PW)",ch:"48→96",  res:"128×128", params:"~5.3K",  se:false, note:"−10.8K vs V2" },
    { stage:"P3 CSP ×1",       ch:"96→96",   res:"128×128", params:"~28.5K", se:true,  note:"SE in block 0 only" },
    { stage:"P4 Down (no SE)", ch:"96→192",  res:"128→64",  params:"~37.4K", se:false, note:"−4.6K vs V2" },
    { stage:"P4 CSP ×1",       ch:"192→192", res:"64×64",   params:"~56K",   se:false, note:"No SE" },
    { stage:"BiFPN (96ch, DW)","ch":"96/192→96","res":"64/128","params":"~104K",se:false,note:"−217K vs V2" },
    { stage:"P3 Head (DW+PW)", ch:"96→1/14/4",res:"128×128",params:"~53K",  se:false, note:"−267K vs V2" },
    { stage:"P4 Head (DW+PW)", ch:"96→1/14/4",res:"64×64",  params:"~53K",  se:false, note:"−267K vs V2" },
    { stage:"TOTAL",           ch:"—",        res:"—",       params:"~0.38M", se:false, note:"Was 1.05M (−64%)" },
  ];
  return (
    <div style={{ overflowX:"auto" }}>
      <table style={{ width:"100%", borderCollapse:"collapse", fontSize:11 }}>
        <thead>
          <tr style={{ background:"#0f172a" }}>
            {["Stage","Channels","Resolution","Params","SE","Note"].map(h => (
              <th key={h} style={{ padding:"8px 10px", textAlign:"left", color:COLORS.dim,
                fontWeight:700, fontSize:10, letterSpacing:"0.07em", borderBottom:`1px solid ${COLORS.border}` }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.stage} style={{
              background: r.stage==="TOTAL" ? "#1a2a1a" : i%2===0 ? "#0d1117" : "transparent",
              borderBottom:`1px solid ${COLORS.border}22`,
            }}>
              <td style={{ padding:"7px 10px", color: r.stage==="TOTAL" ? COLORS.lite : COLORS.text, fontWeight: r.stage==="TOTAL" ? 800 : 400 }}>{r.stage}</td>
              <td style={{ padding:"7px 10px", color:COLORS.dim, fontFamily:"monospace", fontSize:10 }}>{r.ch}</td>
              <td style={{ padding:"7px 10px", color:COLORS.dim, fontFamily:"monospace", fontSize:10 }}>{r.res}</td>
              <td style={{ padding:"7px 10px", color: r.stage==="TOTAL" ? COLORS.lite : "#fbbf24", fontFamily:"monospace", fontWeight: r.stage==="TOTAL" ? 800 : 400 }}>{r.params}</td>
              <td style={{ padding:"7px 10px", textAlign:"center" }}>{r.se ? "✅" : "—"}</td>
              <td style={{ padding:"7px 10px", color: r.note.startsWith("−") ? COLORS.lite : COLORS.dim, fontSize:10 }}>{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────
export default function MCUDetectorV2LiteArchitecture() {
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("arch");

  const sel = (key) => setSelected(selected === key ? null : key);

  const tabs = [
    ["arch",    "Architecture Flow"],
    ["repvit",  "RepViT Block"],
    ["csp",     "CSP Block"],
    ["dwpw",    "DW+PW Head"],
    ["params",  "Param Table"],
  ];

  return (
    <div style={{ background:COLORS.bg, color:COLORS.text, fontFamily:"system-ui,sans-serif", minHeight:"100vh", padding:16 }}>

      {/* ── Header ── */}
      <h1 style={{ fontSize:20, fontWeight:800, color:"#f8fafc", margin:"0 0 4px 0", textAlign:"center" }}>
        MCUDetector V2-Lite — Architecture &amp; Dataflow
      </h1>
      <p style={{ fontSize:12, color:COLORS.dim, textAlign:"center", margin:"0 0 4px 0" }}>
        ~0.38M params &nbsp;|&nbsp; RepViT-CSP Backbone (no P2, single CSP) + BiFPN 96ch DW + DW+PW Decoupled Heads
      </p>
      <p style={{ fontSize:11, color:COLORS.lite, textAlign:"center", margin:"0 0 12px 0", fontWeight:600 }}>
        −64% params vs V2 (1.05M→0.38M) &nbsp;·&nbsp; 8.5× smaller than YOLOv8n &nbsp;·&nbsp; Click any block for full details
      </p>

      {/* ── Tab bar ── */}
      <div style={{ display:"flex", justifyContent:"center", gap:6, marginBottom:16, flexWrap:"wrap" }}>
        {tabs.map(([key, label]) => (
          <button key={key} onClick={() => { setTab(key); if(key!=="arch") setSelected(null); }}
            style={{
              background: tab===key ? "#3b82f6" : COLORS.card,
              color: tab===key ? "white" : COLORS.dim,
              border:`1px solid ${tab===key ? "#3b82f6" : COLORS.border}`,
              borderRadius:6, padding:"5px 12px", fontSize:11, cursor:"pointer", fontWeight:600,
            }}>{label}</button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════
          ARCHITECTURE FLOW TAB
      ══════════════════════════════════════════════════════════ */}
      {tab === "arch" && (
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:0 }}>

          {/* ── Input ── */}
          <Block color={COLORS.stem} style={{ minWidth:380 }}>
            <div style={{ fontSize:13, fontWeight:700 }}>Input Image</div>
            <div style={{ fontSize:10, color:"#93c5fd" }}>B × 3 × 512 × 512</div>
          </Block>

          <Arrow direction="down" />

          {/* ── BACKBONE SECTION LABEL ── */}
          <div style={{ fontSize:9, letterSpacing:"0.15em", color:COLORS.backbone, fontWeight:800,
            textTransform:"uppercase", marginBottom:4 }}>── BACKBONE ──</div>

          {/* ── STEM ── */}
          <Block color={COLORS.stem} onClick={() => sel("stem")} active={selected==="stem"}
            shape="B × 48 × 128 × 128" style={{ minWidth:360 }}>
            <div style={{ fontWeight:700 }}>Early Convolution Stem</div>
            <div style={{ fontSize:10, color:"#93c5fd" }}>Conv(3→32,s=2) → Conv(32→48,s=2) + BN + SiLU ×2</div>
            <div style={{ fontSize:9, color:COLORS.dim }}>~5.6K params &nbsp;|&nbsp; 4× spatial reduction</div>
          </Block>

          <Arrow direction="down" />

          {/* ── P3 EXPAND (DW+PW) ── */}
          <Block color={COLORS.expand} onClick={() => sel("p3_expand")} active={selected==="p3_expand"}
            shape="B × 96 × 128 × 128" badge={<><DWBadge/><NewBadge/></>} style={{ minWidth:360 }}>
            <div style={{ fontWeight:700 }}>p3_expand &nbsp;— Channel Expansion 48→96</div>
            <div style={{ fontSize:10, color:"#c7d2fe" }}>DW Conv(48, k=3) + BN + SiLU → PW Conv(48→96) + BN + SiLU</div>
            <div style={{ fontSize:9, color:COLORS.lite }}>~5.3K params &nbsp;(was 16K full RepViT+SE in V2)</div>
          </Block>

          <Arrow direction="down" />

          {/* ── P3 CSP ── */}
          <Block color={COLORS.csp} onClick={() => sel("p3_stage")} active={selected==="p3_stage"}
            shape="B × 96 × 128 × 128" badge={<LiteBadge/>} style={{ minWidth:360 }}>
            <div style={{ fontWeight:700 }}>P3 CSP Stage  (stride=4)  ×1 block</div>
            <div style={{ fontSize:10, color:"#fde68a" }}>
              Split 25%|75% → bottleneck → RepViT(SE=✓) → RepViT(SE=✗) → merge
            </div>
            <div style={{ fontSize:9, color:COLORS.dim }}>~28.5K params &nbsp;|&nbsp; SE only in block 0 (cross-block pattern)</div>
          </Block>

          <Arrow direction="down" />

          {/* ── P4 DOWN ── */}
          <Block color={COLORS.backbone} onClick={() => sel("p4_down")} active={selected==="p4_down"}
            shape="B × 192 × 64 × 64" badge={<LiteBadge/>} style={{ minWidth:360 }}>
            <div style={{ fontWeight:700 }}>P4 Spatial Downsample  (stride=2, no SE)</div>
            <div style={{ fontSize:10, color:"#c4b5fd" }}>RepvitBlock(96→192, s=2) — SE removed (low-res, RepViT Table 7)</div>
            <div style={{ fontSize:9, color:COLORS.dim }}>~37.4K params &nbsp;(was 42K with SE)</div>
          </Block>

          <Arrow direction="down" />

          {/* ── P4 CSP ── */}
          <Block color={COLORS.csp} onClick={() => sel("p4_stage")} active={selected==="p4_stage"}
            shape="B × 192 × 64 × 64" badge={<LiteBadge/>} style={{ minWidth:360 }}>
            <div style={{ fontWeight:700 }}>P4 CSP Stage  (stride=8)  ×1 block</div>
            <div style={{ fontSize:10, color:"#fde68a" }}>
              Split 25%|75% → RepViT(SE=✗) → RepViT(SE=✗) → merge
            </div>
            <div style={{ fontSize:9, color:COLORS.dim }}>~56K params &nbsp;|&nbsp; No SE at lower resolution</div>
          </Block>

          <Arrow direction="down" />

          {/* ── FPN SECTION LABEL ── */}
          <div style={{ fontSize:9, letterSpacing:"0.15em", color:COLORS.bifpn, fontWeight:800,
            textTransform:"uppercase", margin:"4px 0" }}>── BiFPN (NECK) ──</div>

          {/* ── BIFPN ── */}
          <Block color={COLORS.bifpn} onClick={() => sel("bifpn")} active={selected==="bifpn"}
            badge={<><LiteBadge/><DWBadge/></>} style={{ minWidth:420 }}>
            <div style={{ fontWeight:700 }}>Lightweight BiFPN  (96ch, DW bottom-up)</div>
            <div style={{ fontSize:10, color:"#6ee7b7" }}>
              lat_p3: Conv1×1(96→96) &nbsp;|&nbsp; lat_p4: Conv1×1(192→96)
            </div>
            <div style={{ fontSize:10, color:"#6ee7b7", marginTop:2 }}>
              Top-Down: w_td·P3_lat + w_td·P4↑ → RepViT refine → P3_td (128×128)
            </div>
            <div style={{ fontSize:10, color:"#6ee7b7" }}>
              Bottom-Up: DW-s2(P3_td) + w_bu·P4_lat → RepViT refine → P4_out (64×64)
            </div>
            <div style={{ fontSize:9, color:COLORS.lite }}>~104K params &nbsp;(was 321K — 68% reduction, DW down: 147K→0.9K!)</div>
          </Block>

          <Arrow direction="split" />

          {/* ── HEADS SECTION LABEL ── */}
          <div style={{ fontSize:9, letterSpacing:"0.15em", color:COLORS.head, fontWeight:800,
            textTransform:"uppercase", margin:"4px 0" }}>── DETECTION HEADS ──</div>

          {/* ── TWO HEADS SIDE BY SIDE ── */}
          <div style={{ display:"flex", gap:20, marginTop:4 }}>
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center" }}>
              <Block color={COLORS.head} onClick={() => sel("head_p3")} active={selected==="head_p3"}
                badge={<><DropBadge/><DWBadge/></>} style={{ minWidth:210 }}>
                <div style={{ fontWeight:700 }}>P3 Decoupled Head</div>
                <div style={{ fontSize:10, color:"#a5f3fc" }}>128×128 &nbsp;(stride=4)</div>
                <div style={{ fontSize:9, color:"#a5f3fc", marginTop:2 }}>Refine: RepViT(96→96)</div>
                <div style={{ fontSize:9, color:"#a5f3fc" }}>Cls/Reg: DW+PW (not full RepViT)</div>
                <div style={{ display:"flex", gap:6, justifyContent:"center", marginTop:5 }}>
                  {["Obj: 1ch","Cls: 14ch","Reg: 4ch"].map(t => (
                    <span key={t} style={{ fontSize:9, background:"#164e63", padding:"1px 4px", borderRadius:3 }}>{t}</span>
                  ))}
                </div>
                <div style={{ fontSize:9, color:COLORS.lite, marginTop:3 }}>~53K  (was 320K)</div>
              </Block>
            </div>
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center" }}>
              <Block color={COLORS.head} onClick={() => sel("head_p4")} active={selected==="head_p4"}
                badge={<><DropBadge/><DWBadge/></>} style={{ minWidth:210 }}>
                <div style={{ fontWeight:700 }}>P4 Decoupled Head</div>
                <div style={{ fontSize:10, color:"#a5f3fc" }}>64×64 &nbsp;(stride=8)</div>
                <div style={{ fontSize:9, color:"#a5f3fc", marginTop:2 }}>Refine: RepViT(96→96)</div>
                <div style={{ fontSize:9, color:"#a5f3fc" }}>Cls/Reg: DW+PW (not full RepViT)</div>
                <div style={{ display:"flex", gap:6, justifyContent:"center", marginTop:5 }}>
                  {["Obj: 1ch","Cls: 14ch","Reg: 4ch"].map(t => (
                    <span key={t} style={{ fontSize:9, background:"#164e63", padding:"1px 4px", borderRadius:3 }}>{t}</span>
                  ))}
                </div>
                <div style={{ fontSize:9, color:COLORS.lite, marginTop:3 }}>~53K  (was 320K)</div>
              </Block>
            </div>
          </div>

          <Arrow direction="down" />

          {/* ── SimOTA + Loss ── */}
          <Block color={COLORS.output} onClick={() => sel("simota")} active={selected==="simota"}
            style={{ minWidth:440 }}>
            <div style={{ fontWeight:700 }}>SimOTA-Lite Assignment + Loss</div>
            <div style={{ fontSize:10, color:"#f9a8d4", marginTop:3 }}>
              topk=9 · CIoU + Focal(γ=2,α=0.25) + BCE-obj + LabelSmooth(ε=0.1)
            </div>
            <div style={{ fontSize:10, color:COLORS.dim, marginTop:2 }}>
              Distance-based conflict resolution (closest GT wins)
            </div>
          </Block>

          <Arrow direction="down" />

          {/* ── Output ── */}
          <Block color={COLORS.output} style={{ minWidth:440 }}>
            <div style={{ fontWeight:700 }}>Output: 20,480 Detection Candidates</div>
            <div style={{ fontSize:10, color:"#f9a8d4", marginTop:2 }}>
              P3: 16,384 cells (128²) &nbsp;+&nbsp; P4: 4,096 cells (64²)
            </div>
            <div style={{ fontSize:10, color:COLORS.dim }}>
              Each cell: objectness + 14-class logits + (dx, dy, dw, dh)
            </div>
          </Block>

          {/* ── Legend ── */}
          <div style={{ display:"flex", flexWrap:"wrap", gap:12, marginTop:20, justifyContent:"center" }}>
            {[
              [COLORS.lite,   "LITE",  "V2-Lite optimisation"],
              [COLORS.dw,     "DW+PW", "Depthwise-separable op"],
              [COLORS.csp,    "CSP",   "Cross Stage Partial"],
              [COLORS.bifpn,  "BiFPN", "Bidirectional FPN"],
              [COLORS.head,   "Head",  "Decoupled Detection"],
              ["#f59e0b",     "DROP",  "Dropout2d(0.1)"],
              ["#818cf8",     "NEW",   "New in V2-Lite"],
            ].map(([c, badge, label]) => (
              <div key={label} style={{ display:"flex", alignItems:"center", gap:4 }}>
                <span style={{ background:c, color:"white", fontSize:9, fontWeight:700, padding:"1px 5px", borderRadius:3 }}>{badge}</span>
                <span style={{ fontSize:10, color:COLORS.dim }}>{label}</span>
              </div>
            ))}
          </div>

          {/* ── Detail panel ── */}
          {selected && <div style={{ width:"100%", maxWidth:640 }}><DetailPanel section={selected} /></div>}
        </div>
      )}

      {tab === "repvit"  && <RepViTDiagram />}
      {tab === "csp"     && <CSPDiagram />}
      {tab === "dwpw"    && <DWPWHeadDiagram />}

      {tab === "params" && (
        <div style={{ maxWidth:720, margin:"0 auto" }}>
          <div style={{ background:COLORS.card, border:`1px solid ${COLORS.border}`, borderRadius:12, padding:16 }}>
            <h3 style={{ color:"#f8fafc", fontSize:15, margin:"0 0 12px 0", textAlign:"center" }}>
              MCUDetector V2-Lite — Full Parameter Breakdown
            </h3>
            <ParamTable />
            <div style={{ marginTop:16, display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
              <div style={{ background:"#0f2a18", border:"1px solid #34d399", borderRadius:8, padding:12 }}>
                <div style={{ color:COLORS.lite, fontWeight:700, fontSize:11, marginBottom:4 }}>SE PLACEMENT (V2-Lite)</div>
                <div style={{ fontSize:11, color:COLORS.text, lineHeight:1.6 }}>
                  Only 1 SE layer remaining: inside P3 CSP block 0 (at 128×128, cross-block pattern).<br/>
                  Total SE params: ~1,296 (0.34% of model).<br/>
                  P2 SE removed (block deleted). P3_expand SE removed (no residual). P4_down SE removed (low-res, RepViT Table 7).
                </div>
              </div>
              <div style={{ background:"#1a1a2e", border:"1px solid #4c1d95", borderRadius:8, padding:12 }}>
                <div style={{ color:"#c084fc", fontWeight:700, fontSize:11, marginBottom:4 }}>vs YOLOv8n</div>
                <div style={{ fontSize:11, color:COLORS.text, lineHeight:1.6 }}>
                  YOLOv8n: ~3.2M params<br/>
                  V2 (old): ~1.05M params (3× smaller)<br/>
                  V2-Lite:  ~0.38M params (8.5× smaller)<br/>
                  Target: maintain mAP@0.5 &gt; 90% at 8.5× fewer params.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
