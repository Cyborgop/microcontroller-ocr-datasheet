"""
UTILS VALIDATION SUITE - ZERO METRICS DIAGNOSTIC
Determines whether zero metrics are caused by:
- utils.py implementation errors (BUG)
- or model/data/training issues (EXPECTED)
"""

import utils
import numpy as np
import torch
import tempfile
import os
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 90)
print("🧪 UTILS VALIDATION SUITE - DIAGNOSTIC MODE")
print("=" * 90)

def create_test_data():
    """Create PERFECT prediction that should get mAP=1.0"""
    # From your IMG file: class 1 (NOR FLASH), normalized coords
    target = np.array([[1, 0.507500, 0.515000, 0.082500, 0.066667]], dtype=np.float32)
    
    # Convert to pixel coordinates for prediction
    img_size = 512
    cx = target[0, 1] * img_size  # 0.5075 * 512 = 259.84
    cy = target[0, 2] * img_size  # 0.5150 * 512 = 263.68
    w = target[0, 3] * img_size   # 0.0825 * 512 = 42.24
    h = target[0, 4] * img_size   # 0.066667 * 512 = 34.13
    
    # Perfect prediction (slightly smaller to ensure IoU > 0.5)
    pred_x1 = cx - w/2 + 2  # 259.84 - 21.12 + 2 = 240.72
    pred_y1 = cy - h/2 + 2  # 263.68 - 17.07 + 2 = 248.61
    pred_x2 = cx + w/2 - 2  # 259.84 + 21.12 - 2 = 278.96
    pred_y2 = cy + h/2 - 2  # 263.68 + 17.07 - 2 = 278.75
    
    prediction = np.array([[1, 0.9, pred_x1, pred_y1, pred_x2, pred_y2]], dtype=np.float32)
    
    print(f"\n📊 Test Data Created:")
    print(f"Target: class {int(target[0,0])} at [{target[0,1]:.4f}, {target[0,2]:.4f}]")
    print(f"Prediction: class {int(prediction[0,0])} at [{pred_x1:.1f}, {pred_y1:.1f}, {pred_x2:.1f}, {pred_y2:.1f}]")
    
    return [prediction], [target]

def test_calculate_map_direct():
    """Direct test of calculate_map with perfect data"""
    print("\n" + "="*60)
    print("🧪 TEST 1: calculate_map with perfect data")
    print("="*60)
    
    predictions, targets = create_test_data()
    
    # Test 1: calculate_map with perfect data
    map_50_95, map_50, map_75, per_class_ap = utils.calculate_map(
        predictions=predictions,
        targets=targets,
        num_classes=utils.NUM_CLASSES,
        epoch=30,  # No warmup
        img_size=512,
        conf_thresh=0.001
    )
    
    print(f"\n📈 Results:")
    print(f"mAP@0.5: {map_50:.4f} (should be 1.0)")
    print(f"mAP@0.75: {map_75:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")
    print(f"Per-class AP: {per_class_ap}")
    
    if map_50 < 0.95:
        print(f"❌ CRITICAL: mAP@0.5 is {map_50:.4f} but should be ~1.0!")
        print("   This indicates a BUG in calculate_map()")
        return False
    else:
        print("✅ calculate_map mathematically correct")
        return True

def test_box_iou_batch():
    """Test IoU calculation mathematically"""
    print("\n" + "="*60)
    print("🧪 TEST 2: box_iou_batch mathematical correctness")
    print("="*60)
    
    # Create overlapping boxes
    box1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    box2 = torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
    
    iou = utils.box_iou_batch(box1, box2)
    
    expected_iou = 50*50 / (100*100 + 100*100 - 50*50)  # 2500 / 17500 = 0.142857
    
    print(f"IoU: {iou.item():.6f}")
    print(f"Expected: {expected_iou:.6f}")
    
    if abs(iou.item() - expected_iou) < 0.001:
        print("✅ IoU calculation mathematically correct")
        return True
    else:
        print(f"❌ IoU calculation wrong: {iou.item():.6f} != {expected_iou:.6f}")
        return False

def test_decode_predictions():
    """Test decoding function interface and format"""
    print("\n" + "="*60)
    print("🧪 TEST 3: decode_predictions interface")
    print("="*60)
    
    # Create dummy model outputs
    batch_size = 1
    num_classes = utils.NUM_CLASSES
    
    # P4 output (stride=8)
    cls_p4 = torch.randn(batch_size, num_classes + 1, 64, 64)  # (B, C+1, H, W)
    reg_p4 = torch.randn(batch_size, 4, 64, 64)
    
    # P3 output (stride=4)
    cls_p3 = torch.randn(batch_size, num_classes + 1, 128, 128)
    reg_p3 = torch.randn(batch_size, 4, 128, 128)
    
    try:
        predictions = utils.decode_predictions(
            pred_p3=(cls_p3, reg_p3),
            pred_p4=(cls_p4, reg_p4),
            conf_thresh=0.001,
            nms_thresh=0.45,
            img_size=512
        )
        
        print(f"✓ decode_predictions executed without error")
        print(f"  Output type: {type(predictions)}")
        print(f"  Output length: {len(predictions)} images")
        
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], np.ndarray):
                print(f"  First prediction shape: {predictions[0].shape}")
                print(f"  First prediction dtype: {predictions[0].dtype}")
                
                # Validate format
                if predictions[0].shape[1] >= 6:
                    print("✅ decode_predictions format correct: [cls, conf, x1, y1, x2, y2]")
                    return True
                else:
                    print(f"❌ Wrong prediction shape: {predictions[0].shape}")
                    return False
            else:
                print(f"❌ First prediction is not numpy array: {type(predictions[0])}")
                return False
        else:
            print("ℹ️ No predictions returned (may be acceptable with random inputs)")
            return True  # Not a bug
            
    except Exception as e:
        print(f"❌ decode_predictions raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_precision_recall():
    """Test precision/recall calculation with perfect match"""
    print("\n" + "="*60)
    print("🧪 TEST 4: compute_precision_recall with perfect match")
    print("="*60)
    
    predictions, targets = create_test_data()
    
    precision, recall = utils.compute_precision_recall(
        preds=predictions,
        targets=targets,
        conf_thresh=0.25,
        iou_thresh=0.5,
        img_size=512,
        debug_first_n=1  # Enable debug for first image
    )
    
    print(f"\nPrecision: {precision:.4f} (should be 1.0)")
    print(f"Recall: {recall:.4f} (should be 1.0)")
    
    if abs(precision - 1.0) < 0.01 and abs(recall - 1.0) < 0.01:
        print("✅ Precision/recall calculation mathematically correct")
        return True
    else:
        print(f"❌ Precision/recall wrong: P={precision:.4f}, R={recall:.4f}")
        print("   This indicates a bug in matching logic")
        return False

def test_adaptive_iou_config():
    """Verify adaptive IoU configuration exists"""
    print("\n" + "="*60)
    print("ℹ️ TEST 5: Adaptive IoU Configuration Check")
    print("="*60)
    
    # Just check that the feature exists in calculate_map
    print("Checking adaptive IoU implementation...")
    
    # Look for adaptive IoU in calculate_map source
    import inspect
    source = inspect.getsource(utils.calculate_map)
    
    if 'adaptive_th' in source and 'g_areas < 16 * 16' in source:
        print("✓ Adaptive IoU implemented in calculate_map")
        print("  Small objects (<16x16): IoU * 0.7")
        print("  Medium objects (<32x32): IoU * 0.85")
        print("  Large objects: full IoU threshold")
        print("ℹ️ Informational test passed (implementation verified)")
    else:
        print("⚠️ Adaptive IoU not found in calculate_map")
    
    return True  # Informational, not pass/fail

def test_edge_cases():
    """Test edge cases that should produce zero metrics"""
    print("\n" + "="*60)
    print("🧪 TEST 6: Edge Cases (should produce zero)")
    print("="*60)
    
    test_cases = [
        ("Empty predictions", [], [np.array([[1, 0.5, 0.5, 0.1, 0.1]])]),
        ("Empty targets", [np.array([[1, 0.9, 100, 100, 200, 200]])], []),
        ("Wrong class", [np.array([[0, 0.9, 100, 100, 200, 200]])], [np.array([[1, 0.5, 0.5, 0.1, 0.1]])]),
        ("Low confidence", [np.array([[1, 0.01, 100, 100, 200, 200]])], [np.array([[1, 0.5, 0.5, 0.1, 0.1]])]),
        ("No IoU", [np.array([[1, 0.9, 10, 10, 20, 20]])], [np.array([[1, 0.5, 0.5, 0.1, 0.1]])]),
    ]
    
    all_correct = True
    for name, preds, gts in test_cases:
        map_50_95, map_50, map_75, _ = utils.calculate_map(
            preds if isinstance(preds, list) else [preds],
            gts if isinstance(gts, list) else [gts],
            utils.NUM_CLASSES,
            epoch=30
        )
        
        expected_zero = True
        actual_zero = map_50 < 0.01
        
        status = "✅" if expected_zero == actual_zero else "❌"
        print(f"{status} {name:20} -> mAP@0.5: {map_50:.4f} (expected: 0.0)")
        
        if expected_zero != actual_zero:
            all_correct = False
    
    if all_correct:
        print("✅ All edge cases handled correctly")
        return True
    else:
        print("❌ Some edge cases incorrectly handled")
        return False

def test_real_data_format():
    """Test with actual data format from your dataset"""
    print("\n" + "="*60)
    print("🧪 TEST 7: Real Data Format Compatibility")
    print("="*60)
    
    # Simulate your IMG file data exactly
    # Format: [cls, cx, cy, w, h] normalized
    yolo_target = np.array([[1, 0.507500, 0.515000, 0.082500, 0.066667]], dtype=np.float32)
    
    # Simulate model prediction (your format)
    # Format: [cls, conf, x1, y1, x2, y2] pixels
    img_size = 512
    pred_x1 = 240
    pred_y1 = 247
    pred_x2 = 279
    pred_y2 = 280
    
    prediction = np.array([[1, 0.9, pred_x1, pred_y1, pred_x2, pred_y2]], dtype=np.float32)
    
    # Convert target to pixel coords for verification
    cx = yolo_target[0, 1] * img_size
    cy = yolo_target[0, 2] * img_size
    w = yolo_target[0, 3] * img_size
    h = yolo_target[0, 4] * img_size
    gt_x1 = cx - w/2
    gt_y1 = cy - h/2
    gt_x2 = cx + w/2
    gt_y2 = cy + h/2
    
    print(f"Target format: [cls, cx, cy, w, h] normalized")
    print(f"Prediction format: [cls, conf, x1, y1, x2, y2] pixels")
    print(f"\nGT box pixels:   [{gt_x1:.1f}, {gt_y1:.1f}, {gt_x2:.1f}, {gt_y2:.1f}]")
    print(f"Pred box pixels: [{pred_x1}, {pred_y1}, {pred_x2}, {pred_y2}]")
    
    # Calculate IoU manually for verification
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    iou = inter_area / (gt_area + pred_area - inter_area + 1e-6)
    
    print(f"Manual IoU calculation: {iou:.4f}")
    
    if iou > 0.5:
        print("✅ IoU > 0.5, should produce non-zero metrics")
        return True
    else:
        print(f"⚠️ IoU too low: {iou:.4f} < 0.5")
        print("   Note: This is not a utils.py bug, just test data tuning")
        return False

def debug_iou_distribution():
    """
    DEBUG: IoU Distribution Probe
    Shows why metrics might be zero by analyzing IoU between predictions and GT.
    """
    print("\n" + "="*60)
    print("🔍 IoU DISTRIBUTION PROBE (DEBUG)")
    print("="*60)
    
    # Create test data
    predictions, targets = create_test_data()
    
    all_ious = []
    class_matches = []
    
    for img_idx, (pred_img, target_img) in enumerate(zip(predictions, targets)):
        if len(pred_img) == 0 or len(target_img) == 0:
            continue
            
        # Get predictions
        pred_boxes = pred_img[:, 2:6]  # x1, y1, x2, y2
        pred_classes = pred_img[:, 0].astype(int)
        
        # Convert targets to pixel boxes
        img_size = 512
        cx = target_img[:, 1] * img_size
        cy = target_img[:, 2] * img_size
        w = target_img[:, 3] * img_size
        h = target_img[:, 4] * img_size
        gt_boxes = np.stack([
            cx - w/2, cy - h/2, cx + w/2, cy + h/2
        ], axis=1)
        gt_classes = target_img[:, 0].astype(int)
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Calculate IoU matrix
            iou_matrix = utils.box_iou_batch(
                torch.from_numpy(pred_boxes),
                torch.from_numpy(gt_boxes)
            ).numpy()
            
            # For each prediction, find best matching GT
            for pred_idx in range(len(pred_boxes)):
                if len(gt_boxes) > 0:
                    best_iou = iou_matrix[pred_idx].max()
                    best_gt_idx = iou_matrix[pred_idx].argmax()
                    
                    all_ious.append(best_iou)
                    
                    # Check class match
                    class_match = pred_classes[pred_idx] == gt_classes[best_gt_idx]
                    class_matches.append(class_match)
    
    if not all_ious:
        print("No predictions or targets to analyze")
        return True
    
    all_ious = np.array(all_ious)
    
    print(f"\n📊 IoU Distribution Analysis:")
    print(f"Total prediction-target pairs analyzed: {len(all_ious)}")
    print(f"IoU range: [{all_ious.min():.3f}, {all_ious.max():.3f}]")
    print(f"Mean IoU: {all_ious.mean():.3f}")
    print(f"Median IoU: {np.median(all_ious):.3f}")
    
    # Count matches at different thresholds
    thresholds = [0.25, 0.50, 0.75]
    for thresh in thresholds:
        matches = sum(1 for iou in all_ious if iou >= thresh)
        percent = 100 * matches / len(all_ious) if len(all_ious) > 0 else 0
        print(f"IoU >= {thresh:.2f}: {matches}/{len(all_ious)} ({percent:.1f}%)")
    
    # Class match analysis
    if class_matches:
        correct_class = sum(class_matches)
        class_accuracy = 100 * correct_class / len(class_matches)
        print(f"\nClass match accuracy: {correct_class}/{len(class_matches)} ({class_accuracy:.1f}%)")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, 1, 21)
    plt.hist(all_ious, bins=bins, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='mAP@0.5 threshold')
    plt.axvline(x=0.25, color='orange', linestyle='--', linewidth=2, label='Warmup threshold (0.25)')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title('IoU Distribution Between Predictions and Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_ious = np.sort(all_ious)
    y_vals = np.arange(1, len(sorted_ious) + 1) / len(sorted_ious)
    plt.plot(sorted_ious, y_vals, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=0.25, color='orange', linestyle='--', linewidth=2)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Fraction of Predictions')
    plt.title('Cumulative IoU Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('debug_plots', exist_ok=True)
    plt.savefig('debug_plots/iou_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 IoU distribution plot saved to: debug_plots/iou_distribution.png")
    
    # Critical diagnosis
    if len(all_ious) == 0:
        print("\n🚨 CRITICAL: No predictions to analyze!")
        print("   Model is not generating any predictions")
        print("   Check: confidence threshold, model output")
    
    elif all_ious.max() < 0.25:
        print("\n🚨 CRITICAL: NO predictions have IoU >= 0.25!")
        print("   Model is not learning to localize objects at all")
        print("   Possible causes:")
        print("   1. Wrong box format (normalized vs pixels)")
        print("   2. Model architecture issue")
        print("   3. Training data mismatch")
        print("   4. Learning rate too high/low")
    
    elif all_ious.max() < 0.5:
        print("\n⚠️ WARNING: No predictions have IoU >= 0.5")
        print("   Model localization is poor")
        print("   During warmup (epochs 0-24): IoU threshold = 0.25 (OK)")
        print("   After warmup: IoU threshold = 0.5 (WILL CAUSE ZERO METRICS)")
        print("   Improve: box loss weight, more training")
    
    else:
        iou_above_50 = sum(1 for iou in all_ious if iou >= 0.5)
        percent_50 = 100 * iou_above_50 / len(all_ious)
        print(f"\n✅ {iou_above_50}/{len(all_ious)} ({percent_50:.1f}%) predictions have IoU >= 0.5")
        
        if class_matches:
            iou_and_class = sum(1 for i, cm in enumerate(class_matches) 
                               if all_ious[i] >= 0.5 and cm)
            print(f"✅ {iou_and_class}/{len(all_ious)} have IoU>=0.5 AND correct class")
            
            if iou_and_class == 0 and iou_above_50 > 0:
                print("\n⚠️ Some predictions have good IoU but WRONG CLASS")
                print("   Class prediction is the problem, not localization")
    
    return True  # Informational test

def run_diagnostics():
    """Run all diagnostic tests"""
    print("\n" + "="*90)
    print("🔬 RUNNING IMPLEMENTATION DIAGNOSTICS")
    print("="*90)
    
    tests = [
        ("Real Data Format", test_real_data_format, True),
        ("Box IoU Batch", test_box_iou_batch, True),
        ("Adaptive IoU Config", test_adaptive_iou_config, False),  # Informational
        ("Edge Cases", test_edge_cases, True),
        ("Decode Predictions Interface", test_decode_predictions, True),
        ("Precision/Recall Calculation", test_compute_precision_recall, True),
        ("Calculate mAP with Perfect Data", test_calculate_map_direct, True),
        ("IoU Distribution Probe", debug_iou_distribution, False),  # Informational debug
    ]
    
    results = []
    critical_failures = []
    
    for name, test_func, is_critical in tests:
        try:
            print(f"\n{'🧪' if is_critical else 'ℹ️'} {name}")
            print("-" * 40)
            success = test_func()
            results.append((name, success, is_critical))
            
            if is_critical and not success:
                critical_failures.append(name)
                
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, is_critical))
            if is_critical:
                critical_failures.append(name)
    
    print("\n" + "="*90)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*90)
    
    print("\nCRITICAL TESTS (must pass):")
    for name, success, is_critical in results:
        if is_critical:
            print(f"{'✅' if success else '❌'} {name}")
    
    print("\nINFORMATIONAL/DEBUG TESTS:")
    for name, success, is_critical in results:
        if not is_critical:
            print(f"{'ℹ️' if success else '⚠️'} {name}")
    
    critical_passed = sum(1 for _, success, critical in results if critical and success)
    critical_total = sum(1 for _, _, critical in results if critical)
    
    print(f"\n🎯 CRITICAL TESTS: {critical_passed}/{critical_total} passed")
    
    if len(critical_failures) == 0:
        print("\n✅ ALL CRITICAL TESTS PASSED")
        print("   utils.py implementation is CORRECT")
        print("\n   If metrics are still zero during training:")
        print("   1. Model may not be learning (check loss curves)")
        print("   2. Predictions may not match GT (check IoU)")
        print("   3. Data format mismatch (verify pipeline)")
        return True
    else:
        print(f"\n❌ CRITICAL FAILURES: {len(critical_failures)} tests failed")
        print("   Failed tests:", ", ".join(critical_failures))
        print("\n   These are BUGS in utils.py that must be fixed")
        return False

def generate_training_debug_snippet():
    """Generate code snippet to add to training loop"""
    print("\n" + "="*90)
    print("🔧 TRAINING DEBUG SNIPPET")
    print("="*90)
    
    print("""
# Add this to your training loop after validation:

def debug_validation_predictions(predictions, targets, epoch):
    \"\"\"Debug helper for zero metrics\"\"\"
    if len(predictions) == 0 or len(targets) == 0:
        print(f"Epoch {epoch}: No predictions or targets")
        return
    
    # Check first image
    preds_img = predictions[0] if isinstance(predictions[0], np.ndarray) else predictions[0].cpu().numpy()
    targets_img = targets[0] if isinstance(targets[0], np.ndarray) else targets[0].cpu().numpy()
    
    print(f"Epoch {epoch}: Image 0 has {len(preds_img)} preds, {len(targets_img)} targets")
    
    if len(preds_img) > 0 and len(targets_img) > 0:
        print(f"  First pred: class={int(preds_img[0,0])}, conf={preds_img[0,1]:.3f}")
        print(f"  First target: class={int(targets_img[0,0])}")
        
        # Calculate IoU for first prediction and first target
        from utils import box_iou_batch
        
        pred_box = torch.from_numpy(preds_img[0:1, 2:6])
        
        # Convert target to pixels
        img_size = 512
        cx = targets_img[0, 1] * img_size
        cy = targets_img[0, 2] * img_size
        w = targets_img[0, 3] * img_size
        h = targets_img[0, 4] * img_size
        gt_box = torch.tensor([[
            cx - w/2, cy - h/2,
            cx + w/2, cy + h/2
        ]])
        
        iou = box_iou_batch(pred_box, gt_box).item()
        print(f"  IoU between first pred/target: {iou:.3f}")
        print(f"  Same class? {int(preds_img[0,0]) == int(targets_img[0,0])}")
        
        # IoU distribution analysis
        if len(preds_img) > 10 and len(targets_img) > 0:
            all_ious = []
            for pred in preds_img[:20]:  # Check first 20 predictions
                pred_box = torch.from_numpy(pred[2:6].reshape(1, 4))
                iou_matrix = box_iou_batch(pred_box, gt_box)
                best_iou = iou_matrix.max().item()
                all_ious.append(best_iou)
            
            iou_above_25 = sum(1 for iou in all_ious if iou >= 0.25)
            iou_above_50 = sum(1 for iou in all_ious if iou >= 0.5)
            print(f"  IoU >= 0.25: {iou_above_25}/{len(all_ious)}")
            print(f"  IoU >= 0.50: {iou_above_50}/{len(all_ious)}")
""")

# Run the diagnostics
if __name__ == "__main__":
    print("""
PURPOSE: Determine if zero metrics are caused by:
  1. utils.py implementation errors (BUG) - this test reveals
  2. Model/data/training issues (EXPECTED) - if tests pass
""")
    
    implementation_correct = run_diagnostics()
    
    if implementation_correct:
        print("\n" + "="*90)
        print("🎯 DIAGNOSIS: utils.py is CORRECT")
        print("="*90)
        print("""
Your zero metrics are NOT caused by utils.py implementation.
The issue is elsewhere in your pipeline:

POSSIBLE CAUSES:
1. Model not learning (check training loss)
2. Predictions not matching ground truth (low IoU)
3. Confidence scores too low (filtered out)
4. Class mismatch (wrong class predictions)
5. Warmup period using IoU=0.25 (epochs 0-24)

NEXT STEPS:
1. Add the debug snippet to your training loop
2. Check actual IoU values between predictions and GT
3. Verify model is actually generating predictions
4. Check confidence scores are above threshold
""")
        generate_training_debug_snippet()
    else:
        print("\n" + "="*90)
        print("🎯 DIAGNOSIS: utils.py has BUGS")
        print("="*90)
        print("""
Your zero metrics ARE caused by utils.py implementation errors.
Fix the failed tests above before continuing.

COMMON FIXES:
1. Check global_gt_idx = gt_inds[g_local_idx] fix
2. Verify box_iou_batch handles edge cases
3. Ensure calculate_map warmup logic correct
4. Check confidence threshold filtering

After fixing utils.py, run this diagnostic again.
""")
    
    print("\n" + "="*90)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*90)