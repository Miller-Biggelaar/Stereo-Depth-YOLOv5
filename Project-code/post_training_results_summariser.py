# --- Post-Training Results Summarizer (CSV Version) with Best and Last Epoch Metrics + Report Saving + Column Strip Fix ---

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Path to YOLOv5 training results folder
RESULTS_DIR = './yolov5/runs/val/real_test'

def summarize_results():
    results_csv = os.path.join(RESULTS_DIR, 'results.csv')
    results_png = os.path.join(RESULTS_DIR, 'results.png')
    pr_curve_png = os.path.join(RESULTS_DIR, 'PR_curve.png')
    summary_txt = os.path.join(RESULTS_DIR, 'summary_report.txt')
    summary_csv = os.path.join(RESULTS_DIR, 'summary_report.csv')

    if not os.path.exists(results_csv):
        print("Results CSV not found. Did you finish training?")
        return

    df = pd.read_csv(results_csv)

    if df.empty:
        print("Results CSV is empty.")
        return

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    last_row = df.iloc[-1]
    best_row = df.loc[df['metrics/mAP_0.5'].idxmax()]

    summary_lines = []
    summary_lines.append(f"=== Results Summary for {RESULTS_DIR}: ===\n")
    summary_lines.append(f"Total Epochs Trained: {len(df)}\n")

    summary_lines.append("--- Metrics from Last Epoch ---\n")
    summary_lines.append(f"mAP@0.5: {last_row['metrics/mAP_0.5']:.4f}\n")
    summary_lines.append(f"mAP@0.5:0.95: {last_row['metrics/mAP_0.5:0.95']:.4f}\n")
    summary_lines.append(f"Precision: {last_row['metrics/precision']:.4f}\n")
    summary_lines.append(f"Recall: {last_row['metrics/recall']:.4f}\n")
    summary_lines.append(f"Box Loss: {last_row['val/box_loss']:.4f}\n")
    summary_lines.append(f"Objectness Loss: {last_row['val/obj_loss']:.4f}\n")
    summary_lines.append(f"Classification Loss: {last_row['val/cls_loss']:.4f}\n")

    summary_lines.append("\n--- Metrics from Best Epoch (by mAP@0.5) ---\n")
    summary_lines.append(f"Epoch: {best_row.name}\n")
    summary_lines.append(f"mAP@0.5: {best_row['metrics/mAP_0.5']:.4f}\n")
    summary_lines.append(f"mAP@0.5:0.95: {best_row['metrics/mAP_0.5:0.95']:.4f}\n")
    summary_lines.append(f"Precision: {best_row['metrics/precision']:.4f}\n")
    summary_lines.append(f"Recall: {best_row['metrics/recall']:.4f}\n")
    summary_lines.append(f"Box Loss: {best_row['val/box_loss']:.4f}\n")
    summary_lines.append(f"Objectness Loss: {best_row['val/obj_loss']:.4f}\n")
    summary_lines.append(f"Classification Loss: {best_row['val/cls_loss']:.4f}\n")

    summary_lines.append(f"\nBest Model Saved at: {os.path.join(RESULTS_DIR, 'weights/best.pt')}\n")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(summary_txt, 'w') as f:
        f.write(summary_text)

    summary_df = pd.DataFrame({
        'Metric': ['Epochs Trained',
                   'Last Epoch mAP@0.5', 'Last Epoch mAP@0.5:0.95', 'Last Epoch Precision', 'Last Epoch Recall',
                   'Last Epoch Box Loss', 'Last Epoch Objectness Loss', 'Last Epoch Classification Loss',
                   'Best Epoch Number', 'Best Epoch mAP@0.5', 'Best Epoch mAP@0.5:0.95', 'Best Epoch Precision',
                   'Best Epoch Recall', 'Best Epoch Box Loss', 'Best Epoch Objectness Loss',
                   'Best Epoch Classification Loss'],
        'Value': [len(df),
                  f"{last_row['metrics/mAP_0.5']:.4f}", f"{last_row['metrics/mAP_0.5:0.95']:.4f}",
                  f"{last_row['metrics/precision']:.4f}", f"{last_row['metrics/recall']:.4f}",
                  f"{last_row['val/box_loss']:.4f}", f"{last_row['val/obj_loss']:.4f}", f"{last_row['val/cls_loss']:.4f}",
                  best_row.name, f"{best_row['metrics/mAP_0.5']:.4f}", f"{best_row['metrics/mAP_0.5:0.95']:.4f}",
                  f"{best_row['metrics/precision']:.4f}", f"{best_row['metrics/recall']:.4f}",
                  f"{best_row['val/box_loss']:.4f}", f"{best_row['val/obj_loss']:.4f}", f"{best_row['val/cls_loss']:.4f}"]
    })

    summary_df.to_csv(summary_csv, index=False)

    # Display results.png
    if os.path.exists(results_png):
        img = Image.open(results_png)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Training Loss & Metrics Curves')
        plt.show()
    else:
        print("results.png not found.")

    # Display PR_curve.png
    if os.path.exists(pr_curve_png):
        img = Image.open(pr_curve_png)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Precision-Recall Curve')
        plt.show()
    else:
        print("PR_curve.png not found.")

if __name__ == '__main__':
    summarize_results()