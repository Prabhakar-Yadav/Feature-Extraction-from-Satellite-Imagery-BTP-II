import http.server
import socketserver
import json
import base64
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import datetime
import pandas as pd

PORT = 8088
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches")
REVIEW_DIR = BASE_DIR / "builtup_reannotation_review"
DATA_DIR = REVIEW_DIR / "data"
V3_DIR = DATA_DIR / "masks_v3_human_verified"
META_PATH = DATA_DIR / "patches_meta.json"
CSV_PATH = REVIEW_DIR / "priority_20_patches_reannotation_stats.csv"

BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review")
BRAIN_V3 = BRAIN_DIR / "data" / "masks_v3_human_verified"
BRAIN_META = BRAIN_DIR / "data" / "patches_meta.json"
BRAIN_CSV = BRAIN_DIR / "priority_20_patches_reannotation_stats.csv"

CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren Land: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta / Coral Red
    4: (140, 50, 190),    # Industrial: Steel Violet / Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
    255: (150, 150, 150)  # Uncertain / Ignore: Neutral Gray
}

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

class AnnotationServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/patches":
            if META_PATH.exists():
                with open(META_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/save_mask":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            
            patch_id = data.get("patch_id")
            mask_data_url = data.get("mask_data") # Base64 raw 8-bit image or class index array
            status = data.get("status", "Corrections saved")
            notes = data.get("notes", "")
            
            if not patch_id or not mask_data_url:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "error", "message": "Missing patch_id or mask_data"}')
                return
                
            # Decode base64 image
            if "," in mask_data_url:
                header, encoded = mask_data_url.split(",", 1)
            else:
                encoded = mask_data_url
                
            img_bytes = base64.b64decode(encoded)
            img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            
            # The canvas can send raw 8-bit (grayscale where pixel values are class IDs)
            # or 4-channel RGBA where R channel or Alpha channel encodes class ID
            if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                # If RGB/RGBA, map colors or read red channel
                # Let's check if the client sent categorical map:
                # In our client, we send a 1-channel PNG or encode class in R channel
                mask_raw = img_np[:, :, 0] # Or class ID
            else:
                mask_raw = img_np
                
            # If client sends class values directly:
            mask_raw = mask_raw.astype(np.uint8)
            
            # Save raw uint8 mask
            out_raw_path1 = V3_DIR / f"{patch_id}.png"
            out_raw_path2 = BRAIN_V3 / f"{patch_id}.png"
            cv2.imwrite(str(out_raw_path1), mask_raw)
            cv2.imwrite(str(out_raw_path2), mask_raw)
            
            # Save colorized mask
            color_mask = mask_to_rgb(mask_raw)
            color_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(V3_DIR / f"{patch_id}_color.png"), color_bgr)
            cv2.imwrite(str(BRAIN_V3 / f"{patch_id}_color.png"), color_bgr)
            
            # Recalculate class distribution
            tot_px = mask_raw.shape[0] * mask_raw.shape[1]
            dist = {int(c): round(float((mask_raw == c).sum() / tot_px * 100), 2) for c in range(7)}
            dist[255] = round(float((mask_raw == 255).sum() / tot_px * 100), 2)
            
            # Update patches_meta.json
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if META_PATH.exists():
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for item in meta:
                    if item["patch_id"] == patch_id:
                        item["status"] = status
                        item["last_modified"] = now_iso
                        item["notes"] = notes
                        item["v3_dist"] = dist
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                with open(BRAIN_META, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                    
            # Update CSV
            if CSV_PATH.exists():
                try:
                    df = pd.read_csv(CSV_PATH)
                    idx = df[df["patch_id"] == patch_id].index
                    if len(idx) > 0:
                        df.loc[idx[0], "review_status"] = status
                        df.loc[idx[0], "proposed_barren_pct"] = dist[2]
                        df.loc[idx[0], "proposed_residential_pct"] = dist[3]
                        df.loc[idx[0], "proposed_industrial_pct"] = dist[4]
                        df.loc[idx[0], "proposed_slum_pct"] = dist[5]
                        df.loc[idx[0], "proposed_total_builtup_pct"] = round(dist[3] + dist[4] + dist[5], 2)
                        df.loc[idx[0], "proposed_uncertain_pct"] = dist[255]
                        df.to_csv(CSV_PATH, index=False)
                        df.to_csv(BRAIN_CSV, index=False)
                except Exception as e:
                    print("CSV update error:", e)
                    
            print(f"[SERVER] Successfully saved human-verified mask for {patch_id} (Status: {status})")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = json.dumps({
                "status": "success",
                "patch_id": patch_id,
                "message": f"Mask saved to disk in masks_v3_human_verified/{patch_id}.png",
                "timestamp": now_iso,
                "distribution": dist
            })
            self.wfile.write(resp.encode("utf-8"))
            return
            
        super().do_POST()

def run_server():
    print(f"Starting Interactive HITL Annotation Server on http://localhost:{PORT}...")
    print(f"Serving directory: {BASE_DIR}")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AnnotationServerHandler) as httpd:
        print(f"HITL Annotation Server running at http://localhost:{PORT}/builtup_reannotation_review/builtup_reannotation_portal.html")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()

