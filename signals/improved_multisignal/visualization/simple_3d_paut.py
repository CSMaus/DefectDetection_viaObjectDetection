from manim import *
import numpy as np
from scipy import interpolate
import os, sys

sys.path.append(os.path.dirname(__file__))
from real_data_loader import get_real_data_sample

config.renderer = "opengl"
config.disable_caching = True
config.frame_rate = 24
config.pixel_width = 960
config.pixel_height = 540

class SimplePAUT3D(ThreeDScene):
    def construct(self):
        # Use original file selection method
        loader, filename = get_real_data_sample()
        if not loader:
            return

        data = loader.load_json_file(filename)
        
        print(f"Loaded file: {filename}")
        
        # Find ranges of consecutive beams and scans that contain defects
        all_beam_ids = sorted(data.keys())
        
        # Load metadata to find defect depth ranges
        try:
            import json
            with open("paut_metadata.json", "r") as f:
                metadata = json.load(f)
            defects = metadata.get("defects", [])
            print(f"Found {len(defects)} defects in metadata")
        except:
            print("No metadata found, using default depth range")
            defects = [{"depth_start": 8.0, "depth_end": 12.0}]
        
        # Find which beams contain defects
        beams_with_defects = []
        for beam_id in all_beam_ids:
            beam_scans = data[beam_id]
            has_defect = False
            
            for scan_key, scan_data in beam_scans.items():
                if isinstance(scan_data, dict) and "signal" in scan_data:
                    signal = np.array(scan_data["signal"])
                elif isinstance(scan_data, list):
                    signal = np.array(scan_data)
                else:
                    continue
                
                # Check defect depth ranges
                depth_per_sample = (11.0 - 0.6) / len(signal)
                
                for defect in defects:
                    start_sample = int((defect["depth_start"] - 0.6) / depth_per_sample)
                    end_sample = int((defect["depth_end"] - 0.6) / depth_per_sample)
                    
                    if start_sample < len(signal) and end_sample < len(signal):
                        defect_region = signal[start_sample:end_sample]
                        if len(defect_region) > 0 and np.max(np.abs(defect_region)) > 0.1:
                            has_defect = True
                            break
                
                if has_defect:
                    break
            
            if has_defect:
                beams_with_defects.append(beam_id)
        
        print(f"Beams with defects: {beams_with_defects}")
        
        # Find the largest consecutive range of beams with defects
        if beams_with_defects:
            beam_nums = [int(bid.replace("BeamIdx_", "")) for bid in beams_with_defects]
            beam_nums.sort()
            
            # Find start and end of consecutive range
            start_beam_num = beam_nums[0]
            end_beam_num = beam_nums[0]
            
            for i in range(1, len(beam_nums)):
                if beam_nums[i] == beam_nums[i-1] + 1:
                    end_beam_num = beam_nums[i]
                else:
                    break
            
            # Take consecutive beams from start to end (limit to 8 for performance)
            consecutive_beam_nums = list(range(start_beam_num, min(start_beam_num + 8, end_beam_num + 1)))
            beam_ids = [f"BeamIdx_{num}" for num in consecutive_beam_nums]
        else:
            # Fallback: use center beams
            total_beams = len(all_beam_ids)
            start_idx = total_beams // 4
            beam_ids = all_beam_ids[start_idx:start_idx + 8]
        
        print(f"Using consecutive beams: {beam_ids}")
        
        # Get ALL scans from first beam (consecutive, no gaps) - limit to 50
        first_beam = beam_ids[0]
        all_scan_keys = sorted(data[first_beam].keys())
        scan_keys = all_scan_keys[:50]  # Take first 50 consecutive scans
        
        print(f"Using {len(scan_keys)} consecutive scans")
        
        # Create axes
        axes = ThreeDAxes(
            x_range=[0, len(scan_keys), 10],
            y_range=[0, len(beam_ids), 2], 
            z_range=[0, 32, 8],  # 32 samples after interpolation
            x_length=6, y_length=3, z_length=3
        )
        
        # Find max amplitude for normalization
        max_amp = 1e-9
        for beam_id in beam_ids:
            for scan_key in scan_keys:
                if scan_key in data[beam_id]:
                    scan_data = data[beam_id][scan_key]
                    if isinstance(scan_data, dict) and "signal" in scan_data:
                        signal = np.array(scan_data["signal"])
                    elif isinstance(scan_data, list):
                        signal = np.array(scan_data)
                    else:
                        continue
                    max_amp = max(max_amp, np.max(np.abs(signal)))
        
        # Create points with interpolated signals
        points = VGroup()
        point_count = 0
        
        for b_idx, beam_id in enumerate(beam_ids):
            for s_idx, scan_key in enumerate(scan_keys):
                if scan_key in data[beam_id]:
                    scan_data = data[beam_id][scan_key]
                    if isinstance(scan_data, dict) and "signal" in scan_data:
                        signal = np.array(scan_data["signal"])
                    elif isinstance(scan_data, list):
                        signal = np.array(scan_data)
                    else:
                        continue
                    
                    # SMART INTERPOLATION - preserve defect peaks
                    # First, find defect regions and their peak locations
                    depth_per_sample = (11.0 - 0.6) / len(signal)
                    defect_peaks = []
                    
                    for defect in defects:
                        start_sample = int((defect["depth_start"] - 0.6) / depth_per_sample)
                        end_sample = int((defect["depth_end"] - 0.6) / depth_per_sample)
                        
                        if start_sample < len(signal) and end_sample < len(signal):
                            defect_region = signal[start_sample:end_sample]
                            if len(defect_region) > 0:
                                # Find peak in defect region
                                peak_idx_local = np.argmax(np.abs(defect_region))
                                peak_idx_global = start_sample + peak_idx_local
                                peak_value = signal[peak_idx_global]
                                defect_peaks.append((peak_idx_global, peak_value))
                    
                    # Create interpolated signal with preserved peaks
                    original_indices = np.arange(len(signal))
                    new_indices = np.linspace(0, len(signal)-1, 32)
                    
                    # Standard interpolation
                    interpolated_signal = np.interp(new_indices, original_indices, signal)
                    
                    # PRESERVE DEFECT PEAKS by finding closest new indices and boosting values
                    for peak_idx, peak_value in defect_peaks:
                        # Find closest new index to the original peak
                        closest_new_idx = np.argmin(np.abs(new_indices - peak_idx))
                        
                        # Ensure the peak is preserved (take max of interpolated and original)
                        if abs(peak_value) > abs(interpolated_signal[closest_new_idx]):
                            interpolated_signal[closest_new_idx] = peak_value
                            
                        # Also boost neighboring points to maintain peak shape
                        if closest_new_idx > 0:
                            interpolated_signal[closest_new_idx-1] = max(
                                interpolated_signal[closest_new_idx-1], 
                                peak_value * 0.7
                            )
                        if closest_new_idx < len(interpolated_signal)-1:
                            interpolated_signal[closest_new_idx+1] = max(
                                interpolated_signal[closest_new_idx+1], 
                                peak_value * 0.7
                            )
                    
                    # Create points for interpolated signal with defect highlighting
                    for d_idx, amp in enumerate(interpolated_signal):
                        amp_abs = abs(amp)
                        
                        # Skip near-zero values
                        if amp_abs < 1e-6:
                            continue
                        
                        # Linear opacity mapping (better for low amplitude defects)
                        normalized_amp = amp_abs / max_amp
                        opacity = normalized_amp  # Linear instead of exponential
                        
                    # Create points for COMPLETE interpolated signal (not just defect regions)
                    for d_idx, amp in enumerate(interpolated_signal):
                        amp_abs = abs(amp)
                        
                        # Skip near-zero values
                        if amp_abs < 1e-6:
                            continue
                        
                        # Linear opacity mapping (better for low amplitude defects)
                        normalized_amp = amp_abs / max_amp
                        opacity = normalized_amp  # Linear instead of exponential
                        
                        if opacity > 0.05:  # Only show significant points
                            # Show ALL signal points - just white with transparency
                            point = Dot3D(
                                point=axes.c2p(s_idx, b_idx, d_idx),
                                radius=0.05,
                                color=WHITE  # Only white color
                            )
                            point.set_opacity(max(0.1, opacity))
                            points.add(point)
                            point_count += 1
        
        print(f"Created {point_count} points for visualization")
        
        # Title
        title = Text("PAUT 3D: Consecutive Beams with Defects", font_size=20)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP)
        
        # Animation - fly camera DOWN to show thickness/depth
        # Start from TOP looking down (small phi = high up)
        self.set_camera_orientation(phi=95 * DEGREES, theta=30 * DEGREES)  # High up (small phi)
        self.add(axes)
        self.add(points)
        self.wait(1)
        
        # Fly DOWN to medium angle (increase phi)
        self.move_camera(phi=35 * DEGREES, theta=60 * DEGREES, run_time=3)  # Medium angle
        self.wait(1)
        
        # Continue flying DOWN to low angle (increase phi more)
        self.move_camera(phi=55 * DEGREES, theta=120 * DEGREES, run_time=3)  # Lower angle
        self.wait(1)
        
        # Final view from very low angle to show thickness (large phi)
        self.move_camera(phi=85 * DEGREES, theta=180 * DEGREES, run_time=3)  # Very low angle
        self.wait(1)
