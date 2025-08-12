    def show_parallel_feature_extraction(self):
        # Move all signals to the left and make them BIGGER (0.8 instead of 0.6)
        all_signals = VGroup(self.signal_plots, self.dots, self.dots_text, self.last_signals)
        self.play(all_signals.animate.scale(0.8).shift(LEFT * 4.5), run_time=2)

        # Get signal positions after moving for proper arrow alignment
        signal_positions = []
        # First 4 signals
        for i in range(4):
            y_pos = (2 - i * 1.2) * 0.8 + 1.5  # Account for scale and shift
            signal_positions.append(y_pos)
        
        # Dots position (middle)
        dots_y = -0.2 * 0.8 + 1.5
        
        # Last 2 signals  
        for i in range(2):
            y_pos = (-2.5 - i * 1.2) * 0.8 + 1.5
            signal_positions.append(y_pos)

        # Layout - show 4 individual signals + dots + 2 last signals = 6 total
        lanes_to_show = 6
        x_cnn = 1.2  # per-lane CNN block x
        x_out = 4.8  # feature output x

        # Shared weights box (θ) above lanes
        theta_box = Rectangle(width=2.2, height=0.6, color=YELLOW, fill_opacity=0.15)
        theta_box.move_to([x_cnn, signal_positions[0] + 1.1, 0])
        theta_text = Text("Shared weights θ", font_size=22)
        theta_text.move_to(theta_box.get_center())

        # Per-lane identical CNN blocks + arrows + outputs
        lane_groups = VGroup()
        weight_lines = VGroup()
        out_arrows = VGroup()
        out_feats = VGroup()
        signal_arrows = VGroup()

        # Use real shared feature stats to vary opacity slightly (purely visual)
        shared_feats = data_loader.model_features.get('shared_features', None)
        
        for i in range(lanes_to_show):
            if i < 4:
                # First 4 signals - use actual signal positions
                y = signal_positions[i]
                signal_x = -5.5  # Position of moved signals
                
                # Per-lane CNN block (identical block = same architecture)
                cnn_block = Rectangle(width=1.6, height=0.45, color=GREEN, fill_opacity=0.85)
                cnn_block.move_to([x_cnn, y, 0])
                cnn_label = Text("CNN", font_size=18, color=WHITE).move_to(cnn_block.get_center())

                # Arrow FROM signal plot position TO CNN block
                signal_to_cnn = Arrow(start=[signal_x, y, 0], end=[x_cnn - 0.9, y, 0],
                                    color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)
                
                # Arrow FROM CNN TO output
                to_out = Arrow(start=[x_cnn + 0.9, y, 0], end=[x_out - 0.6, y, 0],
                               color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)

                # Output feature rectangle (64-dim per signal)
                feat = Rectangle(width=0.7, height=0.35, color=ORANGE, fill_opacity=0.8)
                feat.move_to([x_out, y, 0])
                feat_text = Text("64", font_size=16).move_to(feat.get_center())

                # Slight opacity cue from real data (no coupling between lanes)
                if shared_feats is not None and i < len(shared_feats):
                    val = float(np.mean(shared_feats[i]))
                    feat.set_fill(ORANGE, opacity=0.35 + 0.45 * min(1.0, abs(val) / 2.0))

                # Dashed "broadcast" line from shared θ → per-lane CNN (shows weight sharing)
                wline = DashedLine(theta_box.get_bottom(), cnn_block.get_top(), dash_length=0.12, color=YELLOW)

                signal_arrows.add(signal_to_cnn)
                lane_groups.add(VGroup(cnn_block, cnn_label))
                weight_lines.add(wline)
                out_arrows.add(to_out)
                out_feats.add(VGroup(feat, feat_text))
                
            elif i == 4:
                # Dots position - show dots between CNN blocks for hidden signals
                y = dots_y
                
                # Add dots between CNN blocks for hidden signals
                dots_between_cnn = VGroup()
                for dot_i in range(3):
                    dot = Dot(color=YELLOW, radius=0.08)
                    dot.move_to([x_cnn, y - 0.3 + dot_i * 0.15, 0])
                    dots_between_cnn.add(dot)
                lane_groups.add(dots_between_cnn)
                
            else:
                # Last 2 signals - use actual signal positions
                y = signal_positions[i-1]  # Adjust index for last 2 signals
                signal_x = -5.5

                # Per-lane CNN block (identical block = same architecture)
                cnn_block = Rectangle(width=1.6, height=0.45, color=GREEN, fill_opacity=0.85)
                cnn_block.move_to([x_cnn, y, 0])
                cnn_label = Text("CNN", font_size=18, color=WHITE).move_to(cnn_block.get_center())

                # Arrow FROM signal plot position TO CNN block
                signal_to_cnn = Arrow(start=[signal_x, y, 0], end=[x_cnn - 0.9, y, 0],
                                    color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)
                
                # Arrow FROM CNN TO output
                to_out = Arrow(start=[x_cnn + 0.9, y, 0], end=[x_out - 0.6, y, 0],
                               color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)

                # Output feature rectangle (64-dim per signal)
                feat = Rectangle(width=0.7, height=0.35, color=ORANGE, fill_opacity=0.8)
                feat.move_to([x_out, y, 0])
                feat_text = Text("64", font_size=16).move_to(feat.get_center())

                # Slight opacity cue from real data (no coupling between lanes)
                if shared_feats is not None and (i-1) < len(shared_feats):
                    val = float(np.mean(shared_feats[i-1]))
                    feat.set_fill(ORANGE, opacity=0.35 + 0.45 * min(1.0, abs(val) / 2.0))

                # Dashed "broadcast" line from shared θ → per-lane CNN (shows weight sharing)
                wline = DashedLine(theta_box.get_bottom(), cnn_block.get_top(), dash_length=0.12, color=YELLOW)

                signal_arrows.add(signal_to_cnn)
                lane_groups.add(VGroup(cnn_block, cnn_label))
                weight_lines.add(wline)
                out_arrows.add(to_out)
                out_feats.add(VGroup(feat, feat_text))

        # Labels
        lanes_label = Text("×50 lanes in parallel (no mixing)", font_size=22)
        lanes_label.next_to(out_feats, DOWN, buff=0.6)
        out_label = Text("Per-signal feature vectors (64-dim each)", font_size=22)
        out_label.next_to(out_feats, RIGHT, buff=0.4)

        # Animate
        self.play(Create(theta_box), Write(theta_text), run_time=0.6)
        self.play(LaggedStart(*[Create(arrow) for arrow in signal_arrows], lag_ratio=0.05), run_time=1.0)
        self.play(LaggedStart(*[Create(g) for g in lane_groups], lag_ratio=0.05), run_time=1.2)
        self.play(LaggedStart(*[Create(l) for l in weight_lines], lag_ratio=0.05), run_time=0.8)
        self.play(LaggedStart(*[Create(a) for a in out_arrows], lag_ratio=0.05), run_time=0.6)
        self.play(LaggedStart(*[Create(f) for f in out_feats], lag_ratio=0.05), run_time=0.8)
        self.play(Write(lanes_label), Write(out_label), run_time=0.6)

        # Subtle pulse to emphasize simultaneous processing
        for _ in range(2):
            self.play(theta_box.animate.set_fill(opacity=0.25), run_time=0.25)
            self.play(theta_box.animate.set_fill(opacity=0.15), run_time=0.25)

        self.wait(3)
