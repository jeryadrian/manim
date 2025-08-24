def animate_selection_process(self):
    new_title = Text("Step 3: Greedy Selection Process", font_size=40).to_edge(UP)
    self.play(Transform(self.current_title, new_title))
    
    active_tss_vertices, tss_markers, current_routes = [], VGroup(), VGroup()
    # Store the assignment of each origin to TSS index
    origin_assignments = {} # origin_idx -> tss_idx

    for iteration, (_, tss_location) in enumerate(self.optimal_tss_gdf.iterrows()):
        print(f"Animating TSS selection {iteration + 1}/{len(self.optimal_tss_gdf)}")
        tss_vertex, network_pos = self.snap_to_network(tss_location.geometry)
        
        if tss_vertex is not None:
            active_tss_vertices.append(tss_vertex)
            marker_color = self.TSS_MARKER_COLORS[iteration % len(self.TSS_MARKER_COLORS)]
            network_manim_pos = self.transform_coords(network_pos[0], network_pos[1])
            tss_marker = Star(n=6, outer_radius=self.TSS_SIZE, color=marker_color, fill_opacity=1.0).move_to(network_manim_pos)
            self.play(ShowPassingFlash(Circle(radius=self.TSS_SIZE * 2, color=marker_color, stroke_width=6).move_to(network_manim_pos), time_width=0.5), Create(tss_marker))
            tss_markers.add(tss_marker)
            
            new_routes_info = self.calculate_routes(active_tss_vertices)
            print(f"  Found {len(new_routes_info)} routes")
            self.play(FadeOut(current_routes), run_time=0.5)
            
            stable_routes_vgroup = VGroup()
            switched_routes_vgroup = VGroup() # Routes that switched to the NEWEST TSS
            
            # Store previous assignments before calculating new ones
            previous_assignments = origin_assignments.copy()
            new_assignments = {}
            newest_tss_index = len(active_tss_vertices) - 1  # Index of the just-added TSS
            
            for route_path, tss_index, origin_idx in new_routes_info:
                if not route_path: continue
                new_assignments[origin_idx] = tss_index
                
                route_color = self.TSS_ROUTE_COLORS[tss_index % len(self.TSS_ROUTE_COLORS)]
                manim_coords = [self.transform_coords(x, y) for x, y in [self.G.vs[v]['pos'] for v in route_path]]
                line = VMobject().set_points_as_corners(manim_coords).set_stroke(color=route_color, width=2.5, opacity=0.6)

                # Check if the origin switched its assignment to the NEWEST TSS
                previous_tss = previous_assignments.get(origin_idx)
                
                # An origin "switched" if:
                # 1. It's the first iteration (all routes are new), OR
                # 2. It had a different assignment before AND now uses the newest TSS
                is_switch = (iteration == 0) or (previous_tss != tss_index and tss_index == newest_tss_index)
                
                if is_switch:
                    switched_routes_vgroup.add(line)
                else:
                    stable_routes_vgroup.add(line)
            
            # Update assignments for the next iteration
            origin_assignments = new_assignments

            # Animate stable routes first (quick), then highlight the switched ones
            if len(stable_routes_vgroup) > 0:
                self.play(Create(stable_routes_vgroup), run_time=0.75)
            
            if len(switched_routes_vgroup) > 0:
                # Create a more dramatic highlight effect for switched routes
                self.play(
                    Succession(
                        Create(switched_routes_vgroup, lag_ratio=0.02),
                        # Pulse effect to show the "capture"
                        AnimationGroup(
                            switched_routes_vgroup.animate.set_stroke(opacity=1.0, width=5),
                            Flash(tss_marker, color=marker_color, flash_radius=0.3, num_lines=12)
                        ),
                        Wait(0.5),
                        switched_routes_vgroup.animate.set_stroke(opacity=0.7, width=2.5),
                    ),
                    run_time=3.0
                )
            else:
                self.wait(1.0)  # If no switches, just wait a bit

            current_routes = VGroup(stable_routes_vgroup, switched_routes_vgroup)
            
            progress_text = Text(f"TSS {iteration + 1}/{len(self.optimal_tss_gdf)} placed", font_size=32, color=YELLOW).to_edge(RIGHT + UP).shift(DOWN * 1)
            if iteration == 0: 
                self.play(Write(progress_text))
                self.progress_text = progress_text
            else: 
                self.play(Transform(self.progress_text, progress_text))
            
            self.wait(1.5)
    
    self.tss_markers = tss_markers