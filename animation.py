"""
Manim Animation for Logistics Optimization Algorithm Visualization
Fixed road visualization and route display issues.
"""

from manim import *
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import igraph as ig
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

class LogisticsOptimizationAnimation(Scene):
    def construct(self):
        # Configuration
        self.setup_config()
        
        # Load and process data
        self.load_data()
        self.build_network_graph()
        
        # Animation sequence
        self.show_title()
        self.setup_base_map()
        self.show_pool_and_facilities()
        self.animate_selection_process()
        self.show_final_state()
        
    def setup_config(self):
        """Configure animation parameters"""
        self.TARGET_CRS = 'EPSG:28992'
        self.SCALE_FACTOR = 0.001  # Will be auto-calculated
        self.MAX_SNAP_DISTANCE = 500  # meters
        
        # Colors
        self.ROAD_COLOR = WHITE
        self.POOL_COLOR = BLUE
        self.ORIGIN_COLOR = RED
        self.CCH_COLOR = GREEN
        self.TSS_COLOR = YELLOW
        self.ROUTE_COLOR = ORANGE
        
        # Sizes
        self.POOL_SIZE = 0.05
        self.ORIGIN_SIZE = 0.08
        self.CCH_SIZE = 0.12
        self.TSS_SIZE = 0.15
        
    def load_data(self):
        """Load all geospatial data files"""
        try:
            # Load input files
            self.roads_gdf = gpd.read_file('input/road.gpkg').to_crs(self.TARGET_CRS)
            self.grid_gdf = gpd.read_file('input/grid.gpkg').to_crs(self.TARGET_CRS)
            self.origins_gdf = gpd.read_file('input/project.gpkg').to_crs(self.TARGET_CRS)
            self.cch_gdf = gpd.read_file('input/cch.gpkg').to_crs(self.TARGET_CRS)
            
            # Load optimization results
            self.optimal_tss_gdf = gpd.read_file('output/optimal_tss_locations.gpkg').to_crs(self.TARGET_CRS)
            
            print(f"Loaded {len(self.roads_gdf)} roads, {len(self.grid_gdf)} pool points,")
            print(f"{len(self.origins_gdf)} origins, {len(self.cch_gdf)} CCHs,")
            print(f"{len(self.optimal_tss_gdf)} optimal TSS locations")
            
            # Calculate bounds for scaling
            all_geometries = []
            for gdf in [self.roads_gdf, self.grid_gdf, self.origins_gdf, self.cch_gdf]:
                all_geometries.extend(gdf.geometry.tolist())
            
            # Get overall bounds
            combined_gdf = gpd.GeoSeries(all_geometries)
            bounds = combined_gdf.total_bounds
            
            self.min_x, self.min_y, self.max_x, self.max_y = bounds
            
            # Center coordinates for transformation
            self.center_x = (self.min_x + self.max_x) / 2
            self.center_y = (self.min_y + self.max_y) / 2
            
            # Calculate extent for proper scaling
            self.extent_x = self.max_x - self.min_x
            self.extent_y = self.max_y - self.min_y
            max_extent = max(self.extent_x, self.extent_y)
            
            # Adjust scale factor to fit in canvas (leaving some margin)
            canvas_size = 10  # Use smaller canvas to leave margin
            self.SCALE_FACTOR = canvas_size / max_extent
            
            print(f"Coordinate bounds: {bounds}")
            print(f"Extent: {self.extent_x:.0f} x {self.extent_y:.0f}")
            print(f"Scale factor: {self.SCALE_FACTOR:.8f}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def transform_coords(self, x, y):
        """Transform real-world coordinates to Manim canvas coordinates"""
        canvas_x = (x - self.center_x) * self.SCALE_FACTOR
        canvas_y = (y - self.center_y) * self.SCALE_FACTOR
        return np.array([canvas_x, canvas_y, 0])
    
    def build_network_graph(self):
        """Build igraph from road network for routing"""
        print("Building network graph for routing...")
        
        # Initialize data structures
        vertices = {}  # coord -> vertex_id mapping
        edges = []     # list of (source, target, weight) tuples
        vertex_coords = []  # list of (x, y) coordinates
        vertex_id = 0
        
        # --- REVISION ---
        # The .sample() method was removed here. Using the full roads_gdf is essential
        # for creating a connected graph that allows for successful routing.
        # Sampling creates a disconnected graph where paths cannot be found.
        road_sample = self.roads_gdf
        
        # Process roads to create network
        roads_processed = 0
        for idx, road in road_sample.iterrows():
            try:
                if road.geometry is None or road.geometry.is_empty:
                    continue
                    
                if road.geometry.geom_type == 'LineString':
                    geometries = [road.geometry]
                elif road.geometry.geom_type == 'MultiLineString':
                    geometries = list(road.geometry.geoms)
                else:
                    continue
                
                # Process each LineString
                for geom in geometries:
                    coords = list(geom.coords)
                    
                    if len(coords) < 2:
                        continue
                    
                    # Add nodes and edges for each road segment
                    for i in range(len(coords) - 1):
                        coord1, coord2 = coords[i], coords[i + 1]
                        
                        if coord1 == coord2:
                            continue
                        
                        # Round coordinates
                        coord1 = (round(coord1[0], 2), round(coord1[1], 2))
                        coord2 = (round(coord2[0], 2), round(coord2[1], 2))
                        
                        # Get or create vertices
                        if coord1 not in vertices:
                            vertices[coord1] = vertex_id
                            vertex_coords.append(coord1)
                            vertex_id += 1
                        
                        if coord2 not in vertices:
                            vertices[coord2] = vertex_id
                            vertex_coords.append(coord2)
                            vertex_id += 1
                        
                        # Add edge with length as weight
                        vertex1_id = vertices[coord1]
                        vertex2_id = vertices[coord2]
                        
                        if vertex1_id != vertex2_id:
                            length = Point(coord1).distance(Point(coord2))
                            if length > 0:
                                edges.append((vertex1_id, vertex2_id, length))
                
                roads_processed += 1
                    
            except Exception as e:
                print(f"  Error processing road {idx}: {e}")
                continue
        
        print(f"  Processed {roads_processed} roads")
        
        # Create igraph
        if vertex_id > 0 and edges:
            self.G = ig.Graph(n=vertex_id)
            
            # Add edges
            edge_list = [(e[0], e[1]) for e in edges]
            weights = [e[2] for e in edges]
            
            self.G.add_edges(edge_list)
            self.G.es['weight'] = weights
            
            # Store vertex coordinates as graph attributes
            self.G.vs['pos'] = vertex_coords
            
            # Create spatial index for snapping points to network
            self.node_coords = np.array(vertex_coords)
            self.node_tree = cKDTree(self.node_coords)
            
        else:
            print("  Warning: No valid vertices or edges found")
            self.G = ig.Graph()
            self.node_coords = np.array([])
            self.node_tree = None
        
        print(f"Built igraph with {self.G.vcount()} vertices and {self.G.ecount()} edges")
    
    def snap_to_network(self, point):
        """Snap a point to the nearest network vertex"""
        if self.node_tree is None or len(self.node_coords) == 0:
            return None, None
            
        coord = np.array([point.x, point.y])
        try:
            dist, idx = self.node_tree.query(coord)
            
            if dist <= self.MAX_SNAP_DISTANCE:
                vertex_id = int(idx)
                vertex_pos = self.G.vs[vertex_id]['pos']
                return vertex_id, vertex_pos
        except Exception as e:
            print(f"  Error snapping point: {e}")
            
        return None, None
    
    def calculate_routes(self, active_tss_vertices):
        """Calculate optimal routes for current TSS configuration"""
        routes = []
        
        # Limit number of origins to process for performance
        origins_sample = self.origins_gdf.head(min(20, len(self.origins_gdf)))
        
        for idx, origin in origins_sample.iterrows():
            origin_vertex, origin_pos = self.snap_to_network(origin.geometry)
            if origin_vertex is None:
                continue
            
            best_path = None
            min_dist = float('inf')
            
            # If we have active TSS locations, try routing through them
            if active_tss_vertices:
                for tss_vertex in active_tss_vertices:
                    for _, cch in self.cch_gdf.iterrows():
                        cch_vertex, cch_pos = self.snap_to_network(cch.geometry)
                        if cch_vertex is None:
                            continue
                        
                        try:
                            # Path from origin to TSS
                            path1 = self.G.get_shortest_paths(
                                origin_vertex, 
                                tss_vertex, 
                                weights='weight',
                                output='vpath'
                            )[0]
                            
                            # Path from TSS to CCH
                            path2 = self.G.get_shortest_paths(
                                tss_vertex, 
                                cch_vertex, 
                                weights='weight',
                                output='vpath'
                            )[0]
                            
                            if len(path1) > 1 and len(path2) > 1:
                                # Calculate total distance
                                dist1 = self.G.shortest_paths(
                                    origin_vertex, tss_vertex, weights='weight'
                                )[0][0]
                                dist2 = self.G.shortest_paths(
                                    tss_vertex, cch_vertex, weights='weight'
                                )[0][0]
                                
                                total_dist = dist1 + dist2
                                if total_dist < min_dist:
                                    min_dist = total_dist
                                    # Combine paths (remove duplicate TSS vertex)
                                    best_path = path1 + path2[1:]
                                    break  # Use first valid route found
                        except:
                            continue
                    
                    if best_path:  # Break outer loop too
                        break
            
            # If no TSS route found, try direct route to CCH
            if best_path is None:
                for _, cch in self.cch_gdf.iterrows():
                    cch_vertex, cch_pos = self.snap_to_network(cch.geometry)
                    if cch_vertex is None:
                        continue
                    
                    try:
                        path = self.G.get_shortest_paths(
                            origin_vertex, 
                            cch_vertex, 
                            weights='weight',
                            output='vpath'
                        )[0]
                        
                        if len(path) > 1:
                            dist = self.G.shortest_paths(
                                origin_vertex, cch_vertex, weights='weight'
                            )[0][0]
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_path = path
                                break  # Use first valid route found
                    except:
                        continue
            
            if best_path and len(best_path) > 1:
                routes.append(best_path)
        
        return routes
    
    def show_title(self):
        """Show the animation title"""
        title = Text("Logistics Optimization: TSS Location Selection", font_size=48)
        subtitle = Text("Minimizing Transport Effort Through Strategic Facility Placement", font_size=32)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
    
    def setup_base_map(self):
        """Draw the road network - FIXED to show proper lines"""
        road_lines = VGroup()
        
        print("Creating road network visualization...")
        roads_visualized = 0
        
        # --- REVISION ---
        # The .sample() method was removed. Drawing the full roads_gdf ensures
        # that the road network is displayed as a connected whole, rather than
        # a scattered collection of random segments.
        road_sample = self.roads_gdf
        
        for idx, road in road_sample.iterrows():
            try:
                if road.geometry is None or road.geometry.is_empty:
                    continue
                    
                if road.geometry.geom_type == 'LineString':
                    geometries = [road.geometry]
                elif road.geometry.geom_type == 'MultiLineString':
                    geometries = list(road.geometry.geoms)
                else:
                    continue
                
                # Process each LineString
                for geom in geometries:
                    coords = list(geom.coords)
                    if len(coords) < 2:
                        continue
                    
                    # Transform coordinates
                    manim_coords = [self.transform_coords(x, y) for x, y in coords]
                    
                    # FIXED: Create proper line segments
                    if len(manim_coords) >= 2:
                        # Create line using Line class for simple segments
                        if len(manim_coords) == 2:
                            line = Line(manim_coords[0], manim_coords[1])
                            line.set_stroke(self.ROAD_COLOR, width=1, opacity=0.8)
                            road_lines.add(line)
                        else:
                            # For multi-segment lines, use VMobject
                            line = VMobject()
                            line.set_points_as_corners(manim_coords)
                            line.set_stroke(self.ROAD_COLOR, width=1, opacity=0.8)
                            road_lines.add(line)
                        
                        roads_visualized += 1
                        
            except Exception as e:
                print(f"  Error visualizing road {idx}: {e}")
                continue
        
        print(f"  Created visualization for {roads_visualized} roads")
        
        # Show roads
        title = Text("Step 1: Road Network", font_size=40)
        title.to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(road_lines), run_time=3)
        self.wait(1)
        
        self.road_lines = road_lines
        self.current_title = title
    
    def show_pool_and_facilities(self):
        """Show pool of candidate sites, origins, and destinations"""
        # Update title
        new_title = Text("Step 2: Facilities and Candidate Pool", font_size=40)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        # Sample pool points for better visibility
        max_pool_points = 50  # Even fewer for clarity
        if len(self.grid_gdf) > max_pool_points:
            pool_sample = self.grid_gdf.sample(max_pool_points)
        else:
            pool_sample = self.grid_gdf
        
        # Create pool points
        pool_dots = VGroup()
        for idx, point in pool_sample.iterrows():
            pos = self.transform_coords(point.geometry.x, point.geometry.y)
            dot = Dot(pos, radius=self.POOL_SIZE, color=self.POOL_COLOR, fill_opacity=0.6)
            pool_dots.add(dot)
        
        # Create origin points
        origin_dots = VGroup()
        for idx, origin in self.origins_gdf.iterrows():
            pos = self.transform_coords(origin.geometry.x, origin.geometry.y)
            dot = Dot(pos, radius=self.ORIGIN_SIZE, color=self.ORIGIN_COLOR, fill_opacity=0.8)
            origin_dots.add(dot)
        
        # Create CCH points
        cch_dots = VGroup()
        for idx, cch in self.cch_gdf.iterrows():
            pos = self.transform_coords(cch.geometry.x, cch.geometry.y)
            dot = Dot(pos, radius=self.CCH_SIZE, color=self.CCH_COLOR, fill_opacity=0.8)
            cch_dots.add(dot)
        
        # Show elements with labels
        pool_label = Text("Candidate Pool", font_size=28, color=self.POOL_COLOR)
        origin_label = Text("Material Origins", font_size=28, color=self.ORIGIN_COLOR)
        cch_label = Text("Destinations (CCH)", font_size=28, color=self.CCH_COLOR)
        
        # Position labels
        pool_label.to_edge(LEFT + UP).shift(DOWN * 1)
        origin_label.next_to(pool_label, DOWN, aligned_edge=LEFT)
        cch_label.next_to(origin_label, DOWN, aligned_edge=LEFT)
        
        # Animate appearance
        self.play(
            LaggedStart(
                AnimationGroup(Create(pool_dots), Write(pool_label)),
                AnimationGroup(Create(origin_dots), Write(origin_label)),
                AnimationGroup(Create(cch_dots), Write(cch_label)),
                lag_ratio=0.5
            ),
            run_time=4
        )
        
        self.wait(1)
        
        self.pool_dots = pool_dots
        self.origin_dots = origin_dots
        self.cch_dots = cch_dots
        self.labels = VGroup(pool_label, origin_label, cch_label)
        self.pool_sample = pool_sample
    
    def animate_selection_process(self):
        """Animate the TSS selection process - FIXED route visualization"""
        # Update title
        new_title = Text("Step 3: Greedy Selection Process", font_size=40)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        active_tss_vertices = []
        tss_markers = VGroup()
        current_routes = VGroup()
        
        for iteration, (idx, tss_location) in enumerate(self.optimal_tss_gdf.iterrows()):
            print(f"Animating TSS selection {iteration + 1}/{len(self.optimal_tss_gdf)}")
            
            # Find nearest pool point to the selected TSS location
            tss_point = tss_location.geometry
            distances = [tss_point.distance(pool_point.geometry) for _, pool_point in self.pool_sample.iterrows()]
            
            if not distances:
                continue
                
            nearest_pool_idx = np.argmin(distances)
            
            # Get the corresponding dot from our visualized pool points
            if nearest_pool_idx < len(self.pool_dots):
                selected_pool_dot = self.pool_dots[nearest_pool_idx]
                
                # Flash animation
                flash_circle = Circle(radius=self.POOL_SIZE * 5, color=YELLOW, stroke_width=5)
                flash_circle.move_to(selected_pool_dot.get_center())
                
                self.play(
                    selected_pool_dot.animate.set_color(YELLOW).scale(3),
                    Create(flash_circle),
                    run_time=1.5
                )
                
                self.play(
                    flash_circle.animate.scale(3).set_opacity(0),
                    run_time=1
                )
                
                # Snap to network and create TSS marker
                tss_vertex, network_pos = self.snap_to_network(tss_point)
                if tss_vertex is not None:
                    active_tss_vertices.append(tss_vertex)
                    
                    # Create TSS marker
                    network_manim_pos = self.transform_coords(network_pos[0], network_pos[1])
                    tss_marker = Star(n=6, outer_radius=self.TSS_SIZE, 
                                    color=self.TSS_COLOR, fill_opacity=1.0, 
                                    stroke_color=BLACK, stroke_width=2)
                    tss_marker.move_to(network_manim_pos)
                    
                    self.play(Create(tss_marker), run_time=1)
                    tss_markers.add(tss_marker)
                    
                    # FIXED: Calculate and show routes
                    print(f"  Calculating routes with {len(active_tss_vertices)} TSS locations...")
                    new_routes = self.calculate_routes(active_tss_vertices)
                    print(f"  Found {len(new_routes)} routes")
                    
                    # Remove old routes
                    if current_routes:
                        self.play(FadeOut(current_routes), run_time=0.5)
                    
                    # FIXED: Create visible route lines
                    route_lines = VGroup()
                    routes_added = 0
                    max_routes_to_show = 10  # Limit for clarity
                    
                    for route in new_routes[:max_routes_to_show]:
                        if len(route) < 2:
                            continue
                        
                        try:
                            route_coords = [self.G.vs[vertex]['pos'] for vertex in route]
                            manim_coords = [self.transform_coords(x, y) for x, y in route_coords]
                            
                            if len(manim_coords) >= 2:
                                # Create route as connected line segments
                                route_line = VMobject()
                                route_line.set_points_as_corners(manim_coords)
                                # Make routes very visible
                                route_line.set_stroke(self.ROUTE_COLOR, width=6, opacity=1.0)
                                route_lines.add(route_line)
                                routes_added += 1
                        except Exception as e:
                            print(f"    Error creating route visualization: {e}")
                            continue
                    
                    print(f"  Created {routes_added} route visualizations")
                    
                    # Animate new routes
                    if route_lines:
                        self.play(Create(route_lines), run_time=2)
                        current_routes = route_lines
                        print(f"  Routes displayed successfully")
                    else:
                        print(f"  Warning: No routes could be visualized")
                    
                    # Progress text
                    progress_text = Text(f"TSS {iteration + 1}/{len(self.optimal_tss_gdf)} placed", 
                                       font_size=32, color=YELLOW)
                    progress_text.to_edge(RIGHT + UP).shift(DOWN * 1)
                    
                    if iteration > 0:
                        self.play(Transform(self.progress_text, progress_text))
                    else:
                        self.play(Write(progress_text))
                        self.progress_text = progress_text
                    
                    self.wait(2)  # Longer pause to see routes
                
                # Clean up flash elements
                self.remove(flash_circle)
        
        self.tss_markers = tss_markers
        self.current_routes = current_routes
    
    def show_final_state(self):
        """Show the final optimized state"""
        # Update title
        final_title = Text("Final Optimized Network", font_size=40)
        final_title.to_edge(UP)
        self.play(Transform(self.current_title, final_title))
        
        # Fade out unused pool points
        unused_pool_dots = VGroup()
        for dot in self.pool_dots:
            if dot.get_color() != ManimColor(YELLOW):
                unused_pool_dots.add(dot)
        
        self.play(
            FadeOut(unused_pool_dots, shift=DOWN * 0.5),
            FadeOut(self.progress_text),
            run_time=2
        )
        
        # Highlight the final solution
        self.play(
            self.tss_markers.animate.scale(1.5).set_color(GOLD),
            run_time=2
        )
        
        # Summary statistics
        summary_text = VGroup(
            Text(f"Selected {len(self.optimal_tss_gdf)} optimal TSS locations", font_size=28),
            Text("Routes optimized to minimize transport effort", font_size=28),
            Text("Algorithm: Greedy heuristic with Dijkstra routing", font_size=28)
        )
        
        summary_text.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary_text.to_edge(LEFT + DOWN).shift(UP * 1)
        
        self.play(Write(summary_text), run_time=3)
        
        # Final pause to see routes
        self.wait(4)
        
        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=2
        )

if __name__ == "__main__":
    print("FIXED VERSION v3 - Roads and Routes now use the full dataset for connectivity.")
    print("\nTo render this animation, run:")
    print("manim -pql animation.py LogisticsOptimizationAnimation")
    print("\nKey fixes:")
    print("- Removed .sample() from road network visualization to show a connected map.")
    print("- Removed .sample() from graph building to enable successful pathfinding.")
    print("- Routes will now correctly follow the visualized road network.")