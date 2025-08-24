"""
Manim Animation for Logistics Optimization Algorithm Visualization
This script creates an animated visualization of the greedy TSS location optimization process.
Updated to use igraph instead of NetworkX for better performance.
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
        self.SCALE_FACTOR = 0.00001  # Scale down coordinates to fit Manim canvas
        self.MAX_SNAP_DISTANCE = 500  # meters
        
        # Colors
        self.ROAD_COLOR = GRAY
        self.POOL_COLOR = BLUE
        self.ORIGIN_COLOR = RED
        self.CCH_COLOR = GREEN
        self.TSS_COLOR = YELLOW
        self.ROUTE_COLOR = ORANGE
        
        # Sizes
        self.POOL_SIZE = 0.02
        self.ORIGIN_SIZE = 0.04
        self.CCH_SIZE = 0.06
        self.TSS_SIZE = 0.08
        
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
            all_bounds = []
            for gdf in [self.roads_gdf, self.grid_gdf, self.origins_gdf, self.cch_gdf]:
                bounds = gdf.total_bounds
                all_bounds.extend([bounds[0], bounds[1], bounds[2], bounds[3]])
            
            self.min_x, self.min_y = min(all_bounds[::2]), min(all_bounds[1::2])
            self.max_x, self.max_y = max(all_bounds[::2]), max(all_bounds[1::2])
            
            # Center coordinates for transformation
            self.center_x = (self.min_x + self.max_x) / 2
            self.center_y = (self.min_y + self.max_y) / 2
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def transform_coords(self, x, y):
        """Transform real-world coordinates to Manim canvas coordinates"""
        # Center and scale the coordinates
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
        
        # Process roads to create network
        roads_processed = 0
        for idx, road in self.roads_gdf.iterrows():
            try:
                # Handle different geometry types
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
                    
                    # Skip if not enough coordinates
                    if len(coords) < 2:
                        continue
                    
                    # Add nodes and edges for each road segment
                    for i in range(len(coords) - 1):
                        coord1, coord2 = coords[i], coords[i + 1]
                        
                        # Skip if coordinates are identical
                        if coord1 == coord2:
                            continue
                        
                        # Round coordinates to avoid floating point precision issues
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
                        
                        if vertex1_id != vertex2_id:  # Avoid self-loops
                            length = Point(coord1).distance(Point(coord2))
                            if length > 0:  # Only add edges with positive length
                                edges.append((vertex1_id, vertex2_id, length))
                
                roads_processed += 1
                if roads_processed % 5000 == 0:
                    print(f"  Processed {roads_processed}/{len(self.roads_gdf)} roads...")
                    
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
        
        for idx, origin in self.origins_gdf.iterrows():
            origin_vertex, origin_pos = self.snap_to_network(origin.geometry)
            if origin_vertex is None:
                continue
            
            # Find shortest path to CCH directly
            min_direct_dist = float('inf')
            best_direct_path = None
            
            for _, cch in self.cch_gdf.iterrows():
                cch_vertex, cch_pos = self.snap_to_network(cch.geometry)
                if cch_vertex is None:
                    continue
                
                try:
                    # Use igraph's shortest path function
                    path_vertices = self.G.get_shortest_paths(
                        origin_vertex, 
                        cch_vertex, 
                        weights='weight',
                        output='vpath'
                    )[0]
                    
                    if len(path_vertices) > 1:  # Valid path found
                        # Calculate path distance
                        dist = self.G.shortest_paths(
                            origin_vertex, 
                            cch_vertex, 
                            weights='weight'
                        )[0][0]
                        
                        if dist < min_direct_dist:
                            min_direct_dist = dist
                            best_direct_path = path_vertices
                except:
                    continue
            
            # Find shortest path via TSS
            min_tss_dist = float('inf')
            best_tss_path = None
            
            if active_tss_vertices:
                for tss_vertex in active_tss_vertices:
                    try:
                        # Path from origin to TSS
                        path1_vertices = self.G.get_shortest_paths(
                            origin_vertex, 
                            tss_vertex, 
                            weights='weight',
                            output='vpath'
                        )[0]
                        
                        if len(path1_vertices) <= 1:  # No valid path
                            continue
                            
                        dist1 = self.G.shortest_paths(
                            origin_vertex, 
                            tss_vertex, 
                            weights='weight'
                        )[0][0]
                        
                        # Path from TSS to nearest CCH
                        min_tss_to_cch_dist = float('inf')
                        best_tss_to_cch_path = None
                        
                        for _, cch in self.cch_gdf.iterrows():
                            cch_vertex, cch_pos = self.snap_to_network(cch.geometry)
                            if cch_vertex is None:
                                continue
                            
                            try:
                                path2_vertices = self.G.get_shortest_paths(
                                    tss_vertex, 
                                    cch_vertex, 
                                    weights='weight',
                                    output='vpath'
                                )[0]
                                
                                if len(path2_vertices) > 1:  # Valid path found
                                    dist2 = self.G.shortest_paths(
                                        tss_vertex, 
                                        cch_vertex, 
                                        weights='weight'
                                    )[0][0]
                                    
                                    if dist2 < min_tss_to_cch_dist:
                                        min_tss_to_cch_dist = dist2
                                        best_tss_to_cch_path = path2_vertices
                            except:
                                continue
                        
                        if best_tss_to_cch_path:
                            total_dist = dist1 + min_tss_to_cch_dist
                            if total_dist < min_tss_dist:
                                min_tss_dist = total_dist
                                # Combine paths (remove duplicate TSS vertex)
                                best_tss_path = path1_vertices + best_tss_to_cch_path[1:]
                    except:
                        continue
            
            # Choose better route
            if best_direct_path and (min_direct_dist <= min_tss_dist or best_tss_path is None):
                routes.append(best_direct_path)
            elif best_tss_path:
                routes.append(best_tss_path)
        
        return routes
    
    def show_title(self):
        """Show the animation title"""
        title = Text("Logistics Optimization: TSS Location Selection", font_size=36)
        subtitle = Text("Minimizing Transport Effort Through Strategic Facility Placement", font_size=24)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
    
    def setup_base_map(self):
        """Draw the road network"""
        road_lines = VGroup()
        
        print("Creating road network visualization...")
        roads_visualized = 0
        
        # Sample roads for visualization to avoid performance issues
        max_roads_to_show = 2000  # Limit for performance
        road_sample = self.roads_gdf.sample(min(max_roads_to_show, len(self.roads_gdf)))
        
        for idx, road in road_sample.iterrows():
            try:
                # Handle different geometry types
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
                    
                    # Create line
                    if len(manim_coords) >= 2:
                        line = VMobject()
                        line.set_points_as_corners(manim_coords)
                        line.set_stroke(self.ROAD_COLOR, width=0.5, opacity=0.4)
                        road_lines.add(line)
                        roads_visualized += 1
                        
            except Exception as e:
                print(f"  Error visualizing road {idx}: {e}")
                continue
        
        print(f"  Created visualization for {roads_visualized} roads")
        
        # Show roads
        title = Text("Step 1: Road Network", font_size=32)
        title.to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(road_lines), run_time=3)
        self.wait(1)
        
        self.road_lines = road_lines
        self.current_title = title
    
    def show_pool_and_facilities(self):
        """Show pool of candidate sites, origins, and destinations"""
        # Update title
        new_title = Text("Step 2: Facilities and Candidate Pool", font_size=32)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        # Create pool points
        pool_dots = VGroup()
        for idx, point in self.grid_gdf.iterrows():
            pos = self.transform_coords(point.geometry.x, point.geometry.y)
            dot = Dot(pos, radius=self.POOL_SIZE, color=self.POOL_COLOR, fill_opacity=0.4)
            pool_dots.add(dot)
        
        # Create origin points
        origin_dots = VGroup()
        for idx, origin in self.origins_gdf.iterrows():
            pos = self.transform_coords(origin.geometry.x, origin.geometry.y)
            dot = Dot(pos, radius=self.ORIGIN_SIZE, color=self.ORIGIN_COLOR)
            origin_dots.add(dot)
        
        # Create CCH points
        cch_dots = VGroup()
        for idx, cch in self.cch_gdf.iterrows():
            pos = self.transform_coords(cch.geometry.x, cch.geometry.y)
            dot = Dot(pos, radius=self.CCH_SIZE, color=self.CCH_COLOR)
            cch_dots.add(dot)
        
        # Show elements with labels
        pool_label = Text("Candidate Pool", font_size=20, color=self.POOL_COLOR)
        origin_label = Text("Material Origins", font_size=20, color=self.ORIGIN_COLOR)
        cch_label = Text("Destinations (CCH)", font_size=20, color=self.CCH_COLOR)
        
        # Position labels
        pool_label.to_edge(LEFT + UP).shift(DOWN * 0.5)
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
    
    def animate_selection_process(self):
        """Animate the TSS selection process"""
        # Update title
        new_title = Text("Step 3: Greedy Selection Process", font_size=32)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        active_tss_vertices = []
        tss_markers = VGroup()
        current_routes = VGroup()
        
        for iteration, (idx, tss_location) in enumerate(self.optimal_tss_gdf.iterrows()):
            print(f"Animating TSS selection {iteration + 1}/{len(self.optimal_tss_gdf)}")
            
            # Find nearest pool point to the selected TSS location
            tss_point = tss_location.geometry
            distances = [tss_point.distance(pool_point.geometry) for _, pool_point in self.grid_gdf.iterrows()]
            nearest_pool_idx = np.argmin(distances)
            
            # Highlight the selected pool point
            selected_pool_dot = self.pool_dots[nearest_pool_idx]
            
            # Flash animation
            flash_circle = Circle(radius=self.POOL_SIZE * 3, color=YELLOW)
            flash_circle.move_to(selected_pool_dot.get_center())
            
            self.play(
                selected_pool_dot.animate.set_color(YELLOW).scale(2),
                Create(flash_circle),
                run_time=1
            )
            
            self.play(
                flash_circle.animate.scale(2).set_opacity(0),
                run_time=0.5
            )
            
            # Snap to network and create TSS marker
            tss_vertex, network_pos = self.snap_to_network(tss_point)
            if tss_vertex is not None:
                active_tss_vertices.append(tss_vertex)
                
                # Create connection line from pool to network
                pool_pos = selected_pool_dot.get_center()
                network_manim_pos = self.transform_coords(network_pos[0], network_pos[1])
                
                connection_line = Line(pool_pos, network_manim_pos, color=YELLOW, stroke_width=3)
                
                # Create TSS marker
                tss_marker = Star(n=6, outer_radius=self.TSS_SIZE, color=self.TSS_COLOR, fill_opacity=0.8)
                tss_marker.move_to(network_manim_pos)
                
                self.play(
                    Create(connection_line),
                    run_time=1
                )
                
                self.play(
                    Create(tss_marker),
                    FadeOut(connection_line),
                    run_time=1
                )
                
                tss_markers.add(tss_marker)
                
                # Calculate and animate new routes
                new_routes = self.calculate_routes(active_tss_vertices)
                
                # Remove old routes
                if current_routes:
                    self.play(FadeOut(current_routes), run_time=0.5)
                
                # Create new route lines
                route_lines = VGroup()
                for route in new_routes:
                    if len(route) < 2:
                        continue
                    
                    route_coords = [self.G.vs[vertex]['pos'] for vertex in route]
                    manim_coords = [self.transform_coords(x, y) for x, y in route_coords]
                    
                    if len(manim_coords) >= 2:
                        route_line = VMobject()
                        route_line.set_points_as_corners(manim_coords)
                        route_line.set_stroke(self.ROUTE_COLOR, width=2, opacity=0.8)
                        route_lines.add(route_line)
                
                # Animate new routes
                if route_lines:
                    self.play(Create(route_lines), run_time=2)
                    current_routes = route_lines
                
                # Show progress
                progress_text = Text(f"TSS {iteration + 1}/{len(self.optimal_tss_gdf)} placed", 
                                   font_size=24, color=YELLOW)
                progress_text.to_edge(RIGHT + UP).shift(DOWN * 0.5)
                
                if iteration > 0:
                    self.play(Transform(self.progress_text, progress_text))
                else:
                    self.play(Write(progress_text))
                    self.progress_text = progress_text
                
                self.wait(1)
            
            # Clean up flash elements
            self.remove(flash_circle)
        
        self.tss_markers = tss_markers
        self.current_routes = current_routes
    
    def show_final_state(self):
        """Show the final optimized state"""
        # Update title
        final_title = Text("Final Optimized Network", font_size=32)
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
            self.tss_markers.animate.scale(1.2).set_color(GOLD),
            run_time=1
        )
        
        # Summary statistics
        summary_text = VGroup(
            Text(f"Selected {len(self.optimal_tss_gdf)} optimal TSS locations", font_size=20),
            Text("Routes optimized to minimize transport effort", font_size=20),
            Text("Algorithm: Greedy heuristic with Dijkstra routing", font_size=20)
        )
        
        summary_text.arrange(DOWN, aligned_edge=LEFT)
        summary_text.to_edge(LEFT + DOWN).shift(UP * 0.5)
        
        self.play(Write(summary_text), run_time=3)
        
        # Final pause
        self.wait(3)
        
        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=2
        )

# Additional utility for running the animation
if __name__ == "__main__":
    # Instructions for running the animation
    print("To render this animation, run:")
    print("manim -pql logistics_animation.py LogisticsOptimizationAnimation")
    print("\nMake sure you have the following input files:")
    print("- input/road.gpkg")
    print("- input/grid.gpkg") 
    print("- input/project.gpkg")
    print("- input/cch.gpkg")
    print("- output/optimal_tss_locations.gpkg")
    print("\nRequired packages:")
    print("- pip install python-igraph")