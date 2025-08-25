"""
Manim Animation for Logistics Optimization Algorithm Visualization
MODIFIED VERSION: Visualizes CCH location selection where materials move 
directly from origins to CCH destinations. Shows the "switch" by highlighting 
routes that are reassigned to newly selected CCH locations.
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

class LogisticsOptimizationAnimationv1(Scene):
    def construct(self):
        # Set white background
        self.camera.background_color = WHITE
        
        self.setup_config()
        self.load_data()
        self.build_network_graph()
        self.prepare_facilities()
        
        self.setup_base_map()
        self.show_pool_and_facilities()
        self.animate_selection_process()
        self.show_final_state()
        
    def setup_config(self):
        """Configure animation parameters"""
        self.TARGET_CRS = 'EPSG:28992'
        self.SCALE_FACTOR = 0.001
        self.MAX_SNAP_DISTANCE = 500
        
        self.ROAD_COLOR = GRAY
        self.POOL_COLOR = BLUE
        self.ORIGIN_COLOR = RED
        self.CCH_COLOR = GREEN
        
        self.CCH_MARKER_COLORS = [
            ORANGE, TEAL, PINK, PURPLE_A, GREEN_C,
            BLUE_C, MAROON_B, GOLD_D, GRAY_BROWN, RED_B,
            YELLOW_E, BLUE_E, GREEN_E, PURPLE_E, PINK,
            BLUE_D, GREEN_D, PURPLE_D, TEAL_D, RED_D
        ]
        self.CCH_ROUTE_COLORS = self.CCH_MARKER_COLORS
        
        self.POOL_SIZE = 0.01
        self.ORIGIN_SIZE = 0.03
        self.CCH_SIZE = 0.07
        self.SELECTED_CCH_SIZE = 0.09
        
    def load_data(self):
        """Load all geospatial data files from disk."""
        try:
            self.roads_gdf = gpd.read_file('input/road.gpkg').to_crs(self.TARGET_CRS)
            self.cch_pool_gdf = gpd.read_file('input/cch_grid.gpkg').to_crs(self.TARGET_CRS)  # Changed from grid.gpkg
            self.origins_gdf = gpd.read_file('input/project.gpkg').to_crs(self.TARGET_CRS)
            self.all_cch_gdf = gpd.read_file('input/17cch.gpkg').to_crs(self.TARGET_CRS)  # Changed from cch.gpkg
            
            # Use all CCH locations as the selection sequence (instead of optimal_tss_locations)
            self.selected_cch_gdf = self.all_cch_gdf.copy()
            
            all_geometries = pd.concat([gdf.geometry for gdf in [self.roads_gdf, self.cch_pool_gdf, self.origins_gdf, self.all_cch_gdf]])
            bounds = all_geometries.total_bounds
            self.min_x, self.min_y, self.max_x, self.max_y = bounds
            
            self.center_x = (self.min_x + self.max_x) / 2
            self.center_y = (self.min_y + self.max_y) / 2
            
            max_extent = max(self.max_x - self.min_x, self.max_y - self.min_y)
            self.SCALE_FACTOR = 10 / max_extent
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def prepare_facilities(self):
        """Pre-snaps facilities and pre-calculates origin nodes."""
        # Snap all origins once to improve performance
        self.origin_nodes = [self.snap_to_network(origin.geometry)[0] for _, origin in self.origins_gdf.iterrows()]

    def transform_coords(self, x, y):
        return np.array([(x - self.center_x) * self.SCALE_FACTOR, (y - self.center_y) * self.SCALE_FACTOR, 0])
    
    def build_network_graph(self):
        print("Building network graph for routing...")
        vertices, edges, vertex_coords, vertex_id = {}, [], [], 0
        
        for _, road in self.roads_gdf.iterrows():
            if road.geometry is None or road.geometry.is_empty: continue
            geometries = [road.geometry] if road.geometry.geom_type == 'LineString' else list(road.geometry.geoms)
            for geom in geometries:
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    c1, c2 = coords[i], coords[i+1]
                    if c1 == c2: continue
                    for coord in [c1, c2]:
                        if coord not in vertices:
                            vertices[coord] = vertex_id
                            vertex_coords.append(coord)
                            vertex_id += 1
                    v1, v2 = vertices[c1], vertices[c2]
                    if v1 != v2:
                        length = Point(c1).distance(Point(c2))
                        if length > 0:
                            edges.append((v1, v2, length))

        if vertex_id > 0 and edges:
            self.G = ig.Graph(n=vertex_id, directed=False)
            self.G.add_edges([(e[0], e[1]) for e in edges])
            self.G.es['weight'] = [e[2] for e in edges]
            self.G.vs['pos'] = vertex_coords
            self.node_coords = np.array(vertex_coords)
            self.node_tree = cKDTree(self.node_coords)
        else:
            raise RuntimeError("Failed to build a valid graph.")
        print(f"Built igraph with {self.G.vcount()} vertices and {self.G.ecount()} edges")

    def snap_to_network(self, point):
        if not hasattr(self, 'node_tree') or self.node_tree is None: 
            raise AttributeError("node_tree not initialized.")
        dist, idx = self.node_tree.query([point.x, point.y])
        return (int(idx), self.G.vs[int(idx)]['pos']) if dist <= self.MAX_SNAP_DISTANCE else (None, None)

    def calculate_routes(self, active_cch_vertices):
        """
        Calculate direct routes from origins to selected CCH locations.
        Returns a list of tuples: [(path, cch_index, origin_index), ...]
        """
        routes_with_info = []
        if not active_cch_vertices: return []
        
        for origin_idx, origin_vertex in enumerate(self.origin_nodes):
            if origin_vertex is None: continue
            
            # Find the closest CCH among the active ones
            distances_to_cch = self.G.shortest_paths(origin_vertex, active_cch_vertices, weights='weight')[0]
            valid_cch = [(dist, i) for i, dist in enumerate(distances_to_cch) if np.isfinite(dist)]
            if not valid_cch: continue
            
            min_dist_to_cch, best_cch_index = min(valid_cch)
            best_cch_node = active_cch_vertices[best_cch_index]
            
            # Get direct path from origin to CCH
            path = self.G.get_shortest_paths(origin_vertex, best_cch_node, weights='weight', output='vpath')[0]
            if path:
                routes_with_info.append((path, best_cch_index, origin_idx))
                
        return routes_with_info
    
    def show_title(self):
        title = Text("Logistics Optimization: CCH Location Selection", font_size=48, color=BLACK)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))
    
    def setup_base_map(self):
        road_lines = VGroup()
        for _, road in self.roads_gdf.iterrows():
            if road.geometry:
                geometries = [road.geometry] if road.geometry.geom_type == 'LineString' else list(road.geometry.geoms)
                for geom in geometries:
                    if len(geom.coords) >= 2:
                        line = VMobject().set_points_as_corners([self.transform_coords(x,y) for x,y in geom.coords])
                        line.set_stroke(self.ROAD_COLOR, 1, 0.7)
                        road_lines.add(line)
        
        self.play(Create(road_lines), run_time=3)
        self.wait(1)
        self.road_lines = road_lines
    
    def show_pool_and_facilities(self):
        
        # Show CCH candidate pool
        pool_dots = VGroup()
        for _, p in self.cch_pool_gdf.iterrows():
            dot = Dot(self.transform_coords(p.geometry.x, p.geometry.y), radius=self.POOL_SIZE, color=self.POOL_COLOR)
            pool_dots.add(dot)
            
        # Show origins (project sites)
        origin_dots = VGroup()
        for _, o in self.origins_gdf.iterrows():
            dot = Dot(self.transform_coords(o.geometry.x, o.geometry.y), radius=self.ORIGIN_SIZE, color=self.ORIGIN_COLOR)
            origin_dots.add(dot)

        labels = VGroup(
            Text("CCH Candidate Pool", font_size=28, color=self.POOL_COLOR),
            Text("Origins (Project Sites)", font_size=28, color=self.ORIGIN_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT + DOWN).shift(UP * 1)
        
        self.play(
            LaggedStart(
                AnimationGroup(Create(pool_dots), Write(labels[0])),
                AnimationGroup(Create(origin_dots), Write(labels[1])),
                lag_ratio=0.5
            ), 
            run_time=3
        )
        self.wait(1)
        self.pool_dots, self.labels = pool_dots, labels
    
    def animate_selection_process(self):
        
        active_cch_vertices, selected_cch_markers, current_routes = [], VGroup(), VGroup()
        # Store the assignment of each origin to CCH index
        origin_assignments = {} # origin_idx -> cch_idx

        for iteration, (_, cch_location) in enumerate(self.selected_cch_gdf.iterrows()):
            print(f"Animating CCH selection {iteration + 1}/{len(self.selected_cch_gdf)}")
            cch_vertex, network_pos = self.snap_to_network(cch_location.geometry)
            
            if cch_vertex is not None:
                active_cch_vertices.append(cch_vertex)
                marker_color = self.CCH_MARKER_COLORS[iteration % len(self.CCH_MARKER_COLORS)]
                network_manim_pos = self.transform_coords(network_pos[0], network_pos[1])
                
                # Create a star marker for the selected CCH
                cch_marker = Star(n=6, outer_radius=self.SELECTED_CCH_SIZE, color=marker_color, fill_opacity=1.0, stroke_width=2, stroke_color=BLACK).move_to(network_manim_pos)
                
                # Flash effect when selecting CCH
                self.play(
                    ShowPassingFlash(
                        Circle(radius=self.SELECTED_CCH_SIZE * 2, color=marker_color, stroke_width=6).move_to(network_manim_pos), 
                        time_width=0.5
                    ), 
                    Create(cch_marker)
                )
                selected_cch_markers.add(cch_marker)
                
                new_routes_info = self.calculate_routes(active_cch_vertices)
                print(f"  Found {len(new_routes_info)} routes")
                self.play(FadeOut(current_routes), run_time=0.5)
                
                stable_routes_vgroup = VGroup()
                switched_routes_vgroup = VGroup() # Routes that switched to the NEWEST CCH
                
                # Store previous assignments before calculating new ones
                previous_assignments = origin_assignments.copy()
                new_assignments = {}
                newest_cch_index = len(active_cch_vertices) - 1  # Index of the just-added CCH
                
                for route_path, cch_index, origin_idx in new_routes_info:
                    if not route_path: continue
                    new_assignments[origin_idx] = cch_index
                    
                    route_color = self.CCH_ROUTE_COLORS[cch_index % len(self.CCH_ROUTE_COLORS)]
                    manim_coords = [self.transform_coords(x, y) for x, y in [self.G.vs[v]['pos'] for v in route_path]]
                    line = VMobject().set_points_as_corners(manim_coords).set_stroke(color=route_color, width=2.5, opacity=0.6)

                    # Check if the origin switched its assignment to the NEWEST CCH
                    previous_cch = previous_assignments.get(origin_idx)
                    
                    # An origin "switched" if:
                    # 1. It's the first iteration (all routes are new), OR
                    # 2. It had a different assignment before AND now uses the newest CCH
                    is_switch = (iteration == 0) or (previous_cch != cch_index and cch_index == newest_cch_index)
                    
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
                                Flash(cch_marker, color=marker_color, flash_radius=0.3, num_lines=12)
                            ),
                            Wait(0.5),
                            switched_routes_vgroup.animate.set_stroke(opacity=0.7, width=2.5),
                        ),
                        run_time=3.0
                    )
                else:
                    self.wait(1.0)  # If no switches, just wait a bit

                current_routes = VGroup(stable_routes_vgroup, switched_routes_vgroup)
                
                progress_text = Text(f"CCH {iteration + 1}/{len(self.selected_cch_gdf)} selected", font_size=32, color=BLACK).to_edge(RIGHT + UP).shift(DOWN * 1)
                if iteration == 0: 
                    self.play(Write(progress_text))
                    self.progress_text = progress_text
                else: 
                    self.play(Transform(self.progress_text, progress_text))
                
                self.wait(1.5)
        
        self.selected_cch_markers = selected_cch_markers
    
    def show_final_state(self):
        self.play(
            FadeOut(self.pool_dots), 
            FadeOut(self.labels), 
            FadeOut(self.progress_text), 
            run_time=1.5
        )
        self.play(self.selected_cch_markers.animate.scale(1.5))
        
        self.wait(4)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

if __name__ == "__main__":
    print("\nTo render this animation, run:")
    print("manim -pql animationv1.py LogisticsOptimizationAnimationv1 --disable_caching")