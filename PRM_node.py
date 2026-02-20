import networkx as nx
from discopygal.bindings import Polygon_2, Point_2
from discopygal.solvers.prm import PRM
from discopygal.solvers_infra.nearest_neighbors import NearestNeighbors_CGAL


class PRMnode(PRM):
    def __init__(self, polygon: Polygon_2, density_threshold: int):
        self._polygon = polygon
        self._density_threshold = density_threshold
        self._num_landmarks = 100 * self._polygon.area() * self._density_threshold
        self._density = self._calculate_density()
        self._roadmap = self._generate_roadmap()
        self.is_alive = False

    def __str__(self):
        return f"Cell({self.x}, {self.y}, {'Alive' if self.is_alive else 'Dead'})"

    def _calculate_density(self):
        # Placeholder for density calculation logic
        # This should calculate the density based on the polygon's area and some criteria
        return self._density_threshold  # Replace with actual density calculation

    def build_roadmap(self):
        """
        Build a probabilistic roadmap for the rod robot in a given scene (sample random points and connect neighbors)

        Here, you can choose which metric to use by uncommenting the relevant lines.
        :return: None - the roadmap is stored in self.roadmap
        :rtype: :class:`None`
        """
        roadmap = nx.DiGraph()
        nearest_neighbors = self.nearest_neighbors_class()

        # Add start & end points
        self.start = self.scene.robots[0].start
        self.end = self.scene.robots[0].end
        self.collision_detection = {
            robot: collision_detection.ObjectCollisionDetection(self.scene.obstacles, robot) for robot in
            self.scene.robots
        }
        roadmap.add_node(self.start)
        roadmap.add_node(self.end)

        # Add valid points
        for i in range(self.num_landmarks):
            p_rand = self.sample_free()
            roadmap.add_node(p_rand)
            if i % 100 == 0:
                self.log(f'added {i} landmarks in PRM')

        nearest_neighbors.fit(list(map(self.point2vec3, roadmap.nodes)))

        # Connect all points to their k nearest neighbors
        last_nodes = list(roadmap.nodes)
        for cnt, point in enumerate(last_nodes):
            neighbors = nearest_neighbors.k_nearest(self.point2vec3(point), self.k_nn + 1)[1:]
            for neighbor in neighbors:
                neighbor = (Point_2(neighbor[0], neighbor[1]), neighbor[2])
                if type(self.metric) is AngleSpeedMetric or type(self.metric) is TwoEndsMetric:
                    weight = self.metric.dist(point, neighbor).to_double()
                else:
                    weight = self.metric.dist(point[0], neighbor[0]).to_double()
                roadmap.add_edge(point, neighbor, weight=weight)
                roadmap.add_edge(neighbor, point, weight=weight)

            if cnt % 100 == 0:
                self.log(f'connected {cnt} landmarks to their nearest neighbors')

        return roadmap

    def _generate_roadmap(self):
        roadmap = nx.DiGraph()
        nearest_neighbors = NearestNeighbors_CGAL()

        # define entry points and exit points for the cell - we would like to allow for
        # starters for one robot to enter the cell and for another robot to exit the cell, so we can connect the
        # roadmap of this cell to the roadmap of other cells
        self.bounding_points = []
        for edge in self._polygon.edges():
            self.bounding_points.append(edge.source())
            self.bounding_points.append(edge.target())
            midpoint_x = (edge.source().x() + edge.target().x()) / 2
            midpoint_y = (edge.source().y() + edge.target().y()) / 2
            self.bounding_points.append(Point_2(midpoint_x, midpoint_y))

        for point in self.bounding_points:


        # Placeholder for roadmap generation logic
        # This should generate a roadmap based on the polygon's geometry
        return []  # Replace with actual roadmap generation
