from typing import List

from CGALPY.CGALPY.Pol2 import Polygon_2
from discopygal.solvers_infra import Scene
from discopygal.solvers_infra.Solver import Solver


class cPRM(Solver):
    """
    A placeholder for a cell-based PRM solver. Not implemented yet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def load_scene(self, scene: Scene):
        super().load_scene(scene=scene)
        # We want to compartmentalize the space into cells, so we need to calculate the bounding box of the scene and divide it into cells.
        # Then, for each cell, we will build a PRM roadmap and check if the
        self._split_scene_into_cells()
        self._generate_cell_graph()


    def _split_scene_into_cells(self):
        self.cells: List[Polygon_2] = []


    def _generate_cell_graph(self):
        edges_to_polygon = {}
        for cell in self.cells:
            for edge in cell.edges:
                if edge in edges_to_polygon:
                    edges_to_polygon[edge].append(cell)
                else:
                    edges_to_polygon[edge] = [cell]
