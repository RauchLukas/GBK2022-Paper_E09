# author: Lukas Rauch
#date: 20.05.2022
import examples.example_wall_scan
from examples.example_mesh_quality import mesh_quality
from examples.example_reconstruction import reconstruction
from examples.example_volume_pyvista import volume_pyvista
from examples.example_volume_validation import volume_validation
from examples.example_synthetic_mesh import synthetic_mesh
from examples.example_wall_scan import wall_scan
from examples.example_wedges_topography import wedges_terrain
from examples.example_volume_validation_wedge import volume_validation_wedge
from examples.example_blob import example_blob
from examples.example_blob_stats import example_volume_blob_stats

author = "Lukas Rauch"


if __name__ == "__main__":

    print("\nContribution to the 5th. GRAZER BETONKOLLOQUIUM 2022:")
    print("https://www.betonkolloquium.at/\n")
    print("Author: ", author)

    MESH_QUALITY = False
    RECONSTRUCTION = False
    VOLUME_VALIDATION = False
    VOLUME_PYVISTA = False
    SYNTHETIC_MESH = False
    WALL_SCAN = False
    WEDGES_TERRAIN = False
    VOLUME_VALIDATION_WEDGE = False
    VOLUME_VALIDATION_BLOB = False
    VOLUME_ERROR_STATS_BLOB = True

    if MESH_QUALITY:
        mesh_quality()
    if RECONSTRUCTION:
        reconstruction()
    if VOLUME_VALIDATION:
        volume_validation()
    if VOLUME_PYVISTA:
        volume_pyvista()
    if SYNTHETIC_MESH:
        synthetic_mesh()
    if WALL_SCAN:
        wall_scan()
    if WEDGES_TERRAIN:
        wedges_terrain()
    if VOLUME_VALIDATION_WEDGE:
        volume_validation_wedge()
    if VOLUME_VALIDATION_BLOB:
        example_blob()
    if VOLUME_ERROR_STATS_BLOB:
        example_volume_blob_stats()


        
    print("done.")




