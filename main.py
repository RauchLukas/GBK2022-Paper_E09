# author: Lukas Rauch
#date: 20.05.2022

from examples.example_mesh_quality import mesh_quality
from examples.example_reconstruction import reconstruction
from examples.example_volume_pyvista import volume_pyvista
from examples.example_volume_validation import volume_validation

author = "Lukas Rauch"


if __name__ == "__main__":

    print("\nContribution to the 5th. GRAZER BETONKOLLOQUIUM 2022:")
    print("https://www.betonkolloquium.at/\n")
    print("Author: ", author)

    MESH_QUALITY = True
    RECONSTRUCTION = False
    VOLUME_VALIDATION = False
    VOLUME_PYVISTA = False

    if MESH_QUALITY:
        mesh_quality()
    if RECONSTRUCTION:
        reconstruction()
    if VOLUME_VALIDATION:
        volume_validation()
    if VOLUME_PYVISTA:
        volume_pyvista()
        




