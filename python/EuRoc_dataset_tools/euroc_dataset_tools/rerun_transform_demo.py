import rerun as rr
import numpy as np

# Hardcoded recording name
recording_id = "723f3dea3090421393a3c671ce5862d0"
recording_name = "EuRoC_V1_01_easy"

# Initialize rerun (do not spawn viewer)
rr.init(application_id=recording_name, recording_id=recording_id, spawn=True)

# Define a simple translation and identity rotation
translation = [0.0, -0.3, 0.0]
rotation_matrix = np.eye(3)

# Log a static Transform3D to a path
rr.log(
    "/world/ref_mav0",
    rr.Transform3D(
        translation=translation,
        mat3x3=rotation_matrix,
    ),
    static=True,
) 