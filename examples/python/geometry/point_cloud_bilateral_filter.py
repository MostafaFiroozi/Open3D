# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np


def add_gaussian_noise(pcd, sigma=0.005):
    """Add Gaussian noise to point cloud positions."""
    noisy = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(noisy.points)
    pts += np.random.normal(0.0, sigma, pts.shape)
    noisy.points = o3d.utility.Vector3dVector(pts)
    return noisy


if __name__ == "__main__":
    # Load Stanford Bunny (built into Open3D, no download needed)
    bunny_data = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_data.path)
    mesh.compute_vertex_normals()

    # Sample a point cloud from the mesh surface
    print("Sampling point cloud from mesh...")
    pcd_clean = mesh.sample_points_uniformly(number_of_points=50000)
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_clean.orient_normals_consistent_tangent_plane(100)

    # Add synthetic Gaussian noise
    print("Adding Gaussian noise (sigma=0.005)...")
    np.random.seed(42)
    pcd_noisy = add_gaussian_noise(pcd_clean, sigma=0.005)
    pcd_noisy.normals = pcd_clean.normals  # keep original normals for filtering

    # Apply bilateral filter
    # sigma_s: spatial radius ~3x noise sigma; sigma_n: range ~1x noise sigma
    print("Applying bilateral filter...")
    pcd_filtered = pcd_noisy.bilateral_filter(
        num_iterations=3, sigma_s=0.015, sigma_n=0.005)

    # Compute RMSE vs clean cloud
    dists_before = np.asarray(pcd_noisy.compute_point_cloud_distance(pcd_clean))
    dists_after = np.asarray(pcd_filtered.compute_point_cloud_distance(pcd_clean))
    print(f"RMSE before filtering: {np.sqrt(np.mean(dists_before**2)):.6f}")
    print(f"RMSE after  filtering: {np.sqrt(np.mean(dists_after**2)):.6f}")

    # Visualize side by side
    pcd_noisy.paint_uniform_color([1.0, 0.5, 0.0])    # orange = noisy
    pcd_filtered.paint_uniform_color([0.2, 0.7, 0.2]) # green  = filtered

    pcd_noisy.translate([-0.05, 0, 0])
    pcd_filtered.translate([0.05, 0, 0])

    print("Showing: orange = noisy input, green = bilateral filtered")
    o3d.visualization.draw_geometries([pcd_noisy, pcd_filtered],
                                      window_name="Bilateral Filter",
                                      width=1280, height=720)
