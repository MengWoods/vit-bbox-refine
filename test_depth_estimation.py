import torch
import cv2
import numpy as np
import open3d as o3d

# -------------------------------
# Load MiDaS model
# -------------------------------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# -------------------------------
# Load KITTI image
# -------------------------------
img = cv2.imread("/home/mh/Downloads/Dataset/image/0000000000.png")  # replace with your KITTI image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Transform to tensor
input_batch = transform(img_rgb).to(device)

# -------------------------------
# Predict depth
# -------------------------------
with torch.no_grad():
    depth = midas(input_batch)
    # Resize depth to original image size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

# -------------------------------
# Normalize depth for visualization
# -------------------------------
depth_min = depth.min()
depth_max = depth.max()
depth_vis = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
cv2.imshow("Depth", depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------
# Convert depth map to 3D point cloud (using KITTI intrinsics)
# -------------------------------
h, w = depth.shape
fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.854

points = []
colors = []

for v in range(h):
    for u in range(w):
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, y, z])
        colors.append(img_rgb[v, u] / 255.0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# -------------------------------
# Visualize point cloud
# -------------------------------
o3d.visualization.draw_geometries([pcd])
