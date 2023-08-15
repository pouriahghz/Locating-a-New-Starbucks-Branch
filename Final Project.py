import numpy as np
from sklearn.neighbors import NearestNeighbors

# مختصات نقاط موجود
# Coordinates of existing locations
metro_locations = np.array([(1, 2), (1.5, 11), (3.5, 5), (4, 19), (8, 1), (12, 10)])
gas_station_locations = np.array([(2, 4), (5, 15), (6, 7), (10, 8)])
starbucks_locations = np.array([(2.5, 9), (3, 3), (7, 5), (8.5, 16)])

# وزن‌ها
# Weights for different locations
weights = {
    "starbucks": 1,
    "metro": 2,
    "gas_station": 3
}

# تعداد نقاط احتمالی که می‌خواهید پیدا کنید
# Number of probable points to find
num_probable_points = 10

# نقاط احتمالی را در یک لیست ذخیره کنیم
# List to store probable points
probable_points = []

# حلقه تو در تو برای بررسی نقاط
# Nested loop to iterate over points
for x in range(0, 13):  # بازه مختصات x از 0 تا 12#    Range of x coordinates from 0 to 12
    for y in range(0, 21):  # بازه مختصات y از 0 تا 20#    Range of y coordinates from 0 to 20
        if (x, y) == (6, 7):
            continue  # نقطه (6, 7) را از محاسبات حذف کنیم#     Skip the point (6, 7)

        # محاسبه مرتبه زمانی برای هر نقطه
        # Calculate total weighted distance for each point
        total_distance = 0
        for loc in metro_locations:
            distance = np.abs(x - loc[0]) + np.abs(y - loc[1])
            total_distance += distance * weights["metro"]
        for loc in gas_station_locations:
            distance = np.abs(x - loc[0]) + np.abs(y - loc[1])
            total_distance += distance * weights["gas_station"]
        for loc in starbucks_locations:
            distance = np.abs(x - loc[0]) + np.abs(y - loc[1])
            total_distance += distance * weights["starbucks"]

        probable_points.append(((x, y), total_distance))

# نقاطی که به شعبات موجود نزدیک‌تر هستند را انتخاب کنیم
# Sort probable points based on total weighted distance
sorted_points = sorted(probable_points, key=lambda x: x[1])

# انتخاب اولین 10 نقطه که بهینه‌ترین‌ها هستند
# Select the top 10 optimal probable points
best_probable_points = [point[0] for point in sorted_points[:num_probable_points]]

# نمایش نقاط احتمالی بهینه پیدا شده
# Display the best probable points
print("Best Probable Points:")
for point in best_probable_points:
    print(point)

# تعداد همسایه‌ها در KNN
# Number of neighbors for KNN
k_neighbors = 2

# ذخیره نقاط مکان‌های مهم همسایه نزدیک
# List to store important neighbors
important_neighbors = []

# دسته‌بندی نقاط به شعبات
# Loop over best probable points to find neighbors
for point in best_probable_points:
    all_locations = np.concatenate((metro_locations, gas_station_locations, starbucks_locations))
    all_weights = np.concatenate((np.full(metro_locations.shape[0], weights["metro"]),
                                  np.full(gas_station_locations.shape[0], weights["gas_station"]),
                                  np.full(starbucks_locations.shape[0], weights["starbucks"])))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(all_locations)
    distances, indices = nbrs.kneighbors([point])

    weighted_distances = np.sum(distances * all_weights[indices[0]]) / np.sum(all_weights[indices[0]])
    important_neighbors.append((point, weighted_distances))

# مختصات ۵ نقطه بهینه بر اساس وزن‌های همسایه‌های نزدیک و فاصله منهتن
# Sort important neighbors based on weighted distances
sorted_important_neighbors = sorted(important_neighbors, key=lambda x: x[1])[:5]

# نمایش نقاط بهینه بر اساس وزن‌های همسایه‌های نزدیک و فاصله منهتن
# Display the top 5 optimal points based on neighbor weights and Manhattan distance
print("Top 5 Optimal Points based on Neighbor Weights and Manhattan Distance:")
for point in sorted_important_neighbors:
    print("Coordinates:", point[0], "- Weighted Distance:", point[1])

# مقادیر تخمین میزان تقاضا قهوه در هر نقطه بهینه
# Coffee demand estimates for each optimal point
coffee_demands = [100, 120, 90, 150, 80]

# مقادیر هزینه تقریبی احداث شعبه در هر نقطه بهینه (به میلیون تومان)
# Branch cost estimates for each optimal point (in millions of Tomans)
branch_costs = [3200000000, 2750000000, 2100000000, 1800000000, 1900000000]

# انتخاب بهترین نقطه بر اساس نسبت سود به هزینه
# Select the best point based on profit-to-cost ratio
best_point_index = np.argmax(np.array(coffee_demands) / np.array(branch_costs))

# نمایش نقطه بهترین برای احداث شعبه جدید
# Display the best point for branch location
best_point = sorted_important_neighbors[best_point_index][0]
print("Best Point for Branch Location:", best_point)
