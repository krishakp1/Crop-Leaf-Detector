import os

folder = "PlantVillage"
classes = sorted(os.listdir(folder))
print("Actual class folders:")
for c in classes:
    print(f"- '{c}'")
print("Total:", len(classes))
