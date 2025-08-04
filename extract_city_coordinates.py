import bpy, json, os


point_names = {
    "Bottom-Left-Point":   "Bottom-Left",
    "Bottom-Right-Point":  "Bottom-Right",
    "Top-Left-Point":      "Top-Left",
    "Top-Right-Point":     "Top-Right",
    "Center-Point":        "Center"
}

coords = {}
for full, short in point_names.items():
    obj = bpy.data.objects.get(full)
    if obj:
        loc = obj.matrix_world.to_translation()
        coords[short] = [round(loc.x,2), round(loc.y,2), round(loc.z,2)]
    else:
        coords[short] = None  

json_str = json.dumps(coords, indent=4)

out_path = os.path.join(bpy.path.abspath("//"), "five_points.json")
with open(out_path, "w") as f:
    f.write(json_str)

text_name = "five_points.json"
if text_name in bpy.data.texts:
    txt = bpy.data.texts[text_name]
    txt.clear()
else:
    txt = bpy.data.texts.new(text_name)
txt.write(json_str)


print(f"✅ Saved JSON to file: {out_path} and also to internal Text: {text_name}")
