#!/usr/bin/env python3
"""
Create a funny PipeDreams icon - Mario sleeping on a pipe
"""

from PIL import Image, ImageDraw, ImageFont

# Create 512x512 icon
img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw green pipe
pipe_color = (60, 200, 60)
# Main pipe body
draw.rectangle([150, 300, 350, 500], fill=pipe_color, outline=(40, 150, 40), width=4)
# Pipe top (cylinder)
draw.ellipse([130, 280, 370, 320], fill=(80, 220, 80), outline=(40, 150, 40), width=4)
# Inner shadow
draw.ellipse([150, 290, 350, 310], fill=(40, 150, 40))

# Draw sleeping Mario (simplified)
# Red cap
draw.ellipse([220, 200, 320, 250], fill=(255, 0, 0), outline=(180, 0, 0), width=3)
# M on cap
draw.text((250, 210), "M", fill=(255, 255, 255), font=None)

# Face (peach color)
draw.ellipse([230, 240, 310, 300], fill=(255, 220, 180), outline=(200, 150, 100), width=2)

# Mustache
draw.ellipse([235, 270, 270, 285], fill=(80, 40, 20))
draw.ellipse([270, 270, 305, 285], fill=(80, 40, 20))

# Closed eyes (sleeping)
draw.arc([245, 255, 265, 270], 0, 180, fill=(0, 0, 0), width=3)
draw.arc([275, 255, 295, 270], 0, 180, fill=(0, 0, 0), width=3)

# ZZZ (sleeping)
draw.text((320, 180), "Z", fill=(100, 100, 255), font=None)
draw.text((340, 160), "Z", fill=(100, 100, 255), font=None)
draw.text((360, 140), "Z", fill=(100, 100, 255), font=None)

# Audio waveform in background (subtle)
import math
for x in range(0, 512, 10):
    y = 100 + int(30 * math.sin(x / 30.0))
    draw.ellipse([x, y, x+4, y+4], fill=(0, 255, 136, 100))

# Save
img.save('/root/pipedreams/pipedreams_icon.png')
print("Icon created: pipedreams_icon.png")
