import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Print out its shape
print(image.shape)
plt.imshow(image)
plt.show()