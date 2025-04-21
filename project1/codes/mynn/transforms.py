import numpy as np
from scipy.ndimage import zoom

np.random.seed(309)

class ImageTransformer:
    def __init__(self, Images):
        # Assuming Images is a numpy array of shape (num_samples, 784)
        self.images = Images.reshape(-1, 28, 28)

    def normalize(self, mean=0.1307, std=0.3081):
        self.images = (self.images - mean) / std
        return self.images

    def translate(self, tx, ty):
        height, width = self.images.shape[1:]
        translated = np.zeros_like(self.images)

        x_start = max(0, tx)
        y_start = max(0, ty)
        x_end = min(width, width + tx)
        y_end = min(height, height + ty)

        translated[:, y_start:y_end, x_start:x_end] = \
            self.images[:, max(0, -ty):height - max(0, ty), max(0, -tx):width - max(0, tx)]

        return translated

    def rotate(self, angle):
        angle_rad = np.deg2rad(angle)
        height, width = self.images.shape[1:]
        center = np.array([height // 2, width // 2])

        rotated_images = np.zeros_like(self.images)

        for idx, img in enumerate(self.images):
            coords = np.indices((height, width)).reshape(2, -1) - center[:, None]
            rot_coords = np.dot(np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ]), coords) + center[:, None]
            rot_coords = np.round(rot_coords).astype(int)

            valid = (
                (rot_coords[0] >= 0) & (rot_coords[0] < height) &
                (rot_coords[1] >= 0) & (rot_coords[1] < width)
            )

            rotated = np.zeros((height, width))
            rotated[coords[0, valid] + center[0], coords[1, valid] + center[1]] = \
                img[rot_coords[0, valid], rot_coords[1, valid]]

            rotated_images[idx] = rotated

        return rotated_images

    def random_resize(self, scale_range=(0.9, 1.1)):
        resized = np.zeros_like(self.images)
        for i in range(len(self.images)):
            scale = np.random.uniform(*scale_range)
            resized_img = zoom(self.images[i], zoom=scale, order=1)
            h, w = resized_img.shape
            padded = np.zeros((28, 28))
            h_start = max((28 - h) // 2, 0)
            w_start = max((28 - w) // 2, 0)
            padded[h_start:h_start + min(h, 28), w_start:w_start + min(w, 28)] = \
                resized_img[:min(h, 28), :min(w, 28)]
            resized[i] = padded
        return resized

    def pipeline(self, transforms, aug_parallel=False):
        original = self.images.copy()

        if aug_parallel:
            augmented = [t() for t in transforms]
            augmented.insert(0, original)
            all_images = np.concatenate(augmented, axis=0)
        else:
            out = original.copy()
            for transform in transforms:
                out = transform()
            all_images = np.concatenate((original, out), axis=0)

        np.random.shuffle(all_images)
        return all_images