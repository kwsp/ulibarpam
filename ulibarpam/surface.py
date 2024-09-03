import cv2
import numpy as np
from scipy import signal


# %%
def fix_surface_idx_missing(idx):
    """
    Fix cases of missing surface (group of 0s) or incorrect deeper surface
    (disjoint lines)
    """
    ### First clear disjoint surfaces
    MAX_DISTANCE = 30

    def _prev_disjoint(i_prev, i_curr, max_distance):
        return abs(idx[i_curr] - idx[i_prev]) > max_distance

    # Assume idx[0] is correct...
    i = 1
    last_good_i = i - 1
    while i < len(idx):
        # Skip over zeros
        while i < len(idx) and idx[i] == 0:
            i += 1

        # Follow next disjoint segment
        # print(f"{i=} {last_good_i=}")
        if i < len(idx):
            if _prev_disjoint(last_good_i, i, MAX_DISTANCE + i - last_good_i):
                disjoint_start = i
                if disjoint_start == 600:
                    pass
                i += 1
                while i < len(idx) and not _prev_disjoint(
                    i - 1, i, MAX_DISTANCE + i - last_good_i
                ):
                    i += 1
                disjoint_end = i
                # print("Disjoint ", (disjoint_start, disjoint_end))
                idx[disjoint_start:disjoint_end] = 0
            else:
                i += 1
                last_good_i = i - 1

    def _interp(v1: float, v2: float, n: int):
        """
        Linearly interpolate n points inside the range [v1, v2],
        where n is the number of points in between.
        For example, _interp(1., 2., 3) == [1.25, 1.5, 1.75]
        """

        # x = np.arange(1, n + 1)
        # xp = [0, n + 1]
        # fp = [v1, v2]
        # res = np.interp(x, xp, fp)

        res = np.linspace(v1, v2, n + 2)
        res = res[1:-1]
        return res

    def _interp_next_gap_inplace(idx, i):
        """"""
        while i < len(idx):
            if idx[i] == 0:
                # Look ahead until we find first nonzero
                i_start = i
                while i < len(idx) and idx[i] == 0:
                    i += 1
                i_end = i

                # 2
                if i_start > 0 and i_end < idx.size:
                    n = i_end - i_start
                    v1 = idx[i_start - 1]
                    v2 = idx[i_end]

                    res = _interp(v1, v2, n)
                    idx[i_start:i_end] = res
            else:
                i += 1
        return idx, i

    # Cases
    # 1. i_start == 0 (first zero)
    #       Move on and wait til the end. Then wrap i_start to the end
    # 2. i_start != 0, i_end != end (zeros in middle)
    #       Interpolate
    # 3. i_start != 0, i_end = end (last zero)
    #       Interpolate to start

    # Case 1. Ignore for now and do Case 2 first
    i = 0
    while i < len(idx) and idx[i] == 0:
        i += 1

    # Case 2
    while i < len(idx):
        idx, i = _interp_next_gap_inplace(idx, i)

    # Do Case 1 and Case 3
    if idx[0] == 0 or idx[-1] == 0:
        # Find last nonzero
        i = len(idx) - 1
        while idx[i] == 0 and i >= 0:
            i -= 1
        if i < 0:
            # All indices < 0. No surface found.
            # TODO. should handle this error explicitly
            return idx

        n_pts_rotate = len(idx) - i
        idx = np.roll(idx, n_pts_rotate)
        _interp_next_gap_inplace(idx, 1)
        idx = np.roll(idx, -n_pts_rotate)

    return idx


def test_fix_surface_idx_zeros():
    # Test Case 2
    idx = np.array([1.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [1.0, 1.25, 1.5, 1.75, 2.0])

    # Test Case 1
    idx = np.array([0.0, 3.0, 2.0, 1.0, 2.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [2.5, 3.0, 2.0, 1.0, 2.0])

    # Test Case 3
    idx = np.array([2.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [2.0, 3.0, 2.0, 1.0, 1.5])

    # Test complex
    idx = np.array([0.0, 4.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [3.0, 4.0, 3.0, 2.0, 1.0, 2.0])


def test_fix_surface_disjoint():
    idx = np.array([4.0, 4.0, 0.0, 100.0, 0.0, 4.0, 4.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    idx = np.array([400.0, 400.0, 0.0, 100.0, 400.0, 400.0], dtype=np.float64)
    idx = fix_surface_idx_missing(idx)
    assert np.allclose(idx, [400.0, 400.0, 400.0, 400.0, 400.0, 400.0])


test_fix_surface_idx_zeros()
test_fix_surface_disjoint()


# %%
def remove_small_components(img_thresh, area_thresh=0.1):
    """
    area_thresh: fraction of the largest component
    """
    (n_labels, label_ids, values, _) = cv2.connectedComponentsWithStats(img_thresh, 8)
    # max_i = max(range(1, n_labels), key=lambda i: values[i, cv2.CC_STAT_AREA])
    # max_area = values[max_i, cv2.CC_STAT_AREA]
    # area_thresh = 0.5 * max_area
    h, w = img_thresh.shape
    area_thresh = h * w * area_thresh
    use_ids = [
        i for i in range(1, n_labels) if values[i, cv2.CC_STAT_AREA] > area_thresh
    ]

    # Mask of the largest components
    mask = np.isin(label_ids, use_ids).astype(np.uint8)
    return mask


def find_surface_idx(img, thresh=0.25, area_thresh=0.001):
    img_blur = cv2.medianBlur(img, 3)
    # plt.imshow(img_blur, "gray")
    # plt.colorbar()

    thresh = 0.25 * 255
    _, img_thresh = cv2.threshold(img_blur, thresh, 1, cv2.THRESH_BINARY)

    img_mask = remove_small_components(img_thresh, area_thresh=area_thresh)
    # plt.imshow(img_mask, "gray")

    idx = np.argmax(img_mask, axis=0)
    # plt.plot(idx)

    ## Fix missing idx
    idx = fix_surface_idx_missing(idx)

    ## Filter
    # Not sure if wrap mode is the best here.
    idx = signal.savgol_filter(idx, 99, 2, mode="wrap")
    # plt.plot(idx)

    # Optionally crop to ztop

    return idx
