from data.image_comparator.image_metadata import IMAGE_DIMENSIONS


DIMENSION_WEIGHTS = {
    "economic_practicality": 1.15,
    "leisure_preference": 1.00,
    "commute_affinity": 1.20,
    "family_friendliness": 1.00,
}


def initialize_vector():
    return {dimension: 0.0 for dimension in IMAGE_DIMENSIONS}


def compute_raw_preference_vector(selected_images):
    """
    selected_images should be a list of image metadata dictionaries.
    Each selected image must contain a 'scores' dictionary.
    """
    raw_vector = initialize_vector()

    for image in selected_images:
        scores = image.get("scores", {})

        for dimension in IMAGE_DIMENSIONS:
            raw_vector[dimension] += scores.get(dimension, 0)

    return raw_vector


def apply_dimension_weights(raw_vector):
    weighted_vector = {}

    for dimension, value in raw_vector.items():
        weight = DIMENSION_WEIGHTS.get(dimension, 1.0)
        weighted_vector[dimension] = round(value * weight, 4)

    return weighted_vector


def normalize_vector(weighted_vector):
    """
    Normalizes vector values to the range [-1, 1] using max absolute value.

    Positive values indicate preference toward that dimension.
    Negative values indicate preference away from that dimension.
    """
    max_abs_value = max(abs(value) for value in weighted_vector.values())

    if max_abs_value == 0:
        return {dimension: 0.0 for dimension in weighted_vector}

    return {
        dimension: round(value / max_abs_value, 4)
        for dimension, value in weighted_vector.items()
    }


def build_image_preference_output(selected_images):
    """
    Returns all three stages for debugging and report transparency:
    1. raw_vector
    2. weighted_vector
    3. normalized_vector
    """
    raw_vector = compute_raw_preference_vector(selected_images)
    weighted_vector = apply_dimension_weights(raw_vector)
    normalized_vector = normalize_vector(weighted_vector)

    return {
        "raw_vector": raw_vector,
        "weighted_vector": weighted_vector,
        "normalized_vector": normalized_vector,
    }