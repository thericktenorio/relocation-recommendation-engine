IMAGE_DIMENSIONS = [
    "economic_practicality",
    "leisure_preference",
    "commute_affinity",
    "family_friendliness",
]


def make_image(image_id, description, primary, secondary, scores):
    filename = image_id.lower().replace("image", "image") + ".jpg"

    return {
        "image_id": image_id,
        "image_path": f"data/image_comparator/images/{filename}",
        "description": description,
        "primary": primary,
        "secondary": secondary,
        "scores": scores,
    }


IMAGE_PAIRS = [
    {
        "pair_id": "pair_01",
        "left": make_image(
            "image1a",
            "Luxury apartment in modern high rise",
            "economic_practicality",
            None,
            {"economic_practicality": -2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 0},
        ),
        "right": make_image(
            "image1b",
            "Budget apartment, simple, clean, and cozy",
            "economic_practicality",
            None,
            {"economic_practicality": 2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_02",
        "left": make_image(
            "image2a",
            "Hiking trail, scenic mountains, with friends",
            "leisure_preference",
            "economic_practicality",
            {"economic_practicality": 2, "leisure_preference": 1, "commute_affinity": 0, "family_friendliness": 0},
        ),
        "right": make_image(
            "image2b",
            "Rooftop bar, city nightlife, social setting",
            "leisure_preference",
            "economic_practicality",
            {"economic_practicality": -2, "leisure_preference": 1, "commute_affinity": 0, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_03",
        "left": make_image(
            "image3a",
            "Downtown apartment area with walkable streets and bike paths",
            "commute_affinity",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": -1},
        ),
        "right": make_image(
            "image3b",
            "Suburban house in quiet neighborhood with access to major commute routes",
            "commute_affinity",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": 2, "family_friendliness": 1},
        ),
    },
    {
        "pair_id": "pair_04",
        "left": make_image(
            "image4a",
            "Kids playing in backyard of family home",
            "family_friendliness",
            "commute_affinity",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": 1, "family_friendliness": 2},
        ),
        "right": make_image(
            "image4b",
            "Single professional apartment with compact lifestyle",
            "family_friendliness",
            "commute_affinity",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": -1},
        ),
    },
    {
        "pair_id": "pair_05",
        "left": make_image(
            "image5a",
            "Premium dining experience",
            "economic_practicality",
            None,
            {"economic_practicality": -2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 0},
        ),
        "right": make_image(
            "image5b",
            "Value-conscious dining experience",
            "economic_practicality",
            None,
            {"economic_practicality": 2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_06",
        "left": make_image(
            "image6a",
            "Single person reading and listening to music in a cozy apartment",
            "leisure_preference",
            "economic_practicality",
            {"economic_practicality": 1, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 0},
        ),
        "right": make_image(
            "image6b",
            "Single person reading and listening to music outdoors",
            "leisure_preference",
            "economic_practicality",
            {"economic_practicality": 1, "leisure_preference": 2, "commute_affinity": 0, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_07",
        "left": make_image(
            "image7a",
            "Person using a bicycle to commute in a walkable/bike-friendly region",
            "commute_affinity",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": -1},
        ),
        "right": make_image(
            "image7b",
            "Person using a car to commute in a car-dependent region",
            "commute_affinity",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": 2, "family_friendliness": 1},
        ),
    },
    {
        "pair_id": "pair_08",
        "left": make_image(
            "image8a",
            "Group of friends running for exercise in a nature-oriented city",
            "leisure_preference",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 2, "commute_affinity": 0, "family_friendliness": -1},
        ),
        "right": make_image(
            "image8b",
            "Family exercising together near a lake or trail in a suburban/rural setting",
            "leisure_preference",
            "family_friendliness",
            {"economic_practicality": 0, "leisure_preference": 2, "commute_affinity": 0, "family_friendliness": 2},
        ),
    },
    {
        "pair_id": "pair_09",
        "left": make_image(
            "image9a",
            "Luxury indoor birthday party with family, friends, music, and alcohol",
            "economic_practicality",
            "leisure_preference",
            {"economic_practicality": -2, "leisure_preference": -2, "commute_affinity": 0, "family_friendliness": 0},
        ),
        "right": make_image(
            "image9b",
            "Backyard birthday party with family, friends, and outdoor activities",
            "economic_practicality",
            "leisure_preference",
            {"economic_practicality": 2, "leisure_preference": 1, "commute_affinity": 0, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_10",
        "left": make_image(
            "image10a",
            "Person buying a very good used car",
            "economic_practicality",
            "family_friendliness",
            {"economic_practicality": 2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 1},
        ),
        "right": make_image(
            "image10b",
            "Person buying a very good new car",
            "economic_practicality",
            "family_friendliness",
            {"economic_practicality": -2, "leisure_preference": 0, "commute_affinity": 0, "family_friendliness": 1},
        ),
    },
    {
        "pair_id": "pair_11",
        "left": make_image(
            "image11a",
            "Person enjoying the commute",
            "commute_affinity",
            None,
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": 2, "family_friendliness": 0},
        ),
        "right": make_image(
            "image11b",
            "Person enjoying convenience and proximity",
            "commute_affinity",
            None,
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": 0},
        ),
    },
    {
        "pair_id": "pair_12",
        "left": make_image(
            "image12a",
            "Large family having movie night with snacks and fun",
            "family_friendliness",
            "commute_affinity",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": 2},
        ),
        "right": make_image(
            "image12b",
            "Group of friends having movie night with snacks and fun",
            "family_friendliness",
            "commute_affinity",
            {"economic_practicality": 0, "leisure_preference": 0, "commute_affinity": -2, "family_friendliness": -1},
        ),
    },
]