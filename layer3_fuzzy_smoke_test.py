from layer3_fuzzy_inference import rank_pairs_fuzzy


def run_layer3_fuzzy_smoke_test():
    sample_pairs = [
        {
            "pair_id": "PAIR_001",
            "pair_feature_vector": {
                "opportunity_score": 0.90,
                "affordability_score": 0.85,
                "commute_score": 0.80,
                "amenity_score": 0.70,
                "space_fit_score": 0.75,
                "commute_distance_miles": 4.2,
            },
        },
        {
            "pair_id": "PAIR_002",
            "pair_feature_vector": {
                "opportunity_score": 0.70,
                "affordability_score": 0.45,
                "commute_score": 0.50,
                "amenity_score": 0.65,
                "space_fit_score": 0.60,
                "commute_distance_miles": 8.8,
            },
        },
        {
            "pair_id": "PAIR_003",
            "pair_feature_vector": {
                "opportunity_score": 0.30,
                "affordability_score": 0.20,
                "commute_score": 0.25,
                "amenity_score": 0.40,
                "space_fit_score": 0.35,
                "commute_distance_miles": 18.5,
            },
        },
    ]

    sample_preference_vector = {
        "economic_practicality": 0.80,
        "leisure_preference": 0.60,
        "commute_affinity": 0.90,
        "family_friendliness": 0.50,
    }

    ranked_pairs = rank_pairs_fuzzy(
        pair_representations=sample_pairs,
        preference_vector=sample_preference_vector,
    )

    print("\nLayer 3B Fuzzy Smoke Test Results")
    print("-" * 50)

    for rank, pair in enumerate(ranked_pairs, start=1):
        print(f"Rank {rank}: {pair.get('pair_id')}")
        print(f"  Fuzzy Score: {pair['layer3b_fuzzy_score']:.4f}")
        print(f"  Fuzzy Label: {pair['layer3b_fuzzy_label']}")
        print(f"  Output Memberships: {pair['layer3b_output_memberships']}")
        print(f"  Explanation: {pair['layer3b_explanation']}")
        print()


if __name__ == "__main__":
    run_layer3_fuzzy_smoke_test()
    