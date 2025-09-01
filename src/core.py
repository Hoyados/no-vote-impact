from typing import Dict, Tuple
import numpy as np


def randomized_vector(min_val: float, max_val: float, loop_number: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(min_val, max_val, size=loop_number)


def simulate_vectorized(
    loop_number: int,
    ranges: Dict[str, Tuple[float, float]],
    initialratio_N: float,
    initialratio_A: float,
    initialratio_B: float,
    rng: np.random.Generator,
):
    voteratio = randomized_vector(*ranges["voteratio"], loop_number, rng)
    NtoA_ratio = randomized_vector(*ranges["NtoA_ratio"], loop_number, rng)
    AtoB_ratio = randomized_vector(*ranges["AtoB_ratio"], loop_number, rng)
    BtoA_ratio = randomized_vector(*ranges["BtoA_ratio"], loop_number, rng)

    new_voter_NtoA = initialratio_N * voteratio * NtoA_ratio
    new_voter_AtoB = initialratio_A * AtoB_ratio
    new_voter_NtoB = initialratio_N * voteratio * (1 - NtoA_ratio)
    new_voter_BtoA = initialratio_B * BtoA_ratio

    new_A_ratio = initialratio_A - new_voter_AtoB + new_voter_NtoA + new_voter_BtoA
    new_B_ratio = initialratio_B - new_voter_BtoA + new_voter_NtoB + new_voter_AtoB

    winner_flag = (new_B_ratio > new_A_ratio).astype(int)

    result = {
        "voteratio": np.round(voteratio * 100, 2),
        "NtoA_ratio": np.round(NtoA_ratio * 100, 2),
        "NtoB_ratio": np.round((1 - NtoA_ratio) * 100, 2),
        "AtoB_ratio": np.round(AtoB_ratio * 100, 2),
        "BtoA_ratio": np.round(BtoA_ratio * 100, 2),
        "A得票率": np.round(new_A_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "B得票率": np.round(new_B_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "逆転": winner_flag,
        "投票率": np.round((new_A_ratio + new_B_ratio) * 100, 2),
    }
    return result

