from pathlib import Path

import numpy as np

p = Path(__file__).absolute().parent.parent.parent / "logs"


def get_moving_avgs(returns_over_episodes, num_steps):
    """
    Return len(returns_over_episodes) - num_steps + 1 moving
    averages over num_steps steps.
    """
    if num_steps > len(returns_over_episodes):
        print(
            "issue in moving avg generation; "
            f"not enough data for {num_steps} step ma;\n"
        )
        raise ValueError(
            "num_steps should be less than"
            " or equal to length of returns_over_episodes"
        )
    return (
        np.correlate(returns_over_episodes, np.ones(num_steps), mode="valid")
        / num_steps
    )


def overlap_plot(save_path, data1, label1, data2, label2):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(range(len(data1)), data1, label=label1)
    ax.plot(range(len(data2)), data2, linestyle="--", label=label2)
    plt.legend()
    plt.yscale("log")
    ax.set_ylabel("steps per episode")
    ax.set_xlabel("num episodes")
    fig.tight_layout()
    plt.savefig(save_path / "10_steps_vs_2_steps.png")


if __name__ == "__main__":
    import pickle
    import sys

    import matplotlib.pyplot as plt

    assert len(sys.argv) == 3
    lens1, lens2 = sys.argv[1:]
    with open(p / lens1, "rb") as f:
        l1 = get_moving_avgs(pickle.load(f), 30)

    with open(p / lens2, "rb") as f:
        l2 = get_moving_avgs(pickle.load(f), 30)

    overlap_plot(
        p.parent / "assets/imgs/",
        l1,
        f"n={lens1.split('/')[-1].split('_')[0]}",
        l2,
        f"n={lens2.split('/')[-1].split('_')[0]}",
    )
