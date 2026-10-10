"""Peek at a Lightning TensorBoard event file: list scalar tags and counts."""

import argparse

from tensorboard.backend.event_processing import event_accumulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    a = ap.parse_args()

    ea = event_accumulator.EventAccumulator(
        a.events, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    scalars = ea.Tags()["scalars"]
    print("scalar tags:", len(scalars))
    for t in scalars:
        items = ea.Scalars(t)
        first, last = items[0], items[-1]
        print(
            "%-16s n=%-9d step=[%d,%d] wall=[%.0f,%.0f] v0=%.6g vlast=%.6g"
            % (
                t,
                len(items),
                first.step,
                last.step,
                first.wall_time,
                last.wall_time,
                first.value,
                last.value,
            )
        )


if __name__ == "__main__":
    main()
