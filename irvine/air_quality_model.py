#! /usr/bin/env python

from irvine.air_quality_eda import get_air_quality_dataset


def main() -> None:
    x = get_air_quality_dataset()
    print(x)


if __name__ == "__main__":
    main()
