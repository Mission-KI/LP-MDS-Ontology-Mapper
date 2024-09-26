import logging

from edps.edps import EDPS


def main():
    # TODO use beebucket base logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    e = EDPS()


if __name__ == "__main__":
    main()
