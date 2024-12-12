import asyncio
import logging
import sys
from logging import getLogger

from edp.pontusx.args import parse_args
from edp.pontusx.service import run_service


def config_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


async def main():
    config_logging()
    logger = getLogger()
    args = parse_args(sys.argv[1:])
    logger.debug(f"Pontus-X CLI got these arguments: {args}")
    logger.info("Processing asset with DID='%s'", args.did)
    await run_service(logger, args)
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(main())
