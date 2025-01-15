from logging import getLogger

from edps.task import SimpleTaskContext, TaskContext


async def test_task(output_context):
    ctx = SimpleTaskContext(getLogger("task"), output_context)
    r1 = ctx.exec(my_task_with_args, 42, b=10)
    assert r1 == 52
    r2 = await ctx.exec(my_async_task, "there")
    assert r2 == "Hello there!"


def my_task_with_args(ctx: TaskContext, a: int, b: int) -> int:
    ctx.logger.info(f"{a} + {b} = {a + b}")
    return a + b


async def my_async_task(ctx: TaskContext, s: str) -> str:
    ctx.logger.info("Starting..")
    hello = ctx.exec(my_simple_hello_task)
    result = f"{hello} {s}!"
    ctx.logger.info(f"Returning: {result}")
    return result


def my_simple_hello_task(ctx: TaskContext) -> str:
    ctx.logger.info("Working..")
    return "Hello"
