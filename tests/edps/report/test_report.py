from logging import getLogger

from pytest import fixture, mark

from edps.report import HtmlReportGenerator, PdfReportGenerator, ReportInput
from edps.taskcontext import TaskContext
from tests.edps.test_edps import analyse_asset, read_edp_file


@fixture
def report_output_path(tmp_path):
    return tmp_path


# This test generates multiple reports because fixture "asset_path" iterates through multiple assets.
@mark.slow
async def test_all_reports(ctx: TaskContext, asset_path, report_output_path):
    json_path = await analyse_asset(ctx, asset_path)
    edp = read_edp_file(ctx.output_path / json_path)
    report_input = ReportInput(edp=edp)

    report_html = report_output_path / f"{asset_path.name}.html"
    with report_html.open("wb") as file_io:
        await HtmlReportGenerator().generate(report_input, ctx.output_path, file_io)
    assert report_html.exists()

    report_pdf = report_output_path / f"{asset_path.name}.pdf"
    with report_pdf.open("wb") as file_io:
        await PdfReportGenerator().generate(report_input, ctx.output_path, file_io)
    assert report_pdf.exists()

    getLogger().info("Report output path: %s", report_output_path)
