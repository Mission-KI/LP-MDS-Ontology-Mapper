from io import BufferedIOBase, BytesIO
from pathlib import Path

from xhtml2pdf import pisa

from edps.file import build_real_sub_path
from edps.report.base import ReportGenerator, ReportInput
from edps.report.html import HtmlReportGenerator


class PdfReportGenerator(ReportGenerator):
    """Generates a PDF report."""

    def _init_(self):
        pisa.showLogging()

    async def generate(self, input: ReportInput, base_dir: Path, output_buffer: BufferedIOBase):
        html_buffer = BytesIO()
        await HtmlReportGenerator().generate(input, base_dir, html_buffer)

        pisa_context = pisa.CreatePDF(
            html_buffer,
            dest=output_buffer,
            link_callback=lambda url, origin: build_real_sub_path(base_dir, str(url)).as_posix(),
        )
        # Check for errors
        if pisa_context.err:
            raise RuntimeError("Error creating PDF report!")
