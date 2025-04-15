from mds_mapper import convert_edp_to_mds


async def test_service_on_csv(path_edp_csv, path_work):
    await convert_edp_to_mds(path_edp_csv, path_work / "edp_mds.json")


async def test_service_on_pdf(path_edp_pdf, path_work):
    await convert_edp_to_mds(path_edp_pdf, path_work / "edp_mds.json")
