from app.core import c3_sow
from ml.c3 import sow_tables


def test_trainer_tables_mirror_the_sow_writer():
    assert sow_tables.TEACHABLE == c3_sow.TEACHABLE
    assert sow_tables.LIST_SECTION == c3_sow._LIST_SECTION
