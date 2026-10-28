from ml.c3.quote_heads import AFTER, HEIGHT


def test_height_reads_lifts_and_mounting_heights_not_cable_lengths():
    for t in ("Scissor lift required for the lobby", "ceiling height is 14 ft", "mounted at 12 feet",
              "20 ft ceilings in the warehouse", "lift rental"):
        assert HEIGHT.search(t), t
    for t in ("200 foot cable run", "MDF to the Roof", "Cat6 from the IDF, 150 feet"):
        assert not HEIGHT.search(t), t


def test_after_hours_reads_the_work_schedule_not_small_talk():
    for t in ("install after hours only", "work must be done over the weekend", "night shift cutover",
              "outside business hours", "Nights and weekends", "Mount 3 cameras overnight"):
        assert AFTER.search(t), t
    for t in ("hope you had a good Memorial Day weekend", "see you next weekend", "last night's game", "we can send it overnight shipping"):
        assert not AFTER.search(t), t
