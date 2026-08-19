import pandas as pd
from src.data.timezone import in_local_window


def test_london_dst_window_moves_in_utc() -> None:
    winter = pd.DatetimeIndex(["2024-01-15 08:30:00+00:00"])
    summer = pd.DatetimeIndex(["2024-07-15 07:30:00+00:00"])
    assert in_local_window(winter, "Europe/London", "08:00", "10:00").iloc[0]
    assert in_local_window(summer, "Europe/London", "08:00", "10:00").iloc[0]
