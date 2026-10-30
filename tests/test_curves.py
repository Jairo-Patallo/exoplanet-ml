from exoplanet.curves import MJD_OFFSET, expected_transits, kepler_span, select_segments, target_name


def test_expected_transits():
    assert expected_transits(0, 9, 3.0, 1.0) == 3             # t = 1, 4, 7
    assert expected_transits(0, 10, 3.0, 1.0) == 4            # el 10 también cuenta
    assert expected_transits(1.5, 3.5, 3.0, 1.0) == 0
    assert expected_transits(1.0, 1.0, 3.0, 1.0) == 1          # el extremo cuenta


def test_select_segments_accumulates_until_over_the_cap():
    spans = [(i * 30.0, i * 30.0 + 27.0) for i in range(24)]   # 24 sectores tipo TESS de 27 d
    chosen = select_segments(spans, period=6.0, t0=0.0)         # todos con ~4-5 tránsitos esperados
    assert len(chosen) == 4                                     # 27 + 27 + 27 + 27 = 108 d > 100
    assert chosen == sorted(chosen)


def test_select_segments_prefers_segments_with_transits():
    spans = [(0.0, 40.0), (100.0, 190.0), (200.0, 210.0)]
    period, t0 = 30.0, 0.0                                      # tránsitos esperados: 2 (0, 30), 3 (120, 150, 180) y 1 (210)
    assert select_segments(spans, period, t0, max_days=50) == [1]       # el de 3 tránsitos (90 d) ya supera el tope
    assert select_segments(spans, period, t0, max_days=100) == [0, 1]   # luego el de 2 (total 130 d > 100)
    assert select_segments(spans, period, t0, max_days=500) == [0, 1, 2]


def test_select_segments_all_if_under_the_cap():
    spans = [(0.0, 40.0), (50.0, 90.0)]
    assert select_segments(spans, 5.0, 0.0, max_days=200) == [0, 1]


def test_mjd_offsets_match_the_times_measured_in_the_downloaded_files():
    assert abs(54953.02777777778 + MJD_OFFSET["kepler"] - 120.5) < 0.1      # inicio medido de KIC 11446443 Q0: BKJD 120.5
    assert abs(57061.28482844 + MJD_OFFSET["k2"] - 2228.8) < 0.1            # inicio medido de EPIC 211089792 C4: BKJD 2228.8
    assert abs(58324.7940765625 + MJD_OFFSET["tess"] - 1325.3) < 0.1        # inicio medido de TIC 261136679 S1: BTJD 1325.3


def test_kepler_quarters_table():
    start, end = kepler_span("Kepler Quarter 00")
    assert abs(start - 120.5) < 0.1 and abs(end - 130.2) < 0.2               # Q0 medido en el archivo de KIC 11446443
    spans = [kepler_span(f"Kepler Quarter {q:02d}") for q in range(18)]
    assert all(a < b for a, b in spans)
    for (_, prev_end), (nxt_start, _) in zip(spans, spans[1:]):
        assert 0 < nxt_start - prev_end < 20, "trimestres consecutivos: sin solape y con un hueco corto"


def test_target_name():
    assert target_name("KIC_11446443") == "KIC 11446443" and target_name("TIC_261136679") == "TIC 261136679"
