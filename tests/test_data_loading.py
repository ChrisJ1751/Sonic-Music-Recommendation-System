"""Tests for data loading (src/data_loading.py)."""
import numpy as np
import pandas as pd

from src import data_loading as dl


def test_load_artists_is_utf8(tmp_path):
    # artists.dat is UTF-8; regression guard against the latin-1 mojibake bug
    # ("Björk" must not become "BjÃ¶rk").
    (tmp_path / "artists.dat").write_text(
        "id\tname\turl\tpictureURL\n1\tBjörk\thttp://x\thttp://y\n",
        encoding="utf-8",
    )
    df = dl.load_artists(raw_dir=tmp_path)
    assert df.iloc[0]["name"] == "Björk"


def test_build_interaction_matrix_shape_and_maps():
    df = pd.DataFrame({"userID": [2, 2, 5], "artistID": [51, 99, 51], "weight": [10, 3, 7]})
    im = dl.build_interaction_matrix(df)
    assert im.shape == (2, 2)               # 2 users, 2 artists
    assert im.matrix.nnz == 3
    # index maps round-trip to original IDs
    assert im.user_ids[im.user_pos[2]] == 2
    assert im.item_ids[im.item_pos[51]] == 51
    # weight preserved
    assert im.matrix[im.user_pos[2], im.item_pos[51]] == 10
    assert np.array_equal(np.sort(im.user_ids), [2, 5])
