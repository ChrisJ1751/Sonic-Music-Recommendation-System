"""Tests for content-based (tag TF-IDF) similarity (src/content.py)."""
import numpy as np
import scipy.sparse as sp

from src import content


def test_content_similar_shares_tags():
    # artist0 & artist1 share tag 0; artist2 is on a different tag
    tfidf = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    top, scores = content.content_similar(tfidf, item_row=0, k=5)
    assert list(top) == [1]          # artist1 shares a tag; artist2 (sim 0) dropped
    assert scores[0] > 0


def test_content_similar_untagged_returns_empty():
    tfidf = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))  # artist0 has no tags
    top, scores = content.content_similar(tfidf, item_row=0, k=5)
    assert len(top) == 0 and len(scores) == 0


def test_build_artist_tag_tfidf_from_tmp(tmp_path):
    # write a minimal user_taggedartists.dat the loader can read
    p = tmp_path / "user_taggedartists.dat"
    p.write_text(
        "userID\tartistID\ttagID\tday\tmonth\tyear\n"
        "1\t10\t5\t1\t1\t2009\n"
        "1\t10\t6\t1\t1\t2009\n"
        "2\t20\t5\t1\t1\t2009\n"
        "3\t30\t9\t1\t1\t2009\n",
        encoding="utf-8",
    )
    item_ids = np.array([10, 20, 30])
    item_pos = {10: 0, 20: 1, 30: 2}
    tfidf = content.build_artist_tag_tfidf(item_ids, item_pos, raw_dir=tmp_path)
    assert tfidf.shape[0] == 3
    # artist 10 (row 0) and artist 20 (row 1) share tag 5 -> positive similarity
    top, scores = content.content_similar(tfidf, item_row=0, k=5)
    assert 1 in top.tolist()
    # rows are L2-normalised
    norm0 = np.sqrt(tfidf.getrow(0).multiply(tfidf.getrow(0)).sum())
    assert abs(norm0 - 1.0) < 1e-9
