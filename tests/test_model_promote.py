from sentimental_cap_predictor.agent.dispatcher import dispatch


def test_model_promote_swaps_files(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text("src_cfg")
    (src / "weights.bin").write_text("src_w")
    (dst / "config.json").write_text("dst_cfg")
    (dst / "weights.bin").write_text("dst_w")

    res_dry = dispatch(
        {
            "command": "model.promote",
            "src": str(src),
            "dst": str(dst),
            "dry_run": True,
        }
    )
    assert res_dry.ok
    assert (src / "config.json").read_text() == "src_cfg"
    assert (dst / "config.json").read_text() == "dst_cfg"
    assert str(dst / "config.json") in res_dry.artifacts
    assert str(dst / "weights.bin") in res_dry.artifacts

    res = dispatch({"command": "model.promote", "src": str(src), "dst": str(dst)})
    assert res.ok
    assert (dst / "config.json").read_text() == "src_cfg"
    assert (dst / "weights.bin").read_text() == "src_w"
    assert (src / "config.json").read_text() == "dst_cfg"
    assert (src / "weights.bin").read_text() == "dst_w"
    assert str(dst / "config.json") in res.artifacts
    assert str(dst / "weights.bin") in res.artifacts
