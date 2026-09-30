"""Compatibility contract for the retired legacy Query acceptance command."""

from __future__ import annotations

import pytest

from scripts.run_real_query_acceptance import (
    LegacyAcceptanceRetiredError,
    main,
    run,
)


@pytest.mark.asyncio
async def test_legacy_acceptance_run_fails_closed_without_creating_an_output_dir(
    tmp_path,
) -> None:
    """Removing the migration guard must not recreate the old fake-quality path."""
    output = tmp_path / "legacy-acceptance"

    with pytest.raises(LegacyAcceptanceRetiredError, match="run_real_rag_evaluation"):
        await run(output)

    assert not output.exists()


def test_legacy_acceptance_cli_returns_nonzero_with_the_real_evaluation_migration(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The legacy command remains actionable without claiming a passed evaluation."""
    output = tmp_path / "legacy-acceptance"

    status = main(["--output", str(output)])

    captured = capsys.readouterr()
    assert status == 2
    assert "run_real_rag_evaluation" in captured.err
    assert "PASSED" not in captured.out
    assert not output.exists()
