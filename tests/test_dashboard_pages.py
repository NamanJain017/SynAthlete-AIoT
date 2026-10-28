"""Every dashboard page renders without an exception (Streamlit's AppTest, headless).

Skipped when the prepared bundle is absent (CI, or before `python tasks.py demo --prepare`).
The Training lab button is not pressed here (that would train a model).
"""

import pytest

from src.dashboard.prepare import BUNDLE

PAGES = ["Home", "Athletes", "Athlete detail", "Live sensor", "Synthetic data", "Results",
         "Training lab", "Data & honesty"]


@pytest.mark.skipif(not (BUNDLE / "results.json").exists(), reason="dashboard bundle not built")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(BUNDLE.parents[2] / "src" / "dashboard" / "app.py"),
                              default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
