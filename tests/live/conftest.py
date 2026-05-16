# Shared fixtures for the live test tier.
# Live tests never run in CI — guard every module with:
#   pytestmark = pytest.mark.skipif(not os.getenv("LIVE_TESTS"), reason="opt-in: set LIVE_TESTS=1")
