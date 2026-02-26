"""Tests for desktop_performance value objects."""

import pytest

from robotmcp.domains.desktop_performance.value_objects import (
    CacheCapacity,
    CacheKey,
    CacheTTL,
    InteractionSpeed,
    PointerSpeedProfile,
    POINTER_SPEED_FAST,
    POINTER_SPEED_INSTANT,
    POINTER_SPEED_REALISTIC,
    SPEED_PROFILES,
    XPathAxis,
    XPathTransform,
    _is_platynui_xpath,
)


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_valid_creation(self):
        k = CacheKey(xpath="//control:Button[@Name='OK']", session_id="s1")
        assert k.xpath == "//control:Button[@Name='OK']"
        assert k.session_id == "s1"

    def test_empty_xpath_raises(self):
        with pytest.raises(ValueError, match="xpath must not be empty"):
            CacheKey(xpath="", session_id="s1")

    def test_empty_session_raises(self):
        with pytest.raises(ValueError, match="session_id must not be empty"):
            CacheKey(xpath="//Button", session_id="")

    def test_frozen(self):
        k = CacheKey(xpath="//Button", session_id="s1")
        with pytest.raises(AttributeError):
            k.xpath = "changed"

    def test_key_str(self):
        k = CacheKey(xpath="//control:Button[@Name='2']", session_id="s1")
        assert k.key_str == "s1://control:Button[@Name='2']"

    def test_from_keyword_args_with_xpath(self):
        k = CacheKey.from_keyword_args(
            session_id="s1",
            keyword="pointer click",
            arguments=("//control:Button[@Name='2']",),
        )
        assert k is not None
        assert k.xpath == "//control:Button[@Name='2']"

    def test_from_keyword_args_without_xpath(self):
        k = CacheKey.from_keyword_args(
            session_id="s1",
            keyword="pointer click",
            arguments=("not-an-xpath",),
        )
        assert k is None

    def test_from_keyword_args_empty(self):
        k = CacheKey.from_keyword_args(
            session_id="s1",
            keyword="pointer click",
            arguments=(),
        )
        assert k is None

    def test_equality(self):
        k1 = CacheKey(xpath="//Button", session_id="s1")
        k2 = CacheKey(xpath="//Button", session_id="s1")
        assert k1 == k2

    def test_inequality_different_session(self):
        k1 = CacheKey(xpath="//Button", session_id="s1")
        k2 = CacheKey(xpath="//Button", session_id="s2")
        assert k1 != k2

    def test_hashable(self):
        k = CacheKey(xpath="//Button", session_id="s1")
        assert hash(k)  # Just verify it doesn't crash
        s = {k}
        assert k in s


# ---------------------------------------------------------------------------
# CacheTTL
# ---------------------------------------------------------------------------


class TestCacheTTL:
    def test_default_is_60s(self):
        ttl = CacheTTL.default()
        assert ttl.value_seconds == 60.0

    def test_short_is_5s(self):
        ttl = CacheTTL.short()
        assert ttl.value_seconds == 5.0

    def test_below_min_raises(self):
        with pytest.raises(ValueError, match="CacheTTL must be"):
            CacheTTL(value_seconds=0.5)

    def test_above_max_raises(self):
        with pytest.raises(ValueError, match="CacheTTL must be"):
            CacheTTL(value_seconds=400.0)

    def test_boundary_min(self):
        ttl = CacheTTL(value_seconds=1.0)
        assert ttl.value_seconds == 1.0

    def test_boundary_max(self):
        ttl = CacheTTL(value_seconds=300.0)
        assert ttl.value_seconds == 300.0

    def test_is_expired_true(self):
        ttl = CacheTTL(value_seconds=10.0)
        assert ttl.is_expired(cached_at=0.0, now=11.0) is True

    def test_is_expired_false(self):
        ttl = CacheTTL(value_seconds=10.0)
        assert ttl.is_expired(cached_at=0.0, now=5.0) is False

    def test_is_expired_exact_boundary(self):
        ttl = CacheTTL(value_seconds=10.0)
        # Exactly at TTL boundary is not expired (>= not >)
        assert ttl.is_expired(cached_at=0.0, now=10.0) is False

    def test_frozen(self):
        ttl = CacheTTL.default()
        with pytest.raises(AttributeError):
            ttl.value_seconds = 999


# ---------------------------------------------------------------------------
# XPathTransform
# ---------------------------------------------------------------------------


class TestXPathTransform:
    def test_absolute_to_relative(self):
        t = XPathTransform.to_relative('//control:Button[@Name="OK"]')
        assert t.transformed == './/control:Button[@Name="OK"]'
        assert t.scoping_applied is True
        assert t.axis == XPathAxis.ABSOLUTE

    def test_already_relative(self):
        t = XPathTransform.to_relative('.//control:Button[@Name="OK"]')
        assert t.transformed == './/control:Button[@Name="OK"]'
        assert t.scoping_applied is False
        assert t.axis == XPathAxis.RELATIVE

    def test_dot_slash_relative(self):
        t = XPathTransform.to_relative('./control:Button')
        assert t.scoping_applied is False
        assert t.axis == XPathAxis.RELATIVE

    def test_single_slash_unchanged(self):
        t = XPathTransform.to_relative('/Window/Button')
        assert t.transformed == '/Window/Button'
        assert t.scoping_applied is False
        assert t.axis == XPathAxis.ABSOLUTE

    def test_identity_absolute(self):
        t = XPathTransform.identity("//Button")
        assert t.scoping_applied is False
        assert t.axis == XPathAxis.ABSOLUTE

    def test_identity_relative(self):
        t = XPathTransform.identity(".//Button")
        assert t.scoping_applied is False
        assert t.axis == XPathAxis.RELATIVE

    def test_frozen(self):
        t = XPathTransform.to_relative("//Button")
        with pytest.raises(AttributeError):
            t.original = "changed"

    def test_empty_original_raises(self):
        with pytest.raises(ValueError, match="original must not be empty"):
            XPathTransform(original="", transformed="x", axis=XPathAxis.ABSOLUTE, scoping_applied=False)

    def test_empty_transformed_raises(self):
        with pytest.raises(ValueError, match="transformed must not be empty"):
            XPathTransform(original="x", transformed="", axis=XPathAxis.ABSOLUTE, scoping_applied=False)

    def test_preserves_complex_predicate(self):
        xpath = '//control:Button[@Name="2" and @IsEnabled="True"]'
        t = XPathTransform.to_relative(xpath)
        assert t.transformed == '.' + xpath
        assert t.scoping_applied is True


# ---------------------------------------------------------------------------
# CacheCapacity
# ---------------------------------------------------------------------------


class TestCacheCapacity:
    def test_default(self):
        c = CacheCapacity.default()
        assert c.max_entries == 200

    def test_below_min_raises(self):
        with pytest.raises(ValueError, match="CacheCapacity must be"):
            CacheCapacity(max_entries=5)

    def test_above_max_raises(self):
        with pytest.raises(ValueError, match="CacheCapacity must be"):
            CacheCapacity(max_entries=2000)

    def test_boundary_values(self):
        assert CacheCapacity(max_entries=10).max_entries == 10
        assert CacheCapacity(max_entries=1000).max_entries == 1000

    def test_frozen(self):
        c = CacheCapacity.default()
        with pytest.raises(AttributeError):
            c.max_entries = 999


# ---------------------------------------------------------------------------
# PointerSpeedProfile
# ---------------------------------------------------------------------------


class TestPointerSpeedProfile:
    def test_instant_all_zero(self):
        p = POINTER_SPEED_INSTANT
        assert p.after_move_delay_ms == 0
        assert p.after_input_delay_ms == 0
        assert p.press_release_delay_ms == 0
        assert p.after_click_delay_ms == 0
        assert p.motion_mode == 0
        assert p.speed == InteractionSpeed.INSTANT

    def test_fast_minimal_delays(self):
        p = POINTER_SPEED_FAST
        assert p.press_release_delay_ms == 5
        assert p.after_click_delay_ms == 10
        assert p.speed == InteractionSpeed.FAST

    def test_realistic_human_delays(self):
        p = POINTER_SPEED_REALISTIC
        assert p.after_move_delay_ms == 50
        assert p.motion_mode == 2  # BEZIER
        assert p.speed == InteractionSpeed.REALISTIC

    def test_to_overrides_dict_keys(self):
        d = POINTER_SPEED_FAST.to_overrides_dict()
        expected_keys = {
            "after_move_delay_ms",
            "after_input_delay_ms",
            "press_release_delay_ms",
            "after_click_delay_ms",
            "motion",
            "max_move_duration_ms",
            "speed_factor",
        }
        assert set(d.keys()) == expected_keys

    def test_frozen(self):
        with pytest.raises(AttributeError):
            POINTER_SPEED_FAST.speed = InteractionSpeed.INSTANT

    def test_speed_profiles_dict(self):
        assert len(SPEED_PROFILES) == 3
        assert InteractionSpeed.INSTANT in SPEED_PROFILES
        assert InteractionSpeed.FAST in SPEED_PROFILES
        assert InteractionSpeed.REALISTIC in SPEED_PROFILES


# ---------------------------------------------------------------------------
# _is_platynui_xpath helper
# ---------------------------------------------------------------------------


class TestIsPlatynuiXpath:
    def test_absolute_control(self):
        assert _is_platynui_xpath("//control:Button[@Name='2']") is True

    def test_relative_control(self):
        assert _is_platynui_xpath(".//control:Button[@Name='2']") is True

    def test_item_namespace(self):
        assert _is_platynui_xpath("//item:ListItem[@Name='File']") is True

    def test_app_namespace(self):
        assert _is_platynui_xpath("//app:Application[@Name='Calc']") is True

    def test_bare_slash(self):
        assert _is_platynui_xpath("/Window/Button") is True

    def test_css_selector(self):
        assert _is_platynui_xpath("css=.button") is False

    def test_id_locator(self):
        assert _is_platynui_xpath("id=submit") is False

    def test_empty_string(self):
        assert _is_platynui_xpath("") is False

    def test_regular_text(self):
        assert _is_platynui_xpath("just some text") is False

    def test_edit_namespace(self):
        assert _is_platynui_xpath("//edit:TextBox[@Name='Search']") is True

    def test_menu_namespace(self):
        assert _is_platynui_xpath("//menu:MenuItem[@Name='File']") is True
