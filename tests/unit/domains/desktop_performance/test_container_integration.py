"""Tests for desktop_performance container integration."""

import pytest

from robotmcp.container import ServiceContainer, get_container, reset_container
from robotmcp.domains.desktop_performance import (
    ApplicationScope,
    DesktopKeywordOptimizer,
    ElementCache,
    EventCollector,
    InteractionProfile,
)


@pytest.fixture(autouse=True)
def clean_container():
    """Reset the global container before/after each test."""
    reset_container()
    yield
    reset_container()


class TestContainerDesktopIntegration:
    def test_desktop_optimizer_is_singleton(self):
        c = get_container()
        opt1 = c.desktop_optimizer
        opt2 = c.desktop_optimizer
        assert opt1 is opt2
        assert isinstance(opt1, DesktopKeywordOptimizer)

    def test_desktop_event_collector_is_singleton(self):
        c = get_container()
        ec1 = c.desktop_event_collector
        ec2 = c.desktop_event_collector
        assert ec1 is ec2
        assert isinstance(ec1, EventCollector)

    def test_get_element_cache(self):
        c = get_container()
        cache = c.get_element_cache("s1")
        assert isinstance(cache, ElementCache)
        assert cache.session_id == "s1"
        # Same session returns same cache
        assert c.get_element_cache("s1") is cache

    def test_get_element_cache_different_sessions(self):
        c = get_container()
        cache1 = c.get_element_cache("s1")
        cache2 = c.get_element_cache("s2")
        assert cache1 is not cache2

    def test_get_application_scope(self):
        c = get_container()
        scope = c.get_application_scope("s1")
        assert isinstance(scope, ApplicationScope)
        assert scope.session_id == "s1"
        assert c.get_application_scope("s1") is scope

    def test_get_interaction_profile(self):
        c = get_container()
        profile = c.get_interaction_profile("s1")
        assert isinstance(profile, InteractionProfile)
        assert profile.session_id == "s1"

    def test_clear_desktop_session(self):
        c = get_container()
        c.get_element_cache("s1")
        c.get_application_scope("s1")
        c.get_interaction_profile("s1")
        c.clear_desktop_session("s1")
        # New instances created after clear
        cache = c.get_element_cache("s1")
        assert cache.size == 0

    def test_clear_session_includes_desktop(self):
        c = get_container()
        cache = c.get_element_cache("s1")
        c.clear_session("s1")
        # Should create new cache since old one was cleared
        new_cache = c.get_element_cache("s1")
        assert new_cache is not cache

    def test_optimizer_has_event_publisher(self):
        c = get_container()
        opt = c.desktop_optimizer
        assert opt.event_publisher is not None
        assert opt.event_publisher is c.desktop_event_collector

    def test_scope_manager_has_event_publisher(self):
        c = get_container()
        opt = c.desktop_optimizer
        assert opt.scope_manager.event_publisher is not None
        assert opt.scope_manager.event_publisher is c.desktop_event_collector
