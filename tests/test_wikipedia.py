"""Offline environment tests for Wikipedia state, loops, and max steps."""

from __future__ import annotations

from react_reproduction.tools.wikipedia import (
    ActionType,
    ToolAction,
    WikipediaArticle,
    WikipediaEnvironment,
    WikipediaSearchResult,
)


class FakeWikipediaClient:
    def search(self, query: str) -> WikipediaSearchResult:
        if query == "missing":
            return WikipediaSearchResult(query, None, ())
        article = WikipediaArticle(
            title="Albert Einstein",
            text=(
                "Albert Einstein was a German-born theoretical physicist. "
                "He was born in Ulm in 1879. He developed the theory of relativity."
            ),
        )
        return WikipediaSearchResult(query, article, ("Albert Einstein", "Einstein"))


def test_search_sets_current_article_and_lookup_advances() -> None:
    environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=5)

    search = environment.execute(ToolAction(ActionType.SEARCH, "Albert Einstein"))
    first_lookup = environment.execute(ToolAction(ActionType.LOOKUP, "born"))
    second_lookup = environment.execute(ToolAction(ActionType.LOOKUP, "born"))
    third_lookup = environment.execute(ToolAction(ActionType.LOOKUP, "born"))

    assert "Opened 'Albert Einstein'" in search.observation
    assert environment.current_article is not None
    assert "German-born" in first_lookup.observation
    assert "Ulm in 1879" in second_lookup.observation
    assert "No more matches" in third_lookup.observation
    assert third_lookup.terminated is False


def test_missing_article_returns_clear_observation() -> None:
    environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=3)
    result = environment.execute(ToolAction(ActionType.SEARCH, "missing"))
    assert result.terminated is False
    assert "Could not find [missing]" in result.observation
    assert "Similar:" in result.observation


def test_repeated_action_terminates_as_loop() -> None:
    environment = WikipediaEnvironment(
        FakeWikipediaClient(),
        max_steps=7,
        max_repeated_actions=3,
    )
    action = ToolAction(ActionType.SEARCH, "Albert Einstein")
    environment.execute(action)
    recovery = environment.execute(action)
    result = environment.execute(action)

    assert recovery.terminated is False
    assert recovery.tool_called is False
    assert "Repeated Search ignored" in recovery.observation
    assert "Lookup[short literal keyword]" in recovery.observation
    assert result.terminated is True
    assert result.termination_reason == "action_loop"


def test_paper_mode_allows_repeated_search_until_step_budget() -> None:
    environment = WikipediaEnvironment(
        FakeWikipediaClient(),
        max_steps=5,
        guard_repeated_search=False,
    )
    action = ToolAction(ActionType.SEARCH, "Albert Einstein")
    results = [environment.execute(action) for _ in range(5)]
    assert all(result.termination_reason != "action_loop" for result in results)
    assert results[-1].termination_reason == "max_steps_exceeded"


def test_max_steps_terminates_environment() -> None:
    environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=1)
    result = environment.execute(ToolAction(ActionType.SEARCH, "Albert Einstein"))
    assert result.terminated is True
    assert result.termination_reason == "max_steps_exceeded"


def test_finish_returns_answer_without_external_tool_call() -> None:
    environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=3)
    result = environment.execute(ToolAction(ActionType.FINISH, "Germany"))
    assert result.answer == "Germany"
    assert result.tool_called is False
    assert result.termination_reason == "completed"


def test_search_observation_uses_the_first_five_sentences() -> None:
    class SixSentenceClient:
        def search(self, query: str) -> WikipediaSearchResult:
            return WikipediaSearchResult(
                query,
                WikipediaArticle(
                    title="Exact Page",
                    text="One. Two. Three. Four. Five. Six.",
                ),
                ("Exact Page",),
            )

    environment = WikipediaEnvironment(SixSentenceClient(), max_steps=5)
    result = environment.execute(ToolAction(ActionType.SEARCH, "Exact Page"))
    assert "One. Two. Three. Four. Five." in result.observation
    assert "Six." not in result.observation
