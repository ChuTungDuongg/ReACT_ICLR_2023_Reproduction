"""A small stateful Wikipedia environment inspired by the ReAct paper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


DEFAULT_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "ReActPaperReproduction/0.8.1 "
    "(https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction)"
)


class WikipediaError(RuntimeError):
    """Raised after MediaWiki retries are exhausted or a response is invalid."""


class ActionType(str, Enum):
    SEARCH = "Search"
    LOOKUP = "Lookup"
    FINISH = "Finish"


@dataclass(frozen=True, slots=True)
class ToolAction:
    action_type: ActionType
    argument: str

    def __post_init__(self) -> None:
        if not self.argument.strip():
            raise ValueError("Tool action argument cannot be empty.")

    @property
    def canonical(self) -> str:
        return f"{self.action_type.value}[{self.argument.strip()}]"


@dataclass(frozen=True, slots=True)
class WikipediaArticle:
    title: str
    text: str
    is_disambiguation: bool = False


@dataclass(frozen=True, slots=True)
class WikipediaSearchResult:
    query: str
    article: WikipediaArticle | None
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolExecution:
    observation: str
    terminated: bool
    termination_reason: str | None = None
    answer: str | None = None
    tool_called: bool = True


class WikipediaClient:
    """Thin MediaWiki Action API client with bounded retries and timeouts."""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = 10.0,
        max_search_results: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_search_results <= 0:
            raise ValueError("max_search_results must be positive.")
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_search_results = max_search_results
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def search(self, query: str) -> WikipediaSearchResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Wikipedia search query cannot be empty.")
        payload = self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": cleaned_query,
                "srlimit": self.max_search_results,
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
            }
        )
        raw_results = payload.get("query", {}).get("search", [])
        candidates = tuple(
            str(result["title"])
            for result in raw_results
            if isinstance(result, dict) and result.get("title")
        )
        if not candidates:
            return WikipediaSearchResult(cleaned_query, None, ())

        normalized_query = _normalize_title(cleaned_query)
        selected_title = next(
            (
                title
                for title in candidates
                if _normalize_title(title) == normalized_query
            ),
            candidates[0],
        )
        article = self.get_article(selected_title)
        return WikipediaSearchResult(cleaned_query, article, candidates)

    def get_article(self, title: str) -> WikipediaArticle | None:
        payload = self._request(
            {
                "action": "query",
                "prop": "extracts|pageprops",
                "titles": title,
                "redirects": 1,
                "explaintext": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages or not isinstance(pages[0], dict) or pages[0].get("missing"):
            return None
        page = pages[0]
        return WikipediaArticle(
            title=str(page.get("title", title)),
            text=str(page.get("extract", "")).strip(),
            is_disambiguation="disambiguation" in page.get("pageprops", {}),
        )

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._request_with_retry(params)
        except (requests.RequestException, ValueError) as error:
            raise WikipediaError(f"Wikipedia request failed: {error}") from error

    @retry(
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _request_with_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            self.api_url,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("MediaWiki response root is not an object.")
        if "error" in payload:
            raise ValueError(f"MediaWiki API error: {payload['error']}")
        return payload


class WikipediaEnvironment:
    """Execute Search, Lookup, and Finish while retaining article state."""

    def __init__(
        self,
        client: WikipediaClient,
        *,
        max_steps: int = 7,
        max_repeated_actions: int = 3,
        max_observation_chars: int = 1_200,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if max_repeated_actions <= 1:
            raise ValueError("max_repeated_actions must be greater than one.")
        if max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive.")
        self.client = client
        self.max_steps = max_steps
        self.max_repeated_actions = max_repeated_actions
        self.max_observation_chars = max_observation_chars
        self.reset()

    def reset(self) -> None:
        self.current_article: WikipediaArticle | None = None
        self.steps = 0
        self.terminated = False
        self._last_action: str | None = None
        self._consecutive_repeats = 0
        self._lookup_offsets: dict[str, int] = {}

    def execute(self, action: ToolAction) -> ToolExecution:
        if self.terminated:
            raise RuntimeError("Wikipedia environment is already terminated; call reset().")

        canonical = action.canonical.casefold()
        if action.action_type is ActionType.SEARCH and canonical == self._last_action:
            self._consecutive_repeats += 1
        elif action.action_type is ActionType.SEARCH:
            self._last_action = canonical
            self._consecutive_repeats = 1
        else:
            # Repeating Lookup is meaningful: it advances to the next matching
            # sentence. Only identical consecutive Search actions form a loop.
            self._last_action = None
            self._consecutive_repeats = 0

        self.steps += 1

        if self._consecutive_repeats >= self.max_repeated_actions:
            self.terminated = True
            return ToolExecution(
                observation=(
                    f"Action loop detected for {action.canonical}. The repeated "
                    "search was not executed; make a best-effort Finish action."
                ),
                terminated=True,
                termination_reason="action_loop",
                tool_called=False,
            )

        if action.action_type is ActionType.SEARCH and self._consecutive_repeats == 2:
            article_hint = (
                f" The current article is '{self.current_article.title}'."
                if self.current_article is not None
                else ""
            )
            execution = ToolExecution(
                observation=(
                    f"Repeated Search ignored for {action.canonical}. Search opens "
                    f"an article; it does not search inside one.{article_hint} Use "
                    "Lookup[short literal keyword] for a detail in the current "
                    "article, Search[a different canonical entity or candidate "
                    "title] for another article, or Finish[answer]."
                ),
                terminated=False,
                tool_called=False,
            )
            return self._apply_step_limit(execution)

        if action.action_type is ActionType.FINISH:
            self.terminated = True
            return ToolExecution(
                observation=f"Finished with answer: {action.argument.strip()}",
                terminated=True,
                termination_reason="completed",
                answer=action.argument.strip(),
                tool_called=False,
            )

        if action.action_type is ActionType.SEARCH:
            execution = self._search(action.argument)
        else:
            execution = self._lookup(action.argument)

        return self._apply_step_limit(execution)

    def _apply_step_limit(self, execution: ToolExecution) -> ToolExecution:
        if self.steps >= self.max_steps and not execution.terminated:
            self.terminated = True
            return ToolExecution(
                observation=(
                    f"{execution.observation}\nMaximum environment steps reached "
                    f"({self.max_steps})."
                ),
                terminated=True,
                termination_reason="max_steps_exceeded",
                tool_called=execution.tool_called,
            )
        return execution

    def _search(self, query: str) -> ToolExecution:
        result = self.client.search(query)
        if result.article is None:
            self.current_article = None
            return ToolExecution(
                observation=f"No Wikipedia article found for: {query.strip()}.",
                terminated=False,
            )

        self.current_article = result.article
        self._lookup_offsets.clear()
        summary = _article_summary(result.article.text, self.max_observation_chars)
        candidate_text = "; ".join(result.candidates)
        if result.article.is_disambiguation:
            observation = (
                f"Ambiguous Wikipedia article: {result.article.title}. "
                f"Search candidates: {candidate_text}."
            )
        elif _normalize_title(result.article.title) != _normalize_title(query):
            observation = (
                f"Search results: {candidate_text}. Opened top result "
                f"'{result.article.title}'. {summary}"
            )
        else:
            observation = f"Opened '{result.article.title}'. {summary}"
        return ToolExecution(observation=observation, terminated=False)

    def _lookup(self, term: str) -> ToolExecution:
        if self.current_article is None:
            return ToolExecution(
                observation="Lookup requires a current article. Use Search first.",
                terminated=False,
            )
        sentences = _split_sentences(self.current_article.text)
        matches = [
            sentence
            for sentence in sentences
            if term.strip().casefold() in sentence.casefold()
        ]
        lookup_key = term.strip().casefold()
        offset = self._lookup_offsets.get(lookup_key, 0)
        if not matches:
            return ToolExecution(
                observation=(
                    f"No match for '{term.strip()}' in "
                    f"'{self.current_article.title}'."
                ),
                terminated=False,
            )
        if offset >= len(matches):
            return ToolExecution(
                observation=(
                    f"No more matches for '{term.strip()}' in "
                    f"'{self.current_article.title}'."
                ),
                terminated=False,
            )
        self._lookup_offsets[lookup_key] = offset + 1
        observation = (
            f"Lookup result {offset + 1}/{len(matches)} in "
            f"'{self.current_article.title}': {matches[offset]}"
        )
        return ToolExecution(
            observation=observation[: self.max_observation_chars],
            terminated=False,
        )


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ").strip()).casefold()


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def _article_summary(text: str, max_chars: int) -> str:
    sentences = _split_sentences(text)
    summary = " ".join(sentences[:3]) if sentences else "No article extract available."
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."
