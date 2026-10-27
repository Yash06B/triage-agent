from triage.store import IssueStore, tokenize


def test_tokenize_drops_stopwords_and_single_chars():
    assert tokenize("How do I use the --frozen flag in uv?") == ["use", "frozen", "flag"]


def test_search_ranks_the_matching_issue_first(issues, labels):
    store = IssueStore(issues, labels)
    hits = store.search("windows long path torch")
    assert {hits[0].number, hits[1].number} == {3, 5}


def test_as_of_hides_issues_filed_later(issues, labels):
    store = IssueStore(issues, labels, as_of=issues[4].created_at)  # issue 5's creation time
    assert len(store) == 4
    assert store.get(5) is None and store.get(6) is None


def test_as_of_shows_issues_closed_later_as_open(issues, labels):
    store = IssueStore(issues, labels, as_of=issues[4].created_at)
    issue = store.get(3)
    assert issue.state == "open" and issue.state_reason is None and issue.closed_at is None


def test_hidden_issue_is_not_searchable_or_readable(issues, labels):
    store = IssueStore(issues, labels, hide={3})
    assert store.get(3) is None
    assert 3 not in {i.number for i in store.search("long path windows torch")}


def test_state_filter(issues, labels):
    store = IssueStore(issues, labels)
    assert all(i.state == "closed" for i in store.search("windows path", state="closed"))
