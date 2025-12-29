from typing import Literal


def retrieval_sanity_check(hits) -> Literal[True, False, None]:
    if len(hits) == 0:
        return False

    chunks = [h["chunk"] for h in hits]

    if len(set(chunks)) == 1 and len(chunks) > 1:
        return False

    if all(len(c) < 30 for c in chunks):
        return False

    return None
