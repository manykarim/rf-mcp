"""
SimHash-based list folding for token optimization.

Detects and compresses repetitive list items in ARIA accessibility tree snapshots.
This is particularly effective for e-commerce product listings, search results,
data tables, and other content with structurally similar repeated elements.

Example compression:
    Before: 50 listitem elements fully expanded (~2500 tokens)
    After:  listitem [ref=e234] + (... and 47 more similar) [refs: e235-e281] (~50 tokens)

The SimHash algorithm provides O(n) time complexity for detecting similar items,
making it suitable for processing large DOM structures efficiently.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import re


@dataclass
class SimHashConfig:
    """Configuration for SimHash algorithm and list folding behavior."""

    hash_bits: int = 64
    """Number of bits in the SimHash fingerprint (64 recommended for accuracy)."""

    ngram_size: int = 3
    """Size of word n-grams for feature extraction (3 balances accuracy/speed)."""

    similarity_threshold: float = 0.85
    """Minimum similarity (0-1) to consider items as similar (0.85 = 85% similar)."""

    min_items_to_fold: int = 3
    """Minimum items needed before attempting to fold (saves overhead on small lists)."""

    max_folded_refs_display: int = 10
    """Maximum number of individual refs to display before using range notation."""


class SimHash:
    """
    SimHash implementation for near-duplicate detection.

    SimHash is a locality-sensitive hashing technique that produces similar
    hash values for similar input texts. Unlike cryptographic hashes, small
    changes to the input result in small changes to the hash, making it
    ideal for detecting similarity.

    The algorithm:
    1. Extract n-gram features from the text
    2. Hash each feature to get a hash value
    3. For each bit position, sum +1 if the bit is 1, -1 if 0
    4. Final hash: bit is 1 if sum is positive, 0 otherwise

    Similarity is measured by Hamming distance between hashes.
    """

    def __init__(self, config: Optional[SimHashConfig] = None):
        """
        Initialize SimHash with configuration.

        Args:
            config: Optional SimHashConfig, uses defaults if not provided.
        """
        self.config = config or SimHashConfig()
        self._hash_cache: Dict[str, int] = {}

    def compute(self, text: str) -> int:
        """
        Compute SimHash fingerprint for given text.

        Args:
            text: Input text to hash.

        Returns:
            Integer SimHash fingerprint (64-bit by default).
        """
        # Check cache first
        cache_key = text[:1000]  # Limit cache key size
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        # Extract n-gram features
        ngrams = self._get_ngrams(text)

        if not ngrams:
            # Empty or very short text - return zero hash
            return 0

        # Initialize bit counters
        bit_counts = [0] * self.config.hash_bits

        # Process each n-gram
        for ngram in ngrams:
            # Get hash value for this n-gram
            ngram_hash = self._hash_ngram(ngram)

            # Update bit counters
            for i in range(self.config.hash_bits):
                if ngram_hash & (1 << i):
                    bit_counts[i] += 1
                else:
                    bit_counts[i] -= 1

        # Build final hash from bit counts
        fingerprint = 0
        for i in range(self.config.hash_bits):
            if bit_counts[i] > 0:
                fingerprint |= 1 << i

        # Cache the result
        self._hash_cache[cache_key] = fingerprint
        return fingerprint

    def _get_ngrams(self, text: str) -> List[str]:
        """
        Extract n-grams from text.

        N-grams are contiguous sequences of n words, used as features
        for similarity detection.

        Args:
            text: Input text to process.

        Returns:
            List of n-gram strings.
        """
        # Normalize: lowercase and extract words
        words = re.findall(r'\w+', text.lower())

        if len(words) < self.config.ngram_size:
            # Return single word tokens if text is too short for n-grams
            return words if words else []

        n = self.config.ngram_size
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def _hash_ngram(self, ngram: str) -> int:
        """
        Hash a single n-gram to an integer.

        Uses MD5 for distribution quality (not for security).

        Args:
            ngram: The n-gram string to hash.

        Returns:
            Integer hash value.
        """
        # Use MD5 for good distribution (security not needed here)
        digest = hashlib.md5(ngram.encode('utf-8')).digest()

        # Take first 8 bytes as 64-bit integer
        hash_value = int.from_bytes(digest[:8], byteorder='little')

        # Mask to configured bit width
        return hash_value & ((1 << self.config.hash_bits) - 1)

    def similarity(self, hash1: int, hash2: int) -> float:
        """
        Calculate similarity between two SimHashes.

        Similarity is based on Hamming distance - the number of
        differing bit positions between the two hashes.

        Args:
            hash1: First SimHash fingerprint.
            hash2: Second SimHash fingerprint.

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        if hash1 == hash2:
            return 1.0

        # XOR to find differing bits
        xor = hash1 ^ hash2

        # Count differing bits (Hamming distance)
        differing_bits = bin(xor).count('1')

        # Convert to similarity (1 - normalized distance)
        similarity = 1 - (differing_bits / self.config.hash_bits)
        return similarity

    def clear_cache(self) -> None:
        """Clear the hash cache to free memory."""
        self._hash_cache.clear()


@dataclass
class FoldedListItem:
    """
    Represents a potentially folded list item or group.

    When similar items are detected, they are grouped together with
    a representative item shown in full and the similar items summarized.
    """

    representative_item: str
    """The full content of the representative (first) item in the group."""

    representative_ref: str
    """The element reference ID for the representative item."""

    similar_count: int
    """Number of additional similar items folded into this group (0 if not folded)."""

    similar_refs: List[str] = field(default_factory=list)
    """List of element reference IDs for the similar items."""

    similarity_scores: List[float] = field(default_factory=list)
    """Similarity scores for each similar item (for debugging/analysis)."""

    def to_yaml(self, max_refs_display: int = 10) -> str:
        """
        Convert to YAML representation for accessibility tree output.

        Args:
            max_refs_display: Maximum refs to show before using range notation.

        Returns:
            YAML-formatted string representation.
        """
        if self.similar_count == 0:
            return self.representative_item

        refs_str = self._format_refs(max_refs_display)
        return f"{self.representative_item} (... and {self.similar_count} more similar) [refs: {refs_str}]"

    def to_compact(self) -> str:
        """
        Convert to minimal representation for maximum token savings.

        Returns:
            Compact string representation.
        """
        if self.similar_count == 0:
            return self.representative_item

        return f"{self.representative_item} (+{self.similar_count} similar)"

    def _format_refs(self, max_display: int = 10) -> str:
        """
        Format reference IDs, using range notation for many refs.

        Args:
            max_display: Maximum individual refs before using range.

        Returns:
            Formatted reference string.
        """
        if not self.similar_refs:
            return self.representative_ref

        if len(self.similar_refs) <= max_display:
            return ", ".join(self.similar_refs)

        # Try to detect sequential refs and use range notation
        # e.g., e234, e235, e236 -> e234-e236
        first = self.similar_refs[0]
        last = self.similar_refs[-1]

        # Check if refs follow a pattern like e123
        first_match = re.match(r'^([a-zA-Z]+)(\d+)$', first)
        last_match = re.match(r'^([a-zA-Z]+)(\d+)$', last)

        if first_match and last_match and first_match.group(1) == last_match.group(1):
            prefix = first_match.group(1)
            first_num = first_match.group(2)
            last_num = last_match.group(2)
            return f"{prefix}{first_num}-{prefix}{last_num}"

        # Fallback: show first and last with count
        return f"{first}...{last} ({len(self.similar_refs)} refs)"

    @property
    def total_items(self) -> int:
        """Total number of items represented by this group."""
        return 1 + self.similar_count

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio achieved by folding.

        Returns:
            Ratio of original items to folded representation (1.0 if not folded).
        """
        if self.similar_count == 0:
            return 1.0
        return self.total_items / 1.0  # 1 folded item represents total_items


@dataclass
class FoldingStats:
    """Statistics from a list folding operation."""

    original_items: int
    """Number of items before folding."""

    folded_groups: int
    """Number of groups after folding."""

    items_folded: int
    """Number of items that were folded into groups."""

    unique_items: int
    """Number of items that remained unfoldable."""

    estimated_token_reduction: float
    """Estimated token reduction percentage."""

    avg_similarity_in_groups: float
    """Average similarity score within folded groups."""

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio achieved."""
        if self.original_items == 0:
            return 1.0
        return self.original_items / self.folded_groups


class ListFoldingService:
    """
    Service for detecting and folding similar list items.

    This service processes lists of content items, identifies groups of
    similar items using SimHash, and creates folded representations
    that significantly reduce token count while preserving information.

    Usage:
        service = ListFoldingService()
        items = [("Product A - $10", "e1"), ("Product B - $12", "e2"), ...]
        folded = service.fold_list(items)
        # Returns FoldedListItem objects with similar items grouped
    """

    def __init__(self, config: Optional[SimHashConfig] = None):
        """
        Initialize the list folding service.

        Args:
            config: Optional SimHashConfig for tuning behavior.
        """
        self.config = config or SimHashConfig()
        self.simhash = SimHash(self.config)

    def fold_list(
        self,
        items: List[Tuple[str, str]],
        preserve_order: bool = True
    ) -> List[FoldedListItem]:
        """
        Fold similar items in a list.

        Args:
            items: List of (content, ref) tuples where content is the text
                   to analyze and ref is the element reference ID.
            preserve_order: If True, maintains original item order in output.

        Returns:
            List of FoldedListItem objects with similar items grouped.
        """
        if len(items) < self.config.min_items_to_fold:
            # Too few items to benefit from folding
            return [
                FoldedListItem(
                    representative_item=content,
                    representative_ref=ref,
                    similar_count=0,
                    similar_refs=[]
                )
                for content, ref in items
            ]

        # Compute SimHash for each item
        hashes = [
            (self.simhash.compute(content), content, ref)
            for content, ref in items
        ]

        # Group similar items
        groups = self._group_similar(hashes)

        # Convert groups to FoldedListItem objects
        folded_items = self._create_folded_items(groups)

        if preserve_order:
            # Sort by the position of the representative item in original list
            ref_to_position = {ref: i for i, (_, ref) in enumerate(items)}
            folded_items.sort(key=lambda f: ref_to_position.get(f.representative_ref, 0))

        return folded_items

    def fold_list_with_stats(
        self,
        items: List[Tuple[str, str]]
    ) -> Tuple[List[FoldedListItem], FoldingStats]:
        """
        Fold list items and return detailed statistics.

        Args:
            items: List of (content, ref) tuples.

        Returns:
            Tuple of (folded items list, folding statistics).
        """
        folded = self.fold_list(items)

        # Calculate statistics
        items_folded = sum(f.similar_count for f in folded)
        unique_items = sum(1 for f in folded if f.similar_count == 0)

        # Estimate token reduction
        # Assume average item is ~20 tokens, folded group is ~25 tokens
        original_tokens = len(items) * 20
        folded_tokens = sum(25 if f.similar_count > 0 else 20 for f in folded)
        token_reduction = (
            (original_tokens - folded_tokens) / original_tokens * 100
            if original_tokens > 0 else 0
        )

        # Calculate average similarity in groups
        all_similarities = []
        for f in folded:
            all_similarities.extend(f.similarity_scores)
        avg_similarity = (
            sum(all_similarities) / len(all_similarities)
            if all_similarities else 0
        )

        stats = FoldingStats(
            original_items=len(items),
            folded_groups=len(folded),
            items_folded=items_folded,
            unique_items=unique_items,
            estimated_token_reduction=token_reduction,
            avg_similarity_in_groups=avg_similarity
        )

        return folded, stats

    def _group_similar(
        self,
        hashes: List[Tuple[int, str, str]]
    ) -> List[List[Tuple[str, str, float]]]:
        """
        Group items by similarity based on SimHash.

        Uses a greedy algorithm that assigns each item to the first
        group it matches with sufficient similarity.

        Args:
            hashes: List of (simhash, content, ref) tuples.

        Returns:
            List of groups, where each group is a list of
            (content, ref, similarity_score) tuples.
        """
        groups: List[List[Tuple[str, str, float]]] = []
        group_hashes: List[int] = []  # Hash of each group's representative
        used: set = set()

        for i, (hash1, content1, ref1) in enumerate(hashes):
            if i in used:
                continue

            # Try to find an existing group this item belongs to
            matched_group = None
            for g_idx, g_hash in enumerate(group_hashes):
                sim = self.simhash.similarity(hash1, g_hash)
                if sim >= self.config.similarity_threshold:
                    matched_group = g_idx
                    groups[g_idx].append((content1, ref1, sim))
                    used.add(i)
                    break

            if matched_group is not None:
                continue

            # Start a new group with this item as representative
            group: List[Tuple[str, str, float]] = [(content1, ref1, 1.0)]
            used.add(i)

            # Find all similar items for this group
            for j, (hash2, content2, ref2) in enumerate(hashes[i + 1:], i + 1):
                if j in used:
                    continue

                sim = self.simhash.similarity(hash1, hash2)
                if sim >= self.config.similarity_threshold:
                    group.append((content2, ref2, sim))
                    used.add(j)

            groups.append(group)
            group_hashes.append(hash1)

        return groups

    def _create_folded_items(
        self,
        groups: List[List[Tuple[str, str, float]]]
    ) -> List[FoldedListItem]:
        """
        Convert similarity groups to FoldedListItem objects.

        Args:
            groups: List of groups from _group_similar.

        Returns:
            List of FoldedListItem objects.
        """
        result = []

        for group in groups:
            # First item is the representative
            representative_content, representative_ref, _ = group[0]

            # Remaining items are similar
            similar = group[1:]

            result.append(FoldedListItem(
                representative_item=representative_content,
                representative_ref=representative_ref,
                similar_count=len(similar),
                similar_refs=[ref for _, ref, _ in similar],
                similarity_scores=[sim for _, _, sim in similar]
            ))

        return result

    def estimate_compression(self, items: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Estimate compression without fully folding the list.

        Useful for deciding whether to apply folding.

        Args:
            items: List of (content, ref) tuples.

        Returns:
            Dictionary with compression estimates.
        """
        if len(items) < self.config.min_items_to_fold:
            return {
                "should_fold": False,
                "reason": f"Too few items ({len(items)} < {self.config.min_items_to_fold})",
                "estimated_groups": len(items),
                "estimated_compression_ratio": 1.0
            }

        # Quick pass: compute hashes and count potential groups
        hashes = [self.simhash.compute(content) for content, _ in items]

        # Estimate groups by checking representative samples
        sample_size = min(20, len(hashes))
        sample_indices = list(range(0, len(hashes), max(1, len(hashes) // sample_size)))

        groups_found = 0
        matched = set()

        for i in sample_indices:
            if i in matched:
                continue

            groups_found += 1
            matched.add(i)

            for j in range(i + 1, len(hashes)):
                if j in matched:
                    continue
                if self.simhash.similarity(hashes[i], hashes[j]) >= self.config.similarity_threshold:
                    matched.add(j)

        # Extrapolate if we sampled
        if sample_size < len(hashes):
            coverage = len(matched) / len(sample_indices)
            estimated_groups = int(groups_found + (1 - coverage) * (len(items) - len(matched)))
        else:
            estimated_groups = groups_found

        compression_ratio = len(items) / max(1, estimated_groups)

        return {
            "should_fold": compression_ratio > 1.5,
            "reason": "Good compression potential" if compression_ratio > 1.5 else "Limited compression potential",
            "estimated_groups": estimated_groups,
            "estimated_compression_ratio": compression_ratio,
            "items_analyzed": len(items)
        }


def fold_aria_list_items(
    aria_items: List[Dict[str, Any]],
    config: Optional[SimHashConfig] = None
) -> Tuple[List[FoldedListItem], FoldingStats]:
    """
    Convenience function to fold ARIA tree list items.

    Extracts content and ref from ARIA item dictionaries and applies folding.

    Args:
        aria_items: List of ARIA item dictionaries with 'content' and 'ref' keys.
        config: Optional SimHashConfig.

    Returns:
        Tuple of (folded items, statistics).
    """
    service = ListFoldingService(config)

    items = []
    for item in aria_items:
        content = item.get('content', item.get('name', item.get('text', str(item))))
        ref = item.get('ref', item.get('id', ''))
        items.append((str(content), str(ref)))

    return service.fold_list_with_stats(items)
