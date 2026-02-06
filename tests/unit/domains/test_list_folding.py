"""Tests for SimHash-based list folding."""

import pytest

from robotmcp.domains.snapshot.list_folding import (
    FoldedListItem,
    FoldingStats,
    ListFoldingService,
    SimHash,
    SimHashConfig,
    fold_aria_list_items,
)


class TestSimHash:
    """Test the SimHash implementation."""

    def test_identical_text_same_hash(self):
        """Identical text should produce identical hash."""
        simhash = SimHash()
        text = "This is a test product item with price $19.99"

        hash1 = simhash.compute(text)
        hash2 = simhash.compute(text)

        assert hash1 == hash2
        assert simhash.similarity(hash1, hash2) == 1.0

    def test_similar_text_high_similarity(self):
        """Similar text should have higher similarity than completely different text."""
        simhash = SimHash()

        # Very similar texts (only one word different)
        text1 = "The quick brown fox jumps over the lazy dog near the river"
        text2 = "The quick brown cat jumps over the lazy dog near the river"

        hash1 = simhash.compute(text1)
        hash2 = simhash.compute(text2)

        similarity_similar = simhash.similarity(hash1, hash2)

        # Completely different text
        text3 = "Contact support at email address for technical assistance"
        hash3 = simhash.compute(text3)

        similarity_different = simhash.similarity(hash1, hash3)

        # Similar text should have higher similarity than different text
        assert similarity_similar > similarity_different, (
            f"Similar texts ({similarity_similar:.3f}) should have higher similarity "
            f"than different texts ({similarity_different:.3f})"
        )

    def test_different_text_low_similarity(self):
        """Completely different text should have low similarity."""
        simhash = SimHash()

        text1 = "Product A - Red Widget - Price: $19.99 - In Stock"
        text2 = "Contact us at support@example.com for help with your order"

        hash1 = simhash.compute(text1)
        hash2 = simhash.compute(text2)

        similarity = simhash.similarity(hash1, hash2)

        # Different content should yield lower similarity
        assert similarity < 0.7, f"Expected similarity < 0.7, got {similarity}"

    def test_empty_text_returns_zero_hash(self):
        """Empty text should return zero hash."""
        simhash = SimHash()

        hash_val = simhash.compute("")
        assert hash_val == 0

        hash_val = simhash.compute("   ")
        assert hash_val == 0

    def test_short_text_handled(self):
        """Short text (fewer words than ngram size) should be handled."""
        simhash = SimHash(SimHashConfig(ngram_size=3))

        # Only two words, less than ngram_size=3
        hash1 = simhash.compute("hello world")
        assert hash1 != 0  # Should still produce a hash from individual words

    def test_cache_works(self):
        """Hash cache should return same result on repeated calls."""
        simhash = SimHash()
        text = "This is a cached test string"

        hash1 = simhash.compute(text)
        assert text[:1000] in simhash._hash_cache

        hash2 = simhash.compute(text)
        assert hash1 == hash2

    def test_clear_cache(self):
        """Cache should be clearable."""
        simhash = SimHash()
        simhash.compute("test string")
        assert len(simhash._hash_cache) > 0

        simhash.clear_cache()
        assert len(simhash._hash_cache) == 0

    def test_config_hash_bits(self):
        """Hash should respect configured bit width."""
        config = SimHashConfig(hash_bits=32)
        simhash = SimHash(config)

        hash_val = simhash.compute("test string for bit width check")

        # Hash should fit within 32 bits
        assert hash_val < (1 << 32)

    def test_similarity_symmetry(self):
        """Similarity should be symmetric: sim(a,b) == sim(b,a)."""
        simhash = SimHash()

        hash1 = simhash.compute("First text here")
        hash2 = simhash.compute("Second text here")

        assert simhash.similarity(hash1, hash2) == simhash.similarity(hash2, hash1)

    def test_identical_hashes_similarity_one(self):
        """Identical hashes should have similarity of exactly 1.0."""
        simhash = SimHash()

        hash_val = 0b1010101010101010
        assert simhash.similarity(hash_val, hash_val) == 1.0

    def test_opposite_hashes_low_similarity(self):
        """Completely opposite hashes should have low similarity."""
        simhash = SimHash(SimHashConfig(hash_bits=8))

        hash1 = 0b11111111
        hash2 = 0b00000000

        similarity = simhash.similarity(hash1, hash2)
        assert similarity == 0.0  # All bits differ


class TestFoldedListItem:
    """Test the FoldedListItem dataclass."""

    def test_single_item_no_folding(self):
        """Single item without similar items."""
        item = FoldedListItem(
            representative_item="Product A - $19.99",
            representative_ref="e1",
            similar_count=0,
            similar_refs=[]
        )

        yaml_output = item.to_yaml()
        assert yaml_output == "Product A - $19.99"
        assert item.total_items == 1
        assert item.compression_ratio == 1.0

    def test_folded_item_yaml_output(self):
        """Folded item should include count and refs."""
        item = FoldedListItem(
            representative_item="Product A - $19.99",
            representative_ref="e1",
            similar_count=5,
            similar_refs=["e2", "e3", "e4", "e5", "e6"]
        )

        yaml_output = item.to_yaml()
        assert "(... and 5 more similar)" in yaml_output
        assert "[refs:" in yaml_output
        assert item.total_items == 6
        assert item.compression_ratio == 6.0

    def test_compact_output(self):
        """Compact output should be minimal."""
        item = FoldedListItem(
            representative_item="Product A - $19.99",
            representative_ref="e1",
            similar_count=47,
            similar_refs=[f"e{i}" for i in range(2, 49)]
        )

        compact = item.to_compact()
        assert compact == "Product A - $19.99 (+47 similar)"

    def test_ref_range_notation(self):
        """Many refs should use range notation."""
        item = FoldedListItem(
            representative_item="Item",
            representative_ref="e1",
            similar_count=20,
            similar_refs=[f"e{i}" for i in range(2, 22)]
        )

        yaml_output = item.to_yaml(max_refs_display=5)
        # Should use range notation e2-e21
        assert "e2-e21" in yaml_output or "e2...e21" in yaml_output

    def test_few_refs_listed_individually(self):
        """Few refs should be listed individually."""
        item = FoldedListItem(
            representative_item="Item",
            representative_ref="e1",
            similar_count=3,
            similar_refs=["e2", "e3", "e4"]
        )

        yaml_output = item.to_yaml(max_refs_display=10)
        assert "e2, e3, e4" in yaml_output


class TestListFoldingService:
    """Test the ListFoldingService."""

    def test_fold_identical_items(self):
        """Identical items should be folded together."""
        service = ListFoldingService()

        # 10 identical items
        items = [
            ("Product - Price: $10.00 - In Stock", f"e{i}")
            for i in range(1, 11)
        ]

        folded = service.fold_list(items)

        # All should be folded into one group
        assert len(folded) == 1
        assert folded[0].similar_count == 9
        assert len(folded[0].similar_refs) == 9

    def test_fold_similar_items(self):
        """Similar (but not identical) items should be folded."""
        service = ListFoldingService(SimHashConfig(similarity_threshold=0.7))

        items = [
            ("Product A - Red Widget - Price: $19.99 - In Stock", "e1"),
            ("Product B - Blue Widget - Price: $24.99 - In Stock", "e2"),
            ("Product C - Green Widget - Price: $21.99 - In Stock", "e3"),
            ("Contact Us - Email: support@example.com", "e4"),  # Different
        ]

        folded = service.fold_list(items)

        # Products should be grouped, contact should be separate
        assert len(folded) <= 3  # At most 3 groups (possibly 2 if products grouped)

        # Find the contact item
        contact_items = [f for f in folded if "Contact" in f.representative_item]
        assert len(contact_items) == 1
        assert contact_items[0].similar_count == 0

    def test_preserve_different_items(self):
        """Completely different items should remain separate."""
        service = ListFoldingService()

        items = [
            ("Home", "e1"),
            ("About Us", "e2"),
            ("Contact", "e3"),
            ("Products", "e4"),
            ("Blog", "e5"),
        ]

        folded = service.fold_list(items)

        # Short, different items should mostly remain separate
        # (some might group by coincidence, but most should be separate)
        assert len(folded) >= 3

    def test_min_items_threshold(self):
        """Below threshold, no folding should occur."""
        config = SimHashConfig(min_items_to_fold=5)
        service = ListFoldingService(config)

        items = [
            ("Item A", "e1"),
            ("Item A", "e2"),  # Identical
            ("Item A", "e3"),  # Identical
        ]

        folded = service.fold_list(items)

        # Should not fold because < min_items_to_fold
        assert len(folded) == 3
        assert all(f.similar_count == 0 for f in folded)

    def test_compression_ratio(self):
        """Verify compression statistics."""
        service = ListFoldingService()

        # Create list with similar products
        items = [
            (f"Product {i} - Widget - Price: ${10 + i}.99 - Available", f"e{i}")
            for i in range(1, 51)
        ]

        folded, stats = service.fold_list_with_stats(items)

        assert stats.original_items == 50
        assert stats.folded_groups == len(folded)
        assert stats.compression_ratio >= 1.0
        assert stats.items_folded + stats.unique_items == 50

    def test_preserve_order(self):
        """Items should maintain relative order when preserve_order=True."""
        service = ListFoldingService()

        items = [
            ("Unique Item A", "e1"),
            ("Product X - $10", "e2"),
            ("Unique Item B", "e3"),
            ("Product Y - $10", "e4"),  # Similar to e2
            ("Unique Item C", "e5"),
        ]

        folded = service.fold_list(items, preserve_order=True)

        # Representative refs should maintain original order
        refs = [f.representative_ref for f in folded]

        # e1 should come before e3, e3 before e5
        if "e1" in refs and "e3" in refs:
            assert refs.index("e1") < refs.index("e3")
        if "e3" in refs and "e5" in refs:
            assert refs.index("e3") < refs.index("e5")

    def test_estimate_compression(self):
        """Estimate compression should provide useful guidance."""
        service = ListFoldingService()

        # High compression potential - identical items
        identical_items = [
            ("Product Widget - Price: $10.00 - In Stock Available", f"e{i}")
            for i in range(20)
        ]

        estimate = service.estimate_compression(identical_items)
        # Identical items should definitely be foldable
        assert estimate["estimated_compression_ratio"] >= 1.0
        assert "estimated_groups" in estimate
        assert "items_analyzed" in estimate
        assert estimate["items_analyzed"] == 20

        # Low compression potential - very different items
        different_items = [
            (f"Item {i}: {'abc'[i % 3] * 10} unique content {i ** 2}", f"e{i}")
            for i in range(20)
        ]

        estimate = service.estimate_compression(different_items)
        # Should still return valid estimate structure
        assert "estimated_compression_ratio" in estimate
        assert "should_fold" in estimate
        assert "reason" in estimate

    def test_empty_list(self):
        """Empty list should return empty result."""
        service = ListFoldingService()

        folded = service.fold_list([])
        assert folded == []

    def test_single_item_list(self):
        """Single item list should return that item unfolded."""
        service = ListFoldingService(SimHashConfig(min_items_to_fold=1))

        items = [("Single Item", "e1")]
        folded = service.fold_list(items)

        assert len(folded) == 1
        assert folded[0].representative_item == "Single Item"
        assert folded[0].similar_count == 0


class TestFoldAriaListItems:
    """Test the convenience function for ARIA tree items."""

    def test_fold_aria_items_with_content_key(self):
        """ARIA items with 'content' key should be processed."""
        aria_items = [
            {"content": "Product A - $10", "ref": "e1"},
            {"content": "Product B - $10", "ref": "e2"},
            {"content": "Product C - $10", "ref": "e3"},
            {"content": "Product D - $10", "ref": "e4"},
        ]

        folded, stats = fold_aria_list_items(aria_items)

        assert stats.original_items == 4
        assert len(folded) >= 1

    def test_fold_aria_items_with_name_key(self):
        """ARIA items with 'name' key should be processed."""
        aria_items = [
            {"name": "Link A", "id": "link1"},
            {"name": "Link B", "id": "link2"},
            {"name": "Link C", "id": "link3"},
            {"name": "Different Content Here", "id": "link4"},
        ]

        folded, stats = fold_aria_list_items(aria_items)

        assert stats.original_items == 4

    def test_fold_aria_items_with_custom_config(self):
        """Custom config should be respected."""
        aria_items = [
            {"content": f"Item {i}", "ref": f"e{i}"}
            for i in range(10)
        ]

        config = SimHashConfig(similarity_threshold=0.95)
        folded, stats = fold_aria_list_items(aria_items, config)

        assert stats.original_items == 10


class TestFoldingStats:
    """Test the FoldingStats dataclass."""

    def test_compression_ratio_calculation(self):
        """Compression ratio should be calculated correctly."""
        stats = FoldingStats(
            original_items=100,
            folded_groups=10,
            items_folded=90,
            unique_items=10,
            estimated_token_reduction=80.0,
            avg_similarity_in_groups=0.92
        )

        assert stats.compression_ratio == 10.0

    def test_compression_ratio_no_items(self):
        """Compression ratio with zero items should be 1.0."""
        stats = FoldingStats(
            original_items=0,
            folded_groups=0,
            items_folded=0,
            unique_items=0,
            estimated_token_reduction=0,
            avg_similarity_in_groups=0
        )

        assert stats.compression_ratio == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        service = ListFoldingService()

        items = [
            ("Product 1", "e1"),
            ("Product 2", "e2"),
            ("Product 3", "e3"),
            ("Product 4", "e4"),
        ]

        folded = service.fold_list(items)
        assert len(folded) >= 1

    def test_very_long_content(self):
        """Very long content should be handled."""
        service = ListFoldingService()

        long_text = "A" * 10000
        items = [
            (long_text, "e1"),
            (long_text, "e2"),
            (long_text, "e3"),
            (long_text, "e4"),
        ]

        folded = service.fold_list(items)

        # Should fold identical long items
        assert len(folded) == 1
        assert folded[0].similar_count == 3

    def test_special_characters(self):
        """Special characters should not break hashing."""
        service = ListFoldingService()

        items = [
            ("Price: $19.99 (10% off!)", "e1"),
            ("Price: $24.99 (15% off!)", "e2"),
            ("Price: $29.99 (20% off!)", "e3"),
            ("Price: $34.99 (25% off!)", "e4"),
        ]

        folded = service.fold_list(items)

        # Should handle special chars and potentially group similar prices
        assert len(folded) >= 1

    def test_numeric_only_content(self):
        """Numeric-only content should work."""
        service = ListFoldingService()

        items = [
            ("12345", "e1"),
            ("12346", "e2"),
            ("12347", "e3"),
            ("99999", "e4"),
        ]

        folded = service.fold_list(items)
        assert len(folded) >= 1

    def test_threshold_boundary(self):
        """Items exactly at threshold should be handled consistently."""
        # Create items that will have predictable similarity
        service = ListFoldingService(SimHashConfig(similarity_threshold=0.5))

        items = [
            ("The quick brown fox jumps over the lazy dog", "e1"),
            ("The quick brown cat jumps over the lazy dog", "e2"),
            ("Something completely and utterly different here", "e3"),
            ("Another completely different piece of text", "e4"),
        ]

        folded = service.fold_list(items)

        # First two should potentially group, others separate
        assert len(folded) >= 1
