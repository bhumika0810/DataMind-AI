from __future__ import annotations

import unittest

from services.analytics_engine.intent import SUPPORTED_INTENTS, detect_intent, detect_intents


class IntentClassifierTests(unittest.TestCase):
    def assertIntent(self, question: str, expected: str) -> None:
        intent = detect_intent(question)
        self.assertEqual(intent, expected)
        self.assertIsInstance(intent, str)
        self.assertIn(intent, SUPPORTED_INTENTS)
        self.assertEqual(detect_intents(question), expected)

    def test_quick_answer_for_row_and_column_counts(self) -> None:
        self.assertIntent("How many rows are in this dataset?", "quick_answer")
        self.assertIntent("What is the column count?", "quick_answer")
        self.assertIntent("Number of records?", "quick_answer")

    def test_quick_answer_for_data_quality_facts(self) -> None:
        self.assertIntent("How many missing values are there?", "quick_answer")
        self.assertIntent("Show duplicate rows count", "quick_answer")
        self.assertIntent("What are the data types?", "quick_answer")

    def test_quick_answer_for_numeric_and_column_facts(self) -> None:
        self.assertIntent("What is the average revenue?", "quick_answer")
        self.assertIntent("Give me the maximum salary", "quick_answer")
        self.assertIntent("List columns", "quick_answer")
        self.assertIntent("Show unique values for region", "quick_answer")

    def test_factual_questions_override_other_modes(self) -> None:
        self.assertIntent("How many rows for the executive summary?", "quick_answer")
        self.assertIntent("Plot the average revenue", "quick_answer")
        self.assertIntent("Recommend the maximum discount", "quick_answer")
        self.assertIntent("Analyze missing values", "quick_answer")

    def test_executive_report_requires_explicit_report_phrase(self) -> None:
        self.assertIntent("Create an executive summary for leadership", "executive_report")
        self.assertIntent("Prepare a business report", "executive_report")
        self.assertIntent("Draft a board report", "executive_report")
        self.assertIntent("Write a management summary", "executive_report")

    def test_executive_words_alone_do_not_trigger_report(self) -> None:
        self.assertIntent("What should leadership know?", "quick_answer")
        self.assertIntent("Summarize management risk", "analysis")

    def test_visualization_only_for_chart_graph_plot_requests(self) -> None:
        self.assertIntent("Create a chart of sales by region", "visualization")
        self.assertIntent("Show a graph of revenue over time", "visualization")
        self.assertIntent("Plot the distribution of salary", "visualization")
        self.assertIntent("Visualize profit by product", "visualization")

    def test_analysis_for_explicit_analytical_language(self) -> None:
        self.assertIntent("Analyze revenue by month", "analysis")
        self.assertIntent("Explain customer churn", "analysis")
        self.assertIntent("Show the trend in sales", "analysis")
        self.assertIntent("Compare sales by region", "analysis")
        self.assertIntent("Find the correlation between price and demand", "analysis")
        self.assertIntent("Summarize the dataset", "analysis")

    def test_recommendation_only_for_recommendation_language(self) -> None:
        self.assertIntent("Recommend improvements for sales", "recommendation")
        self.assertIntent("Give suggestions to improve margin", "recommendation")
        self.assertIntent("How can we optimize cost?", "recommendation")
        self.assertIntent("Suggest improvements for data quality", "recommendation")

    def test_exactly_one_supported_intent_is_returned_for_unknown_questions(self) -> None:
        self.assertIntent("Tell me about this file", "quick_answer")
        self.assertEqual(len({detect_intent("Create a chart and recommend improvements")}), 1)


if __name__ == "__main__":
    unittest.main()
