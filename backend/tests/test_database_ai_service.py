from __future__ import annotations

import unittest

from services import database_ai_service


class DatabaseAIServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.original_call_llm = database_ai_service.call_llm
        self.original_execute_sql = database_ai_service.execute_sql

    def tearDown(self) -> None:
        database_ai_service.call_llm = self.original_call_llm
        database_ai_service.execute_sql = self.original_execute_sql

    async def test_factual_question_returns_concise_answer_without_second_llm_call(self) -> None:
        prompts = []

        async def fake_call_llm(prompt: str) -> str:
            prompts.append(prompt)
            return "SELECT COUNT(*) AS employee_count FROM employees LIMIT 100"

        def fake_execute_sql(sql: str):
            self.assertEqual(sql, "SELECT COUNT(*) AS employee_count FROM employees LIMIT 100")
            return [{"employee_count": 42}]

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("How many employees?", {"employees": []})

        self.assertEqual(response["answer"], "There are 42 employees.")
        self.assertEqual(response["rows"], [{"employee_count": 42}])
        self.assertEqual(len(prompts), 1)

    async def test_average_question_returns_concise_scalar_answer(self) -> None:
        async def fake_call_llm(prompt: str) -> str:
            return "SELECT AVG(salary) AS average_salary FROM employees LIMIT 100"

        def fake_execute_sql(sql: str):
            return [{"average_salary": 50000.0}]

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("Average salary?", {"employees": []})

        self.assertEqual(response["answer"], "The average salary is 50,000.")

    async def test_list_question_returns_values_only(self) -> None:
        async def fake_call_llm(prompt: str) -> str:
            return "SELECT DISTINCT department FROM employees LIMIT 100"

        def fake_execute_sql(sql: str):
            return [{"department": "Sales"}, {"department": "Finance"}]

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("List departments.", {"employees": []})

        self.assertEqual(response["answer"], "The departments are: Sales, Finance.")

    async def test_executive_report_uses_board_ready_prompt(self) -> None:
        prompts = []

        async def fake_call_llm(prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return "SELECT department, COUNT(*) AS employees FROM employees GROUP BY department LIMIT 100"
            return "Answer\nBusiness Summary\nStatistics\nCharts\nInsights\nRecommendations\nConfidence Score\nExecution Time"

        def fake_execute_sql(sql: str):
            return [{"department": "Sales", "employees": 12}]

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("Create an executive report", {"employees": []})

        self.assertIn("Generate a board-ready answer", prompts[1])
        self.assertIn("Business Summary", response["answer"])
        self.assertEqual(len(prompts), 2)

    async def test_non_executive_non_factual_uses_concise_prompt(self) -> None:
        prompts = []

        async def fake_call_llm(prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return "SELECT department, COUNT(*) AS employees FROM employees GROUP BY department LIMIT 100"
            return "Sales has the most employees."

        def fake_execute_sql(sql: str):
            return [{"department": "Sales", "employees": 12}]

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("Compare employees by department", {"employees": []})

        self.assertIn("Do not generate an executive report", prompts[1])
        self.assertNotIn("Generate a board-ready answer", prompts[1])
        self.assertEqual(response["answer"], "Sales has the most employees.")

    async def test_read_only_protection_is_unchanged(self) -> None:
        async def fake_call_llm(prompt: str) -> str:
            return "DELETE FROM employees"

        def fake_execute_sql(sql: str):
            raise AssertionError("execute_sql should not be called for non-read-only SQL")

        database_ai_service.call_llm = fake_call_llm
        database_ai_service.execute_sql = fake_execute_sql

        response = await database_ai_service.ask_database("Delete employees", {"employees": []})

        self.assertEqual(response["answer"], "Only read-only SQL queries are allowed.")
        self.assertEqual(response["rows"], [])


class DatabaseAIIntentTests(unittest.TestCase):
    def test_detect_response_intent(self) -> None:
        self.assertEqual(database_ai_service.detect_response_intent("How many employees?"), "factual")
        self.assertEqual(database_ai_service.detect_response_intent("Highest revenue?"), "factual")
        self.assertEqual(database_ai_service.detect_response_intent("List departments"), "factual")
        self.assertEqual(database_ai_service.detect_response_intent("Create an executive report"), "executive_report")
        self.assertEqual(database_ai_service.detect_response_intent("Prepare a management summary"), "executive_report")
        self.assertEqual(database_ai_service.detect_response_intent("Compare revenue by region"), "concise")


if __name__ == "__main__":
    unittest.main()
